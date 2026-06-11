// Package agent runs one user turn as a loop of model steps until the model
// stops asking for tools, executing tool calls sequentially in emission order
// and upholding the transcript invariant after every mutation (design §8, §4).
package agent

import (
	"context"
	"errors"
	"fmt"

	"harness/internal/llm"
	"harness/internal/tools"
)

// defaultMaxSteps caps model round-trips per user turn (design §8.1).
const defaultMaxSteps = 50

// EventSink receives the turn's observable events for rendering. The agent loop
// owns the transcript and the control flow; the sink only reports. Phase 10's
// renderer implements it (design §8.1, §10).
type EventSink interface {
	TextDelta(text string)            // incremental assistant text
	ToolStart(call llm.ToolCall)      // a tool call is about to run
	ToolResult(result llm.ToolResult) // a tool call finished
	Notice(msg string)                // out-of-band notices (max-steps, cancelled)
	TurnComplete(usage TurnUsage)     // end of the turn
}

// TurnUsage is the per-turn summary handed to the sink (design §10 usage line).
type TurnUsage struct {
	Steps int
	Usage llm.Usage
}

// Options configures an Agent. The zero value is valid; MaxSteps falls back to
// the default.
type Options struct {
	MaxSteps int
	// Model is the resolved model id stamped onto every request. The agent loop
	// owns Request.Model because the provider config carries no model (one
	// provider can serve many models); main injects the resolved value here.
	Model string
	// ContextWindow is the resolved -context-window override (tokens). When
	// positive it drives the compaction trigger and degradation budget instead of
	// the model registry's window; zero means "use the registry default" (design
	// §6, §12). Plumbing it here is what makes the override actually move the
	// trigger for unknown/local models whose real window differs from the default
	// default.
	ContextWindow int
	// Registry supplies model context windows and pricing loaded from provider
	// config files.
	Registry *llm.Registry
}

// Agent drives the turn loop against one provider and tool registry, owning the
// running transcript.
type Agent struct {
	provider      llm.Provider
	tools         *tools.Registry
	registry      *llm.Registry
	transcript    []llm.Message
	system        string
	model         string
	maxSteps      int
	contextWindow int // -context-window override; 0 = use the registry default
}

// New constructs an Agent. A non-positive Options.MaxSteps uses the default.
func New(provider llm.Provider, registry *tools.Registry, opts Options) *Agent {
	maxSteps := opts.MaxSteps
	if maxSteps <= 0 {
		maxSteps = defaultMaxSteps
	}
	modelRegistry := opts.Registry
	if modelRegistry == nil {
		modelRegistry = llm.NewRegistry(nil)
	}
	return &Agent{
		provider:      provider,
		tools:         registry,
		registry:      modelRegistry,
		model:         opts.Model,
		maxSteps:      maxSteps,
		contextWindow: opts.ContextWindow,
	}
}

// window returns the context window the compaction trigger and degradation
// budget should use: the resolved -context-window override when positive,
// otherwise the model registry's window (256k by default for unknown models). This is what
// honors the §6 "overridable with -context-window" promise in the §12 trigger.
func (a *Agent) window() int {
	if a.contextWindow > 0 {
		return a.contextWindow
	}
	return a.registry.ContextWindow(a.model)
}

// SetSystem sets the system prompt sent on every request.
func (a *Agent) SetSystem(system string) { a.system = system }

// SetProvider replaces the provider used for subsequent model calls.
func (a *Agent) SetProvider(provider llm.Provider) {
	if provider != nil {
		a.provider = provider
	}
}

// SetModel replaces the model id stamped onto subsequent requests. contextWindow
// is the same override as Options.ContextWindow: zero means use the registry.
func (a *Agent) SetModel(model string, contextWindow int) {
	a.model = model
	a.contextWindow = contextWindow
}

// SetTranscript replaces the running transcript (used when resuming a session).
func (a *Agent) SetTranscript(msgs []llm.Message) { a.transcript = msgs }

// Transcript returns the current transcript. The slice is owned by the Agent;
// callers must not mutate it.
func (a *Agent) Transcript() []llm.Message { return a.transcript }

// stepResult holds what one stream produced after assembly.
type stepResult struct {
	text       string
	toolCalls  []llm.ToolCall
	usage      llm.Usage
	stopReason llm.StopReason
}

// RunTurn appends the user message, then loops model steps until the model
// stops requesting tools or the step budget is hit (design §8.1). Cancellation
// mid-stream applies the §4 cancel repair and returns ctx.Err(); the transcript
// is left valid (re-sendable) in every exit path.
func (a *Agent) RunTurn(ctx context.Context, userText string, sink EventSink) error {
	a.transcript = append(a.transcript, llm.Message{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: userText}},
	})

	var total llm.Usage
	var lastInput int // input tokens the final step reported (drives the trigger)
	steps := 0

	for steps < a.maxSteps {
		req := llm.Request{
			Model:    a.model,
			System:   a.system,
			Messages: a.transcript,
			Tools:    a.tools.Specs(),
		}

		res, err := a.stream(ctx, req, sink)
		steps++
		total = add(total, res.usage)
		lastInput = res.usage.InputTokens

		if err != nil {
			// Cancellation repair: keep streamed partial text as a text-only
			// assistant message; drop the message entirely if nothing streamed.
			// Un-executed tool calls are never appended.
			if res.text != "" {
				a.transcript = append(a.transcript, llm.Message{
					Role:    llm.RoleAssistant,
					Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: res.text}},
				})
			}
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				sink.Notice("[cancelled]")
			}
			sink.TurnComplete(TurnUsage{Steps: steps, Usage: total})
			return err
		}

		a.transcript = append(a.transcript, assistantMessage(res))

		if res.stopReason != llm.StopToolUse {
			break
		}

		results := make([]llm.ContentBlock, 0, len(res.toolCalls))
		for _, call := range res.toolCalls {
			sink.ToolStart(call)
			result := a.tools.Dispatch(ctx, call)
			sink.ToolResult(result)
			results = append(results, llm.ContentBlock{
				Kind:        llm.BlockToolResult,
				ResultForID: result.ForID,
				ResultText:  result.Text,
				ResultError: result.IsError,
			})
		}
		a.transcript = append(a.transcript, llm.Message{
			Role:    llm.RoleUser,
			Content: results,
		})

		if steps >= a.maxSteps {
			sink.Notice(maxStepsNotice(a.maxSteps))
			break
		}
	}

	// Post-turn compaction trigger (design §12, §8.1): fires after the turn
	// completes, before returning to the prompt. The summary call's usage folds
	// into the turn total so session totals (via the sink) include compaction. A
	// compaction error never fails the turn — the warning was already reported and
	// the transcript was kept intact.
	if compUsage, err := a.MaybeCompact(ctx, lastInput, sink); err == nil {
		total = add(total, compUsage)
	}

	sink.TurnComplete(TurnUsage{Steps: steps, Usage: total})
	return nil
}

// stream consumes one provider stream: it forwards text deltas to the sink,
// assembles completed tool calls in emission order, and captures the final
// usage and stop reason. A terminal stream error is returned with whatever
// partial text streamed so far (for cancel repair).
func (a *Agent) stream(ctx context.Context, req llm.Request, sink EventSink) (stepResult, error) {
	var res stepResult
	var text []byte

	for ev, err := range a.provider.Stream(ctx, req) {
		if err != nil {
			res.text = string(text)
			return res, err
		}
		switch ev.Kind {
		case llm.EventTextDelta:
			text = append(text, ev.Text...)
			sink.TextDelta(ev.Text)
		case llm.EventToolCallDone:
			res.toolCalls = append(res.toolCalls, llm.ToolCall{
				ID:    ev.ToolID,
				Name:  ev.ToolName,
				Input: ev.ToolInput,
			})
		case llm.EventUsage:
			if ev.Usage != nil {
				res.usage = *ev.Usage
			}
		case llm.EventDone:
			if ev.Usage != nil {
				res.usage = *ev.Usage
			}
			res.stopReason = ev.StopReason
		}
	}

	res.text = string(text)
	return res, nil
}

// assistantMessage builds the assistant message for a completed step: the text
// block (if any) first, then tool_use blocks in emission order (design §8.1).
func assistantMessage(res stepResult) llm.Message {
	content := make([]llm.ContentBlock, 0, 1+len(res.toolCalls))
	if res.text != "" {
		content = append(content, llm.ContentBlock{Kind: llm.BlockText, Text: res.text})
	}
	for _, call := range res.toolCalls {
		content = append(content, llm.ContentBlock{
			Kind:      llm.BlockToolUse,
			ToolUseID: call.ID,
			ToolName:  call.Name,
			ToolInput: call.Input,
		})
	}
	return llm.Message{Role: llm.RoleAssistant, Content: content}
}

// maxStepsNotice is the exact guard message printed when the step budget is
// exhausted (design §8.1).
func maxStepsNotice(maxSteps int) string {
	return fmt.Sprintf("[stopped: reached max steps (%d); say \"continue\" to keep going]", maxSteps)
}

func add(a, b llm.Usage) llm.Usage {
	return llm.Usage{
		InputTokens:      a.InputTokens + b.InputTokens,
		OutputTokens:     a.OutputTokens + b.OutputTokens,
		CacheReadTokens:  a.CacheReadTokens + b.CacheReadTokens,
		CacheWriteTokens: a.CacheWriteTokens + b.CacheWriteTokens,
	}
}
