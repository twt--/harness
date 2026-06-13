// Package agent runs one user turn as a loop of model steps until the model
// stops asking for tools, executing each step's tool calls in emission order
// (concurrently when they are all read-only) and upholding the transcript
// invariant after every mutation (design §8, §4).
package agent

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"harness/internal/llm"
	"harness/internal/retry"
	"harness/internal/tools"
)

// defaultMaxSteps caps model round-trips per user turn (design §8.1).
const defaultMaxSteps = 50

// streamRetries is the per-step mid-stream retry budget: a step whose stream
// fails after the first byte may be re-requested this many times (spec §2).
// Retries do not consume the maxSteps budget.
const streamRetries = 2

// maxAutoContinues bounds AutoContinue budgets per turn so a pathological
// loop still terminates (spec §5).
const maxAutoContinues = 3

// maxParallelTools bounds concurrent read-only dispatch (spec §8).
const maxParallelTools = 8

// EventSink receives the turn's observable events for rendering. The agent loop
// owns the transcript and the control flow; the sink only reports. Phase 10's
// renderer implements it (design §8.1, §10).
type EventSink interface {
	TextDelta(text string) // incremental assistant text
	ModelStepStart(step, attempt int, ctx ContextEstimate)
	ToolUseStart(call llm.ToolCall)
	ToolUseDelta(index int, delta string)
	ToolStart(call llm.ToolCall)      // a tool call is about to run
	ToolResult(result llm.ToolResult) // a tool call finished
	Notice(msg string)                // out-of-band notices (max-steps, cancelled)
	TurnComplete(usage TurnUsage)     // end of the turn
}

// TurnUsage is the per-turn summary handed to the sink (design §10 usage line).
type TurnUsage struct {
	Steps   int
	Usage   llm.Usage
	Context ContextEstimate
}

// ContextEstimate is a coarse request-footprint estimate for UI diagnostics.
type ContextEstimate struct {
	Total    int
	Window   int
	System   int
	Tools    int
	Messages int
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
	// Reasoning is forwarded to every model request. Empty means provider
	// default.
	Reasoning llm.ReasoningConfig
	// AutoContinue grants up to maxAutoContinues fresh step budgets when the
	// model still wants tools at the cap, instead of stopping (spec §5).
	AutoContinue bool
	// CompactKeepTurns controls how many whole recent turns remain verbatim after
	// compaction. Zero uses the default.
	CompactKeepTurns int
	// CompactSummaryMaxTokens caps summarization output. Zero uses the default.
	CompactSummaryMaxTokens int
	// CompactToolResultMaxBytes caps old tool-result bodies before they are sent
	// to the summarizer. Zero uses the default; negative disables this pre-pass.
	CompactToolResultMaxBytes int
}

// Agent drives the turn loop against one provider and tool registry, owning the
// running transcript.
type Agent struct {
	provider                  llm.Provider
	tools                     *tools.Registry
	registry                  *llm.Registry
	transcript                []llm.Message
	system                    string
	model                     string
	maxSteps                  int
	contextWindow             int // -context-window override; 0 = use the registry default
	reasoning                 llm.ReasoningConfig
	autoContinue              bool                // grant fresh step budgets at the cap (spec §5)
	sleep                     func(time.Duration) // mid-stream retry backoff; nil-free, set in New
	compactKeepTurns          int
	compactSummaryMaxTokens   int
	compactToolResultMaxBytes int
	archiveCompaction         CompactionArchiver
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
		provider:                  provider,
		tools:                     registry,
		registry:                  modelRegistry,
		model:                     opts.Model,
		maxSteps:                  maxSteps,
		contextWindow:             opts.ContextWindow,
		reasoning:                 opts.Reasoning,
		autoContinue:              opts.AutoContinue,
		sleep:                     time.Sleep,
		compactKeepTurns:          opts.CompactKeepTurns,
		compactSummaryMaxTokens:   opts.CompactSummaryMaxTokens,
		compactToolResultMaxBytes: opts.CompactToolResultMaxBytes,
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

// ToolNames returns the names of tools in the agent's active registry in
// registration order.
func (a *Agent) ToolNames() []string { return a.tools.Names() }

// ToolSpecs returns the model-facing tool specs in registration order.
func (a *Agent) ToolSpecs() []llm.ToolSchema { return a.tools.Specs() }

// SetTools replaces the tool registry used for subsequent requests. Because the
// agent advertises (Specs) and dispatches from the same registry, swapping it
// changes both what the model sees and what it can call — the hook a run-mode
// switch uses. A nil registry is ignored.
func (a *Agent) SetTools(registry *tools.Registry) {
	if registry != nil {
		a.tools = registry
	}
}

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

// SetSleep replaces the mid-stream retry backoff function. Tests inject a no-op
// to keep the loop free of real time; a nil argument is ignored.
func (a *Agent) SetSleep(sleep func(time.Duration)) {
	if sleep != nil {
		a.sleep = sleep
	}
}

// SetCompactionArchiver installs the callback used to preserve raw messages
// removed from the active transcript. A nil callback disables archiving.
func (a *Agent) SetCompactionArchiver(archive CompactionArchiver) {
	a.archiveCompaction = archive
}

// Transcript returns the current transcript. The slice is owned by the Agent;
// callers must not mutate it.
func (a *Agent) Transcript() []llm.Message { return a.transcript }

// EstimateContext estimates the next request footprint using the current
// transcript, system prompt, and advertised tools.
func (a *Agent) EstimateContext() ContextEstimate {
	return estimateRequest(llm.Request{
		System:   a.system,
		Messages: a.transcript,
		Tools:    a.tools.Specs(),
	}, a.window())
}

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
	a.transcript = append(a.transcript, textMessage(llm.RoleUser, userText))

	var total llm.Usage
	var lastInput int // input tokens the final step reported (drives the trigger)
	var lastContext ContextEstimate
	steps := 0
	budget := a.maxSteps
	continues := 0

	for steps < budget {
		lastContext = a.EstimateContext()
		// Proactive trigger (spec §4): a turn whose tool results balloon the
		// context compacts before the next request, not after the turn. The
		// estimate catches growth the last reported count knows nothing about.
		if a.overThreshold(max(lastInput, lastContext.Total)) {
			if compUsage, err := a.Compact(ctx, sink); err == nil {
				total = add(total, compUsage)
				// The old reported count no longer describes the compacted
				// transcript and would re-trigger every step.
				lastInput = 0
				lastContext = a.EstimateContext()
			}
		}

		req := llm.Request{
			Model:     a.model,
			System:    a.system,
			Messages:  a.transcript,
			Tools:     a.tools.Specs(),
			Reasoning: a.reasoning,
		}

		res, wasted, err := a.streamWithRetry(ctx, req, sink, steps+1, lastContext)
		steps++
		total = add(total, add(res.usage, wasted))
		// Context-size signal, not billing: cached tokens occupy the window too.
		lastInput = res.usage.InputTokens + res.usage.CacheReadTokens + res.usage.CacheWriteTokens

		if err != nil {
			// Cancellation repair: keep streamed partial text as a text-only
			// assistant message; drop the message entirely if nothing streamed.
			// Un-executed tool calls are never appended.
			if res.text != "" {
				a.transcript = append(a.transcript, textMessage(llm.RoleAssistant, res.text))
			}
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				sink.Notice("[cancelled]")
			}
			sink.TurnComplete(TurnUsage{Steps: steps, Usage: total, Context: lastContext})
			return err
		}

		a.transcript = append(a.transcript, assistantMessage(res))

		if res.stopReason != llm.StopToolUse {
			break
		}

		results, toolUsage := a.dispatchCalls(ctx, res.toolCalls, sink)
		total = add(total, toolUsage)
		a.transcript = append(a.transcript, llm.Message{
			Role:    llm.RoleUser,
			Content: results,
		})

		if steps >= budget {
			if a.autoContinue && continues < maxAutoContinues {
				continues++
				budget += a.maxSteps
				sink.Notice(fmt.Sprintf("[max steps reached; auto-continuing (%d/%d)]", continues, maxAutoContinues))
			} else {
				sink.Notice(maxStepsNotice(a.maxSteps))
				break
			}
		}
	}

	// Post-turn compaction trigger (design §12, §8.1): fires after the turn
	// completes, before returning to the prompt. The summary call's usage folds
	// into the turn total so session totals (via the sink) include compaction. A
	// compaction error never fails the turn — the warning was already reported and
	// the transcript was kept intact.
	if compUsage, err := a.MaybeCompact(ctx, lastInput, sink); err == nil {
		total = add(total, compUsage)
		lastContext = a.EstimateContext()
	}

	sink.TurnComplete(TurnUsage{Steps: steps, Usage: total, Context: lastContext})
	return nil
}

// dispatchCalls runs one step's tool calls: concurrently when the step is
// all-read-only with 2+ calls, sequentially otherwise. Sink events and the
// returned blocks are in emission order either way, and the sink is only
// ever called from this goroutine (spec §8).
func (a *Agent) dispatchCalls(ctx context.Context, calls []llm.ToolCall, sink EventSink) ([]llm.ContentBlock, llm.Usage) {
	blocks := make([]llm.ContentBlock, len(calls))
	var total llm.Usage

	if len(calls) >= 2 && a.tools.AllReadOnly(calls) {
		for _, call := range calls {
			sink.ToolStart(call)
		}
		results := make([]llm.ToolResult, len(calls))
		sem := make(chan struct{}, maxParallelTools)
		var wg sync.WaitGroup
		for i, call := range calls {
			wg.Add(1)
			go func() {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()
				results[i] = a.tools.Dispatch(ctx, call)
			}()
		}
		wg.Wait()
		for i, r := range results {
			sink.ToolResult(r)
			blocks[i] = resultBlock(r)
			total = add(total, r.Usage)
		}
		return blocks, total
	}

	for i, call := range calls {
		sink.ToolStart(call)
		r := a.tools.Dispatch(ctx, call)
		sink.ToolResult(r)
		blocks[i] = resultBlock(r)
		total = add(total, r.Usage)
	}
	return blocks, total
}

func resultBlock(r llm.ToolResult) llm.ContentBlock {
	return llm.ContentBlock{
		Kind:        llm.BlockToolResult,
		ResultForID: r.ForID,
		ResultText:  r.Text,
		ResultError: r.IsError,
	}
}

// streamWithRetry runs stream, re-requesting the step from scratch when it
// fails mid-flight with a retryable error. Partial output from a failed
// attempt is never committed to the transcript; wasted carries the usage
// failed attempts reported (paid for, so counted) — it never drives the
// compaction trigger.
func (a *Agent) streamWithRetry(ctx context.Context, req llm.Request, sink EventSink, step int, estimate ContextEstimate) (res stepResult, wasted llm.Usage, err error) {
	for attempt := 0; ; attempt++ {
		sink.ModelStepStart(step, attempt+1, estimate)
		res, err = a.stream(ctx, req, sink)
		if err == nil || attempt >= streamRetries || !retryableStreamError(err) {
			return res, wasted, err
		}
		wasted = add(wasted, res.usage)
		sink.Notice(fmt.Sprintf("[stream interrupted: %v; retrying step]", err))
		a.sleep(retry.Next(attempt, 0))
	}
}

// retryableStreamError reports whether a mid-stream failure may be retried by
// re-requesting the step. Cancellation is the user's call to stop; a
// non-retryable APIError (invalid_request, auth) will not get better by
// asking again. Everything else — truncated streams, transport resets,
// retryable API errors — is transient (spec §2).
func retryableStreamError(err error) bool {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var apiErr *llm.APIError
	if errors.As(err, &apiErr) {
		return apiErr.Retryable
	}
	return true
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
		case llm.EventToolCallStart:
			sink.ToolUseStart(llm.ToolCall{
				ID:    ev.ToolID,
				Name:  ev.ToolName,
				Input: ev.ToolInput,
			})
		case llm.EventToolCallDelta:
			sink.ToolUseDelta(ev.Index, ev.ArgsDelta)
		case llm.EventToolCallDone:
			res.toolCalls = append(res.toolCalls, llm.ToolCall{
				ID:    ev.ToolID,
				Name:  ev.ToolName,
				Input: ev.ToolInput,
			})
		case llm.EventUsage:
			if ev.Usage != nil {
				res.usage = mergeUsage(res.usage, *ev.Usage)
			}
		case llm.EventDone:
			if ev.Usage != nil {
				res.usage = mergeUsage(res.usage, *ev.Usage)
			}
			res.stopReason = ev.StopReason
		}
	}

	res.text = string(text)
	return res, nil
}

// textMessage builds the single-text-block message shape shared by user
// prompts, cancel repair, and compaction summaries.
func textMessage(role llm.Role, text string) llm.Message {
	return llm.Message{Role: role, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: text}}}
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
		ReasoningTokens:  a.ReasoningTokens + b.ReasoningTokens,
	}
}

// mergeUsage merges a cumulative usage snapshot into acc element-wise. The
// provider contract says snapshots are cumulative; max keeps a zeroed or
// partial late frame from erasing earlier numbers (spec §3).
func mergeUsage(acc, in llm.Usage) llm.Usage {
	return llm.Usage{
		InputTokens:      max(acc.InputTokens, in.InputTokens),
		OutputTokens:     max(acc.OutputTokens, in.OutputTokens),
		CacheReadTokens:  max(acc.CacheReadTokens, in.CacheReadTokens),
		CacheWriteTokens: max(acc.CacheWriteTokens, in.CacheWriteTokens),
		ReasoningTokens:  max(acc.ReasoningTokens, in.ReasoningTokens),
	}
}
