// Package delegate implements the read-only sub-agent tool. It lives outside
// internal/tools to avoid a tools -> agent import cycle: the tool starts a
// child agent, and agent already dispatches through tools.
package delegate

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"harness/internal/agent"
	"harness/internal/llm"
	"harness/internal/tools"
)

const DefaultMaxTurns = 20

const schema = `{
  "type": "object",
  "properties": {
    "task": {"type": "string", "description": "The self-contained task for the delegate agent."},
    "agent": {"type": "string", "description": "Optional configured agent name to run. When omitted, uses the current active agent."},
    "max_turns": {"type": "integer", "minimum": 1, "description": "Optional model-turn cap for this delegate call. Values above the configured cap are reduced to the cap."}
  },
  "required": ["task"]
}`

// Runtime is the parent agent state a delegate call needs to start a child.
type Runtime struct {
	Provider      llm.Provider
	ProviderName  string
	Model         string
	ContextWindow int
	Registry      *llm.Registry
	Reasoning     llm.ReasoningConfig
	System        string
	Agent         string
}

// Launch is the fully resolved child-agent runtime for one delegate call.
type Launch struct {
	Provider      llm.Provider
	Model         string
	ContextWindow int
	Registry      *llm.Registry
	Reasoning     llm.ReasoningConfig
	System        string
	Tools         *tools.Registry
}

// State stores the current runtime snapshot. Main updates it on startup and
// after /model or /agent switches; delegate calls read it when they begin.
type State struct {
	mu      sync.RWMutex
	runtime Runtime
}

func NewState(runtime Runtime) *State {
	return &State{runtime: runtime}
}

func (s *State) Set(runtime Runtime) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.runtime = runtime
}

func (s *State) Snapshot() Runtime {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.runtime
}

// Options configures the delegate tool.
type Options struct {
	MaxTurns                  int
	CompactKeepTurns          int
	CompactSummaryMaxTokens   int
	CompactToolResultMaxBytes int
}

// Tool is a model-callable configured-agent launcher.
type Tool struct {
	snapshot func() Runtime
	resolve  func(Runtime, string) (Launch, error)
	opts     Options
}

func New(snapshot func() Runtime, resolve func(Runtime, string) (Launch, error), opts Options) *Tool {
	return &Tool{snapshot: snapshot, resolve: resolve, opts: opts}
}

func (*Tool) Name() string { return "delegate" }

func (*Tool) Description() string {
	return "Run a configured delegate agent on a self-contained task and return its final report."
}

func (*Tool) Schema() json.RawMessage { return json.RawMessage(schema) }

func (*Tool) ReadOnly() bool { return false }

func (t *Tool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	result, err := t.RunMetered(ctx, input)
	return result.Text, err
}

func (t *Tool) RunMetered(ctx context.Context, input json.RawMessage) (tools.MeteredResult, error) {
	var args struct {
		Task     string `json:"task"`
		Agent    string `json:"agent"`
		MaxTurns *int   `json:"max_turns"`
	}
	if err := json.Unmarshal(input, &args); err != nil {
		return tools.MeteredResult{}, err
	}
	task := strings.TrimSpace(args.Task)
	if task == "" {
		return tools.MeteredResult{}, fmt.Errorf("task is required")
	}
	maxTurns, err := t.maxTurns(args.MaxTurns)
	if err != nil {
		return tools.MeteredResult{}, err
	}

	runtime := t.snapshot()
	if runtime.Provider == nil {
		return tools.MeteredResult{}, fmt.Errorf("delegate runtime is not initialized")
	}
	if t.resolve == nil {
		return tools.MeteredResult{}, fmt.Errorf("delegate resolver is not initialized")
	}
	launch, err := t.resolve(runtime, strings.TrimSpace(args.Agent))
	if err != nil {
		return tools.MeteredResult{}, err
	}
	if launch.Provider == nil {
		return tools.MeteredResult{}, fmt.Errorf("delegate provider is not initialized")
	}
	if launch.Tools == nil {
		return tools.MeteredResult{}, fmt.Errorf("delegate tool registry is not initialized")
	}

	child := agent.New(launch.Provider, launch.Tools, agent.Options{
		MaxTurns:                  maxTurns,
		Model:                     launch.Model,
		ContextWindow:             launch.ContextWindow,
		Registry:                  launch.Registry,
		Reasoning:                 launch.Reasoning,
		CompactKeepTurns:          t.opts.CompactKeepTurns,
		CompactSummaryMaxTokens:   t.opts.CompactSummaryMaxTokens,
		CompactToolResultMaxBytes: t.opts.CompactToolResultMaxBytes,
	})
	child.SetSystem(launch.System)

	sink := &quietSink{}
	if err := child.RunTurn(ctx, task, sink); err != nil {
		return tools.MeteredResult{Usage: sink.usage.Usage}, err
	}

	report := strings.TrimSpace(lastAssistantText(child.Transcript()))
	if report == "" {
		report = "(delegate completed without a final text response)"
	}
	report += fmt.Sprintf("\n\n[delegate: %s, %d input tokens, %d output tokens]",
		modelTurnPhrase(sink.usage.ModelTurns), sink.usage.Usage.InputTokens, sink.usage.Usage.OutputTokens)
	return tools.MeteredResult{Text: report, Usage: sink.usage.Usage}, nil
}

func (t *Tool) maxTurns(requested *int) (int, error) {
	cap := t.opts.MaxTurns
	if cap <= 0 {
		cap = DefaultMaxTurns
	}
	if requested == nil {
		return cap, nil
	}
	if *requested <= 0 {
		return 0, fmt.Errorf("max_turns must be positive")
	}
	if *requested > cap {
		return cap, nil
	}
	return *requested, nil
}

func modelTurnPhrase(n int) string {
	if n == 1 {
		return "1 model turn"
	}
	return fmt.Sprintf("%d model turns", n)
}

func lastAssistantText(msgs []llm.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != llm.RoleAssistant {
			continue
		}
		var parts []string
		for _, b := range msgs[i].Content {
			if b.Kind == llm.BlockText && b.Text != "" {
				parts = append(parts, b.Text)
			}
		}
		return strings.Join(parts, "")
	}
	return ""
}

type quietSink struct {
	usage agent.TurnUsage
}

func (*quietSink) TextDelta(string) {}

func (*quietSink) ModelTurnStart(int, int, agent.ContextEstimate) {}

func (*quietSink) ToolUseStart(llm.ToolCall) {}

func (*quietSink) ToolUseDelta(int, string) {}

func (*quietSink) ToolStart(llm.ToolCall) {}

func (*quietSink) ToolResult(llm.ToolResult) {}

func (*quietSink) Notice(string) {}

func (s *quietSink) TurnComplete(usage agent.TurnUsage) {
	s.usage = usage
}
