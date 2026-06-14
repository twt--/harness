package delegate

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"harness/internal/llm"
	"harness/internal/llm/llmtest"
	"harness/internal/tools"
)

type fakeChildTool struct {
	name string
	out  string
}

func (t fakeChildTool) Name() string            { return t.name }
func (t fakeChildTool) Description() string     { return "child test tool" }
func (t fakeChildTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (t fakeChildTool) ReadOnly() bool          { return true }
func (t fakeChildTool) Run(context.Context, json.RawMessage) (string, error) {
	return t.out, nil
}

func TestDelegateRunsChildAgentAndReturnsFinalReport(t *testing.T) {
	childTools := &tools.Registry{}
	childTools.Register(fakeChildTool{name: "read_file", out: "file contents"})
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "final report"}},
		Stop:   llm.StopEndTurn,
		Usage:  llm.Usage{InputTokens: 11, OutputTokens: 5},
	})
	state := NewState(Runtime{
		Provider: fp,
		Model:    "claude-opus-4-8",
		Registry: llm.NewRegistry(nil),
		System:   "parent system",
	})
	tool := New(state.Snapshot, func(runtime Runtime, name string) (Launch, error) {
		if name != "" {
			t.Fatalf("delegate agent name = %q, want empty", name)
		}
		return Launch{
			Provider:      runtime.Provider,
			Model:         runtime.Model,
			ContextWindow: runtime.ContextWindow,
			Registry:      runtime.Registry,
			Reasoning:     runtime.Reasoning,
			System:        runtime.System,
			Tools:         childTools,
		}, nil
	}, Options{MaxTurns: 3})

	result, err := tool.RunMetered(context.Background(), json.RawMessage(`{"task":"inspect the repo"}`))
	if err != nil {
		t.Fatalf("RunMetered: %v", err)
	}
	if !strings.Contains(result.Text, "final report") || !strings.Contains(result.Text, "[delegate: 1 model turn") {
		t.Fatalf("delegate output = %q", result.Text)
	}
	if result.Usage.InputTokens != 11 || result.Usage.OutputTokens != 5 {
		t.Fatalf("usage = %+v, want 11/5", result.Usage)
	}
	if len(fp.Requests) != 1 {
		t.Fatalf("child requests = %d, want 1", len(fp.Requests))
	}
	req := fp.Requests[0]
	if req.Model != "claude-opus-4-8" {
		t.Fatalf("request model = %q", req.Model)
	}
	if req.System != "parent system" {
		t.Fatalf("child system = %q, want exact parent system", req.System)
	}
	if len(req.Messages) != 1 || req.Messages[0].Content[0].Text != "inspect the repo" {
		t.Fatalf("child transcript = %+v", req.Messages)
	}
	if len(req.Tools) != 1 || req.Tools[0].Name != "read_file" {
		t.Fatalf("child tools = %+v, want only read_file", req.Tools)
	}
}

func TestDelegateCapsMaxTurns(t *testing.T) {
	childTools := &tools.Registry{}
	childTools.Register(fakeChildTool{name: "read_file", out: "ok"})
	fp := llmtest.New("fake", llmtest.Step{Stop: llm.StopEndTurn})
	tool := New(func() Runtime {
		return Runtime{Provider: fp, Model: "m", Registry: llm.NewRegistry(nil)}
	}, func(runtime Runtime, name string) (Launch, error) {
		return Launch{
			Provider: runtime.Provider,
			Model:    runtime.Model,
			Registry: runtime.Registry,
			Tools:    childTools,
		}, nil
	}, Options{})

	if _, err := tool.RunMetered(context.Background(), json.RawMessage(`{"task":"go","max_turns":0}`)); err == nil {
		t.Fatalf("explicit max_turns=0 should be rejected")
	}

	result, err := tool.RunMetered(context.Background(), json.RawMessage(`{"task":"go","max_turns":99}`))
	if err != nil {
		t.Fatalf("RunMetered with capped max_turns: %v", err)
	}
	if !strings.Contains(result.Text, "[delegate: 1 model turn") {
		t.Fatalf("delegate output = %q", result.Text)
	}
}

func TestDelegatePassesRequestedAgentToResolver(t *testing.T) {
	childTools := &tools.Registry{}
	childTools.Register(fakeChildTool{name: "write_file", out: "ok"})
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "style report"}},
		Stop:   llm.StopEndTurn,
	})
	state := NewState(Runtime{Provider: fp, Model: "parent-model", Registry: llm.NewRegistry(nil)})
	var gotName string
	tool := New(state.Snapshot, func(runtime Runtime, name string) (Launch, error) {
		gotName = name
		return Launch{
			Provider: runtime.Provider,
			Model:    "style-model",
			Registry: runtime.Registry,
			System:   "style system",
			Tools:    childTools,
		}, nil
	}, Options{})

	_, err := tool.RunMetered(context.Background(), json.RawMessage(`{"task":"check style","agent":"style_review"}`))
	if err != nil {
		t.Fatalf("RunMetered: %v", err)
	}
	if gotName != "style_review" {
		t.Fatalf("resolver agent = %q, want style_review", gotName)
	}
	req := fp.Requests[0]
	if req.Model != "style-model" || req.System != "style system" {
		t.Fatalf("request model/system = %q/%q", req.Model, req.System)
	}
	if len(req.Tools) != 1 || req.Tools[0].Name != "write_file" {
		t.Fatalf("child tools = %+v, want configured write_file", req.Tools)
	}
}
