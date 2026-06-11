package tools

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"harness/internal/llm"
)

// ctxTool honors ctx like a well-behaved tool: it blocks until cancelled.
type ctxTool struct{}

func (ctxTool) Name() string            { return "ctx_tool" }
func (ctxTool) Description() string     { return "blocks until ctx is done" }
func (ctxTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (ctxTool) Run(ctx context.Context, _ json.RawMessage) (string, error) {
	<-ctx.Done()
	return "", ctx.Err()
}

// stuckTool ignores ctx entirely — the case the ceiling exists for.
type stuckTool struct{ release chan struct{} }

func (s *stuckTool) Name() string            { return "stuck_tool" }
func (s *stuckTool) Description() string     { return "ignores ctx" }
func (s *stuckTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (s *stuckTool) Run(_ context.Context, _ json.RawMessage) (string, error) {
	<-s.release
	return "released", nil
}

func TestDispatchTimeoutCeiling(t *testing.T) {
	r := &Registry{}
	r.Register(ctxTool{})
	r.SetDispatchTimeout(20 * time.Millisecond)

	res := r.Dispatch(context.Background(), llm.ToolCall{ID: "1", Name: "ctx_tool", Input: json.RawMessage(`{}`)})
	if !res.IsError || !strings.Contains(res.Text, "timed out after 20ms") {
		t.Errorf("want timeout is_error result, got %+v", res)
	}
}

func TestDispatchTimeoutUnblocksCtxIgnoringTool(t *testing.T) {
	stuck := &stuckTool{release: make(chan struct{})}
	defer close(stuck.release) // let the abandoned goroutine finish
	r := &Registry{}
	r.Register(stuck)
	r.SetDispatchTimeout(20 * time.Millisecond)

	done := make(chan llm.ToolResult, 1)
	go func() {
		done <- r.Dispatch(context.Background(), llm.ToolCall{ID: "1", Name: "stuck_tool", Input: json.RawMessage(`{}`)})
	}()
	select {
	case res := <-done:
		if !res.IsError || !strings.Contains(res.Text, "timed out") {
			t.Errorf("want timeout is_error result, got %+v", res)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Dispatch did not return: ceiling failed to unblock a ctx-ignoring tool")
	}
}

func TestDispatchOuterCancellationIsNotATimeout(t *testing.T) {
	r := &Registry{}
	r.Register(ctxTool{})

	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(10 * time.Millisecond); cancel() }()
	res := r.Dispatch(ctx, llm.ToolCall{ID: "1", Name: "ctx_tool", Input: json.RawMessage(`{}`)})
	if !res.IsError {
		t.Fatalf("want is_error result, got %+v", res)
	}
	if strings.Contains(res.Text, "timed out") {
		t.Errorf("outer cancellation must not be reported as a timeout: %q", res.Text)
	}
	if !strings.Contains(res.Text, context.Canceled.Error()) {
		t.Errorf("want cancellation error in result, got %q", res.Text)
	}
}
