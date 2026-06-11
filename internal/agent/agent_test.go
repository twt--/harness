package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"harness/internal/llm"
	"harness/internal/llm/llmtest"
	"harness/internal/sse"
	"harness/internal/tools"
)

// recordSink captures every sink callback so tests can assert what the UI would
// have been told.
type recordSink struct {
	text       strings.Builder
	starts     []llm.ToolCall
	results    []llm.ToolResult
	notices    []string
	turnUsage  []TurnUsage
	stepCounts []int
}

func (s *recordSink) TextDelta(t string)          { s.text.WriteString(t) }
func (s *recordSink) ToolStart(c llm.ToolCall)    { s.starts = append(s.starts, c) }
func (s *recordSink) ToolResult(r llm.ToolResult) { s.results = append(s.results, r) }
func (s *recordSink) Notice(msg string)           { s.notices = append(s.notices, msg) }
func (s *recordSink) TurnComplete(u TurnUsage) {
	s.turnUsage = append(s.turnUsage, u)
	s.stepCounts = append(s.stepCounts, u.Steps)
}

// recordTool is a fake tool whose Run is scriptable; it records the inputs it
// received in call order. The mutex guards inputs because read-only steps now
// dispatch Run concurrently.
type recordTool struct {
	name     string
	readOnly bool
	run      func(ctx context.Context, input json.RawMessage) (string, error)
	mu       sync.Mutex
	inputs   []string
}

func (t *recordTool) Name() string            { return t.name }
func (t *recordTool) Description() string     { return "fake tool" }
func (t *recordTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (t *recordTool) ReadOnly() bool          { return t.readOnly }
func (t *recordTool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	t.mu.Lock()
	t.inputs = append(t.inputs, string(input))
	t.mu.Unlock()
	return t.run(ctx, input)
}

func textDelta(s string) llm.StreamEvent {
	return llm.StreamEvent{Kind: llm.EventTextDelta, Text: s}
}

func toolDone(index int, id, name, input string) llm.StreamEvent {
	return llm.StreamEvent{
		Kind:      llm.EventToolCallDone,
		Index:     index,
		ToolID:    id,
		ToolName:  name,
		ToolInput: json.RawMessage(input),
	}
}

func mustValid(t *testing.T, msgs []llm.Message) {
	t.Helper()
	if err := llm.ValidateTranscript(msgs); err != nil {
		t.Fatalf("transcript invalid: %v\n%s", err, dump(msgs))
	}
}

func dump(msgs []llm.Message) string {
	b, _ := json.MarshalIndent(msgs, "", "  ")
	return string(b)
}

func newAgent(p llm.Provider, reg *tools.Registry, opts Options) *Agent {
	if opts.Registry == nil {
		opts.Registry = llm.NewRegistry(map[string]llm.ModelInfo{
			"claude-opus-4-8": {
				ContextWindow: 1_000_000,
				Price:         llm.Price{Input: 5.0, Output: 25.0, CacheRead: 0.5, CacheWrite: 6.25},
			},
		})
	}
	return New(p, reg, opts)
}

func TestTextOnlyTurn(t *testing.T) {
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("hello "), textDelta("world")},
		Stop:   llm.StopEndTurn,
		Usage:  llm.Usage{InputTokens: 10, OutputTokens: 5},
	})
	a := newAgent(fp, tools.Default(), Options{})
	sink := &recordSink{}

	if err := a.RunTurn(context.Background(), "hi", sink); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	msgs := a.Transcript()
	mustValid(t, msgs)
	if len(msgs) != 2 {
		t.Fatalf("want 2 messages (user+assistant), got %d:\n%s", len(msgs), dump(msgs))
	}
	if msgs[0].Role != llm.RoleUser || msgs[0].Content[0].Text != "hi" {
		t.Errorf("first message should be the user prompt, got %+v", msgs[0])
	}
	if msgs[1].Role != llm.RoleAssistant {
		t.Errorf("second message should be the assistant reply, got role %q", msgs[1].Role)
	}
	if got := sink.text.String(); got != "hello world" {
		t.Errorf("text deltas = %q, want %q", got, "hello world")
	}
	if len(fp.Requests) != 1 {
		t.Errorf("provider called %d times, want 1", len(fp.Requests))
	}
}

func TestParallelToolCallsSequentialInOrder(t *testing.T) {
	tool := &recordTool{name: "echo", run: func(_ context.Context, in json.RawMessage) (string, error) {
		return "ran " + string(in), nil
	}}
	reg := &tools.Registry{}
	reg.Register(tool)

	fp := llmtest.New("fake",
		llmtest.Step{
			Events: []llm.StreamEvent{
				textDelta("calling tools"),
				toolDone(0, "call_a", "echo", `{"n":1}`),
				toolDone(1, "call_b", "echo", `{"n":2}`),
			},
			Stop:  llm.StopToolUse,
			Usage: llm.Usage{InputTokens: 20, OutputTokens: 8},
		},
		llmtest.Step{
			Events: []llm.StreamEvent{textDelta("done")},
			Stop:   llm.StopEndTurn,
			Usage:  llm.Usage{InputTokens: 30, OutputTokens: 4},
		},
	)
	a := newAgent(fp, reg, Options{})
	sink := &recordSink{}

	if err := a.RunTurn(context.Background(), "go", sink); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	msgs := a.Transcript()
	mustValid(t, msgs)

	// user, assistant(text+2 tool_use), user(2 results), assistant(text)
	if len(msgs) != 4 {
		t.Fatalf("want 4 messages, got %d:\n%s", len(msgs), dump(msgs))
	}

	// Assistant message preserves emission order: text then both tool_use blocks.
	asst := msgs[1]
	if asst.Role != llm.RoleAssistant || len(asst.Content) != 3 {
		t.Fatalf("assistant message shape wrong:\n%s", dump([]llm.Message{asst}))
	}
	if asst.Content[0].Kind != llm.BlockText || asst.Content[1].ToolUseID != "call_a" || asst.Content[2].ToolUseID != "call_b" {
		t.Errorf("assistant content order wrong:\n%s", dump([]llm.Message{asst}))
	}

	// Results message: two tool_result blocks in call order.
	resMsg := msgs[2]
	if resMsg.Role != llm.RoleUser || len(resMsg.Content) != 2 {
		t.Fatalf("results message shape wrong:\n%s", dump([]llm.Message{resMsg}))
	}
	if resMsg.Content[0].ResultForID != "call_a" || resMsg.Content[1].ResultForID != "call_b" {
		t.Errorf("results out of order:\n%s", dump([]llm.Message{resMsg}))
	}

	// Tools executed sequentially in emission order.
	if len(tool.inputs) != 2 || tool.inputs[0] != `{"n":1}` || tool.inputs[1] != `{"n":2}` {
		t.Errorf("tool execution order wrong: %v", tool.inputs)
	}

	// Loop re-called the provider after dispatching tools.
	if len(fp.Requests) != 2 {
		t.Errorf("provider called %d times, want 2", len(fp.Requests))
	}
	if len(sink.starts) != 2 || len(sink.results) != 2 {
		t.Errorf("sink saw %d starts and %d results, want 2 each", len(sink.starts), len(sink.results))
	}
}

func TestFailingToolFedBackAsError(t *testing.T) {
	tool := &recordTool{name: "boom", run: func(_ context.Context, _ json.RawMessage) (string, error) {
		return "", errors.New("kaboom")
	}}
	reg := &tools.Registry{}
	reg.Register(tool)

	fp := llmtest.New("fake",
		llmtest.Step{
			Events: []llm.StreamEvent{toolDone(0, "call_x", "boom", `{}`)},
			Stop:   llm.StopToolUse,
		},
		llmtest.Step{
			Events: []llm.StreamEvent{textDelta("ok")},
			Stop:   llm.StopEndTurn,
		},
	)
	a := newAgent(fp, reg, Options{})
	sink := &recordSink{}

	if err := a.RunTurn(context.Background(), "go", sink); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	mustValid(t, a.Transcript())

	// The error result is appended as an is_error tool_result.
	resMsg := a.Transcript()[2]
	if len(resMsg.Content) != 1 || !resMsg.Content[0].ResultError {
		t.Fatalf("expected an is_error result:\n%s", dump([]llm.Message{resMsg}))
	}
	if !strings.Contains(resMsg.Content[0].ResultText, "kaboom") {
		t.Errorf("error text = %q, want it to mention kaboom", resMsg.Content[0].ResultText)
	}

	// The next request carries the error result so the model can self-correct.
	if len(fp.Requests) != 2 {
		t.Fatalf("provider called %d times, want 2", len(fp.Requests))
	}
	second := fp.Requests[1]
	var carried bool
	for _, m := range second.Messages {
		for _, b := range m.Content {
			if b.Kind == llm.BlockToolResult && strings.Contains(b.ResultText, "kaboom") {
				carried = true
			}
		}
	}
	if !carried {
		t.Errorf("second request did not carry the error result:\n%s", dump(second.Messages))
	}
	if len(sink.results) != 1 || !sink.results[0].IsError {
		t.Errorf("sink should have seen one is_error result, got %+v", sink.results)
	}
}

func TestMaxStepsStop(t *testing.T) {
	tool := &recordTool{name: "loop", run: func(_ context.Context, _ json.RawMessage) (string, error) {
		return "again", nil
	}}
	reg := &tools.Registry{}
	reg.Register(tool)

	// Every step asks for a tool: the loop must stop at the limit.
	always := llmtest.Step{
		Events: []llm.StreamEvent{toolDone(0, "id", "loop", `{}`)},
		Stop:   llm.StopToolUse,
	}
	fp := llmtest.New("fake", always, always, always)
	a := newAgent(fp, reg, Options{MaxSteps: 3})
	sink := &recordSink{}

	if err := a.RunTurn(context.Background(), "go", sink); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	mustValid(t, a.Transcript())

	if len(fp.Requests) != 3 {
		t.Errorf("provider called %d times, want 3 (the limit)", len(fp.Requests))
	}

	var sawMaxSteps bool
	for _, n := range sink.notices {
		if strings.Contains(n, "max steps") {
			sawMaxSteps = true
			if !strings.Contains(n, "(3)") {
				t.Errorf("max-steps notice should name the limit: %q", n)
			}
		}
	}
	if !sawMaxSteps {
		t.Errorf("sink not told about max-steps stop, notices=%v", sink.notices)
	}
}

func TestAutoContinuePastMaxSteps(t *testing.T) {
	tool := &recordTool{name: "loop", run: func(_ context.Context, _ json.RawMessage) (string, error) {
		return "again", nil
	}}
	reg := &tools.Registry{}
	reg.Register(tool)

	// Every step asks for a tool; with MaxSteps 2 and 3 auto-continues the
	// loop runs 2*(1+3)=8 steps, then stops with the final notice.
	always := llmtest.Step{
		Events: []llm.StreamEvent{toolDone(0, "id", "loop", `{}`)},
		Stop:   llm.StopToolUse,
	}
	steps := make([]llmtest.Step, 10)
	for i := range steps {
		steps[i] = always
	}
	fp := llmtest.New("fake", steps...)
	a := newAgent(fp, reg, Options{MaxSteps: 2, AutoContinue: true})
	sink := &recordSink{}

	if err := a.RunTurn(context.Background(), "go", sink); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	mustValid(t, a.Transcript())

	if len(fp.Requests) != 8 {
		t.Errorf("provider called %d times, want 8 (4 budgets of 2)", len(fp.Requests))
	}
	if sink.turnUsage[0].Steps != 8 {
		t.Errorf("TurnComplete steps = %d, want 8", sink.turnUsage[0].Steps)
	}
	var continues, stops int
	for _, n := range sink.notices {
		if strings.Contains(n, "auto-continuing") {
			continues++
		}
		if strings.Contains(n, "say \"continue\"") {
			stops++
		}
	}
	if continues != 3 {
		t.Errorf("want 3 auto-continue notices, got %d (%v)", continues, sink.notices)
	}
	if stops != 1 {
		t.Errorf("want the final stop notice once, got %d (%v)", stops, sink.notices)
	}
}

func TestCancellationMidStreamKeepsPartialText(t *testing.T) {
	tool := &recordTool{name: "noop", run: func(_ context.Context, _ json.RawMessage) (string, error) {
		return "", nil
	}}
	reg := &tools.Registry{}
	reg.Register(tool)

	ctx, cancel := context.WithCancel(context.Background())
	// The step emits partial text, then a tool_use, but cancellation fires before
	// the terminal event. Un-executed tool_use must be stripped; partial text kept.
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("partial answer")},
		Stop:   llm.StopToolUse,
		Block:  func(_ context.Context) { cancel() },
	})
	a := newAgent(fp, reg, Options{})
	sink := &recordSink{}

	err := a.RunTurn(ctx, "go", sink)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("RunTurn err = %v, want context.Canceled", err)
	}

	msgs := a.Transcript()
	mustValid(t, msgs)
	if len(msgs) != 2 {
		t.Fatalf("want user + partial assistant, got %d:\n%s", len(msgs), dump(msgs))
	}
	asst := msgs[1]
	if asst.Role != llm.RoleAssistant {
		t.Fatalf("second message should be assistant, got %q", asst.Role)
	}
	for _, b := range asst.Content {
		if b.Kind == llm.BlockToolUse {
			t.Errorf("dangling tool_use not stripped:\n%s", dump([]llm.Message{asst}))
		}
	}
	if asst.Content[0].Text != "partial answer" {
		t.Errorf("partial text not kept, got %q", asst.Content[0].Text)
	}
}

func TestCancellationWithNoTextDropsMessage(t *testing.T) {
	reg := &tools.Registry{}
	ctx, cancel := context.WithCancel(context.Background())
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{},
		Stop:   llm.StopEndTurn,
		Block:  func(_ context.Context) { cancel() },
	})
	a := newAgent(fp, reg, Options{})

	err := a.RunTurn(ctx, "go", &recordSink{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("RunTurn err = %v, want context.Canceled", err)
	}
	msgs := a.Transcript()
	mustValid(t, msgs)
	// Nothing streamed: the partial assistant message is dropped, leaving only the
	// user message.
	if len(msgs) != 1 || msgs[0].Role != llm.RoleUser {
		t.Fatalf("want only the user message, got %d:\n%s", len(msgs), dump(msgs))
	}
}

func TestUsageAccumulatedAcrossSteps(t *testing.T) {
	tool := &recordTool{name: "echo", run: func(_ context.Context, _ json.RawMessage) (string, error) {
		return "x", nil
	}}
	reg := &tools.Registry{}
	reg.Register(tool)

	fp := llmtest.New("fake",
		llmtest.Step{
			Events: []llm.StreamEvent{toolDone(0, "a", "echo", `{}`)},
			Stop:   llm.StopToolUse,
			Usage:  llm.Usage{InputTokens: 100, OutputTokens: 10},
		},
		llmtest.Step{
			Events: []llm.StreamEvent{textDelta("done")},
			Stop:   llm.StopEndTurn,
			Usage:  llm.Usage{InputTokens: 200, OutputTokens: 20},
		},
	)
	a := newAgent(fp, reg, Options{})
	sink := &recordSink{}

	if err := a.RunTurn(context.Background(), "go", sink); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	if len(sink.turnUsage) != 1 {
		t.Fatalf("want one TurnComplete, got %d", len(sink.turnUsage))
	}
	tu := sink.turnUsage[0]
	if tu.Usage.InputTokens != 300 || tu.Usage.OutputTokens != 30 {
		t.Errorf("turn usage = %+v, want 300 in / 30 out", tu.Usage)
	}
	if tu.Steps != 2 {
		t.Errorf("turn steps = %d, want 2", tu.Steps)
	}
}

// SetTools swaps the registry that backs both the advertised specs and
// dispatch, so a mode switch immediately changes what the model sees and can
// call.
func TestSetToolsChangesAdvertisedAndDispatchableTools(t *testing.T) {
	full, err := tools.Catalog().Subset([]string{"read_file", "grep"})
	if err != nil {
		t.Fatalf("subset: %v", err)
	}
	restricted, err := tools.Catalog().Subset([]string{"read_file"})
	if err != nil {
		t.Fatalf("subset: %v", err)
	}

	fp := llmtest.New("fake",
		llmtest.Step{Events: []llm.StreamEvent{textDelta("a")}, Stop: llm.StopEndTurn},
		llmtest.Step{Events: []llm.StreamEvent{textDelta("b")}, Stop: llm.StopEndTurn},
	)
	a := newAgent(fp, full, Options{})

	if err := a.RunTurn(context.Background(), "one", &recordSink{}); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	a.SetTools(restricted)
	if err := a.RunTurn(context.Background(), "two", &recordSink{}); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	if names := specNames(fp.Requests[0].Tools); !slices.Contains(names, "grep") {
		t.Errorf("first request should advertise grep, got %v", names)
	}
	if names := specNames(fp.Requests[1].Tools); slices.Contains(names, "grep") {
		t.Errorf("after SetTools, grep should no longer be advertised, got %v", names)
	}

	// A call to the now-removed tool must be undispatchable.
	res := a.tools.Dispatch(context.Background(), llm.ToolCall{ID: "1", Name: "grep", Input: json.RawMessage(`{}`)})
	if !res.IsError || !strings.Contains(res.Text, "unknown tool") {
		t.Errorf("removed tool should be undispatchable, got %+v", res)
	}
}

func TestMidStreamRetrySucceedsOnSecondAttempt(t *testing.T) {
	fp := llmtest.New("fake",
		llmtest.Step{
			Events: []llm.StreamEvent{
				textDelta("partial "),
				{Kind: llm.EventUsage, Usage: &llm.Usage{InputTokens: 40}},
			},
			Err: &llm.APIError{StatusCode: 529, Message: "overloaded", Retryable: true},
		},
		llmtest.Step{
			Events: []llm.StreamEvent{textDelta("hello")},
			Stop:   llm.StopEndTurn,
			Usage:  llm.Usage{InputTokens: 10, OutputTokens: 5},
		},
	)
	a := newAgent(fp, tools.Default(), Options{})
	a.sleep = func(time.Duration) {}
	sink := &recordSink{}

	if err := a.RunTurn(context.Background(), "hi", sink); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	msgs := a.Transcript()
	mustValid(t, msgs)
	if len(msgs) != 2 {
		t.Fatalf("want 2 messages, got %d:\n%s", len(msgs), dump(msgs))
	}
	if got := msgs[1].Content[0].Text; got != "hello" {
		t.Errorf("assistant text = %q, want %q (failed attempt must not be committed)", got, "hello")
	}
	if len(fp.Requests) != 2 {
		t.Errorf("provider called %d times, want 2", len(fp.Requests))
	}
	var retried bool
	for _, n := range sink.notices {
		if strings.Contains(n, "retrying step") {
			retried = true
		}
	}
	if !retried {
		t.Errorf("no retry notice, notices=%v", sink.notices)
	}
	// Wasted usage from the failed attempt is paid for and counted.
	if got := sink.turnUsage[0].Usage.InputTokens; got != 50 {
		t.Errorf("turn input tokens = %d, want 50 (40 wasted + 10)", got)
	}
}

func TestMidStreamRetryBudgetExhausted(t *testing.T) {
	fail := llmtest.Step{Err: &llm.APIError{StatusCode: 529, Message: "overloaded", Retryable: true}}
	fp := llmtest.New("fake", fail, fail, fail)
	a := newAgent(fp, tools.Default(), Options{})
	a.sleep = func(time.Duration) {}
	sink := &recordSink{}

	err := a.RunTurn(context.Background(), "hi", sink)
	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("RunTurn err = %v, want the APIError after budget exhaustion", err)
	}
	if len(fp.Requests) != 3 {
		t.Errorf("provider called %d times, want 3 (1 + 2 retries)", len(fp.Requests))
	}
	mustValid(t, a.Transcript())
}

func TestMidStreamNonRetryableNotRetried(t *testing.T) {
	fp := llmtest.New("fake",
		llmtest.Step{Err: &llm.APIError{StatusCode: 400, Message: "bad request", Retryable: false}},
	)
	a := newAgent(fp, tools.Default(), Options{})
	a.sleep = func(time.Duration) {}

	err := a.RunTurn(context.Background(), "hi", &recordSink{})
	if err == nil {
		t.Fatal("RunTurn should fail")
	}
	if len(fp.Requests) != 1 {
		t.Errorf("provider called %d times, want 1 (no retry)", len(fp.Requests))
	}
}

func TestTruncatedStreamRetried(t *testing.T) {
	fp := llmtest.New("fake",
		llmtest.Step{Err: fmt.Errorf("stream ended early: %w", sse.ErrTruncatedStream)},
		llmtest.Step{Events: []llm.StreamEvent{textDelta("ok")}, Stop: llm.StopEndTurn},
	)
	a := newAgent(fp, tools.Default(), Options{})
	a.sleep = func(time.Duration) {}

	if err := a.RunTurn(context.Background(), "hi", &recordSink{}); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	if len(fp.Requests) != 2 {
		t.Errorf("provider called %d times, want 2", len(fp.Requests))
	}
}

func TestCancellationDuringRetryBackoff(t *testing.T) {
	// A retryable failure schedules a retry; cancellation arrives during the
	// backoff sleep, before the next attempt. The loop must honor it: return
	// context.Canceled, attempt no further request, and leave a valid transcript.
	fail := llmtest.Step{Err: &llm.APIError{StatusCode: 529, Message: "overloaded", Retryable: true}}
	fp := llmtest.New("fake", fail, fail, fail)
	a := newAgent(fp, tools.Default(), Options{})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	a.sleep = func(time.Duration) { cancel() }

	err := a.RunTurn(ctx, "hi", &recordSink{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("RunTurn err = %v, want context.Canceled", err)
	}
	// One real attempt, then cancellation during the backoff stops the loop
	// before any retry re-requests the step.
	if len(fp.Requests) > 2 {
		t.Errorf("provider called %d times, want at most 2 (no retry after cancel)", len(fp.Requests))
	}
	mustValid(t, a.Transcript())
}

func TestZeroedFinalUsageFrameDoesNotEraseEarlier(t *testing.T) {
	// The Done event carries zero usage (FakeProvider appends Done with
	// step.Usage, here the zero value); the mid-stream snapshot must survive.
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{
			{Kind: llm.EventUsage, Usage: &llm.Usage{InputTokens: 100, OutputTokens: 10, CacheReadTokens: 7}},
			textDelta("hi"),
		},
		Stop: llm.StopEndTurn,
	})
	a := newAgent(fp, tools.Default(), Options{})
	sink := &recordSink{}

	if err := a.RunTurn(context.Background(), "hi", sink); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	u := sink.turnUsage[0].Usage
	if u.InputTokens != 100 || u.OutputTokens != 10 || u.CacheReadTokens != 7 {
		t.Errorf("usage = %+v, want the mid-stream snapshot preserved", u)
	}
}

func specNames(specs []llm.ToolSchema) []string {
	names := make([]string, len(specs))
	for i, s := range specs {
		names[i] = s.Name
	}
	return names
}

func TestRequestCarriesResolvedModel(t *testing.T) {
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("hi")},
		Stop:   llm.StopEndTurn,
	})
	a := newAgent(fp, tools.Default(), Options{Model: "claude-opus-4-8"})
	sink := &recordSink{}

	if err := a.RunTurn(context.Background(), "hi", sink); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	mustValid(t, a.Transcript())
	if len(fp.Requests) != 1 {
		t.Fatalf("provider called %d times, want 1", len(fp.Requests))
	}
	if got := fp.Requests[0].Model; got != "claude-opus-4-8" {
		t.Errorf("Request.Model = %q, want %q", got, "claude-opus-4-8")
	}
}

// barrierRun returns a Run that only completes once n calls have entered it —
// it deadlocks (then errors via timeout) under sequential dispatch.
func barrierRun(n int) func(context.Context, json.RawMessage) (string, error) {
	var wg sync.WaitGroup
	wg.Add(n)
	return func(ctx context.Context, _ json.RawMessage) (string, error) {
		wg.Done()
		done := make(chan struct{})
		go func() { wg.Wait(); close(done) }()
		select {
		case <-done:
			return "ok", nil
		case <-time.After(2 * time.Second):
			return "", errors.New("barrier timeout: calls were not concurrent")
		}
	}
}

func TestAllReadOnlyStepDispatchesConcurrently(t *testing.T) {
	run := barrierRun(2)
	t1 := &recordTool{name: "r1", readOnly: true, run: run}
	t2 := &recordTool{name: "r2", readOnly: true, run: run}
	reg := &tools.Registry{}
	reg.Register(t1)
	reg.Register(t2)

	fp := llmtest.New("fake",
		llmtest.Step{
			Events: []llm.StreamEvent{
				toolDone(0, "a", "r1", `{}`),
				toolDone(1, "b", "r2", `{}`),
			},
			Stop: llm.StopToolUse,
		},
		llmtest.Step{Events: []llm.StreamEvent{textDelta("done")}, Stop: llm.StopEndTurn},
	)
	a := newAgent(fp, reg, Options{})
	sink := &recordSink{}

	if err := a.RunTurn(context.Background(), "go", sink); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	mustValid(t, a.Transcript())

	resMsg := a.Transcript()[2]
	if len(resMsg.Content) != 2 || resMsg.Content[0].ResultForID != "a" || resMsg.Content[1].ResultForID != "b" {
		t.Fatalf("results not in emission order:\n%s", dump([]llm.Message{resMsg}))
	}
	for _, b := range resMsg.Content {
		if b.ResultError {
			t.Errorf("read-only calls were not concurrent: %s", b.ResultText)
		}
	}
	// Sink saw both starts (emission order) before both results.
	if len(sink.starts) != 2 || sink.starts[0].ID != "a" || sink.starts[1].ID != "b" {
		t.Errorf("ToolStart order wrong: %+v", sink.starts)
	}
	if len(sink.results) != 2 || sink.results[0].ForID != "a" || sink.results[1].ForID != "b" {
		t.Errorf("ToolResult order wrong: %+v", sink.results)
	}
}

func TestMixedStepStaysSequential(t *testing.T) {
	var mu sync.Mutex
	var trace []string
	mk := func(name string, ro bool) *recordTool {
		return &recordTool{name: name, readOnly: ro, run: func(_ context.Context, _ json.RawMessage) (string, error) {
			mu.Lock()
			trace = append(trace, "start:"+name)
			mu.Unlock()
			mu.Lock()
			trace = append(trace, "end:"+name)
			mu.Unlock()
			return "ok", nil
		}}
	}
	reg := &tools.Registry{}
	reg.Register(mk("reader", true))
	reg.Register(mk("writer", false))

	fp := llmtest.New("fake",
		llmtest.Step{
			Events: []llm.StreamEvent{
				toolDone(0, "a", "reader", `{}`),
				toolDone(1, "b", "writer", `{}`),
			},
			Stop: llm.StopToolUse,
		},
		llmtest.Step{Events: []llm.StreamEvent{textDelta("done")}, Stop: llm.StopEndTurn},
	)
	a := newAgent(fp, reg, Options{})

	if err := a.RunTurn(context.Background(), "go", &recordSink{}); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	want := []string{"start:reader", "end:reader", "start:writer", "end:writer"}
	if !slices.Equal(trace, want) {
		t.Errorf("mixed step interleaving = %v, want strictly sequential %v", trace, want)
	}
}
