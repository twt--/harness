# Roadmap Items 2–9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all eight open roadmap items (mid-stream retry, defensive usage, proactive compaction, maxSteps auto-continue, dispatch timeout, Anthropic cache breakpoints, parallel read-only dispatch, gitignore-aware grep) per the approved spec.

**Architecture:** Lanes of tasks grouped so no two concurrent lanes touch the same file. Wave 0 is one docs commit on main. Wave 1 runs lanes 1–4 in parallel worktrees; wave 2 (lane 5) and wave 3 (lane 6) each run alone after the prior wave merges. Spec: `docs/superpowers/specs/2026-06-11-roadmap-items-design.md`.

**Tech Stack:** Go 1.24+, stdlib only. Tests use the existing patterns: `llmtest.FakeProvider` + `recordSink` for the agent loop, stub tools for dispatch, `httptest` for providers, temp git repos for grep.

---

## Orchestration protocol (for the lead session)

- **Wave 0** is executed by the orchestrator directly on `main` (Task 0).
- Each lane is dispatched to one subagent in an isolated git worktree on a fresh branch from current `main`. Lane → branch:
  - Lane 1: `feat/midstream-retry-usage` (items #2, #6) — wave 1
  - Lane 2: `feat/dispatch-timeout` (item #5) — wave 1
  - Lane 3: `feat/anthropic-tool-cache` (item #7) — wave 1
  - Lane 4: `feat/grep-gitignore` (item #9) — wave 1
  - Lane 5: `feat/proactive-compaction-autocontinue` (items #3, #4) — wave 2
  - Lane 6: `feat/parallel-readonly-dispatch` (item #8) — wave 3
- Every lane finishes with: `go build ./... && go vet ./... && go test ./...` green, plus its roadmap bookkeeping task.
- Between waves the orchestrator reviews each lane's diff, merges with `git merge --no-ff <branch>`, re-runs `go test ./...` on main, then dispatches the next wave from the new main.

---

## Task 0 (Wave 0, orchestrator): Amend the non-goals

**Files:**
- Modify: `docs/design.md` (§1 Non-goals, ~line 25)
- Modify: `AGENTS.md` (line 103)

- [ ] **Step 1: Edit `docs/design.md` §1**

Replace the non-goals bullet:

```markdown
- `.gitignore`-aware search, markdown rendering, parallel tool execution, MCP, sub-agents.
```

with:

```markdown
- Markdown rendering, MCP, sub-agents.
- Adopted in v1.1 (no longer non-goals; see
  `docs/superpowers/specs/2026-06-11-roadmap-items-design.md`): parallel
  dispatch of read-only tool calls, and gitignore-aware search delegated
  to `git ls-files` rather than an in-tree matcher.
```

- [ ] **Step 2: Edit `AGENTS.md` line 103**

Replace:

```markdown
- Do not add parallel tool execution, sub-agents, MCP, `.gitignore`-aware search, or markdown rendering. All explicit v1 non-goals (see `docs/design.md` §1).
```

with:

```markdown
- Do not add sub-agents, MCP, or markdown rendering. Explicit non-goals (see `docs/design.md` §1). Parallel read-only tool dispatch and gitignore-aware grep (via `git ls-files`) were adopted in v1.1 — see `docs/superpowers/specs/2026-06-11-roadmap-items-design.md`.
```

- [ ] **Step 3: Commit**

```bash
git add docs/design.md AGENTS.md
git commit -m "docs: adopt parallel read-only dispatch and gitignore-aware grep into scope"
```

---

## Lane 1 (Wave 1): Mid-stream retry (#2) + defensive usage (#6)

**Files:**
- Modify: `internal/agent/agent.go`
- Modify: `internal/agent/compact.go` (`summarize`)
- Modify: `internal/llm/provider.go` (doc comment only)
- Modify: `internal/llm/anthropic/provider.go` (error-frame retryability)
- Test: `internal/agent/agent_test.go`, `internal/agent/compact_test.go`, `internal/llm/anthropic/provider_test.go`

Spec sections 2 and 3. Read `internal/agent/agent.go` and `internal/llm/llmtest/fake.go` in full before starting.

### Task 1.1: Anthropic error-frame retryability

- [ ] **Step 1: Write the failing test**

Append to `internal/llm/anthropic/provider_test.go` (match the file's existing helper style; if it already has an SSE-server helper, use it instead of inlining):

```go
func TestMidStreamErrorFrameRetryability(t *testing.T) {
	cases := []struct {
		errType   string
		retryable bool
	}{
		{"overloaded_error", true},
		{"api_error", true},
		{"rate_limit_error", true},
		{"invalid_request_error", false},
	}
	for _, tc := range cases {
		t.Run(tc.errType, func(t *testing.T) {
			body := "event: message_start\n" +
				`data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}` + "\n\n" +
				"event: error\n" +
				`data: {"type":"error","error":{"type":"` + tc.errType + `","message":"x"}}` + "\n\n"
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("content-type", "text/event-stream")
				io.WriteString(w, body)
			}))
			defer srv.Close()

			p := New(Config{APIKey: "k", BaseURL: srv.URL})
			var streamErr error
			for _, err := range p.Stream(context.Background(), llm.Request{Model: "m"}) {
				if err != nil {
					streamErr = err
				}
			}
			var apiErr *llm.APIError
			if !errors.As(streamErr, &apiErr) {
				t.Fatalf("stream error = %v, want *llm.APIError", streamErr)
			}
			if apiErr.Retryable != tc.retryable {
				t.Errorf("Retryable = %v, want %v", apiErr.Retryable, tc.retryable)
			}
		})
	}
}
```

- [ ] **Step 2: Run it; expect failure**

Run: `go test ./internal/llm/anthropic/ -run TestMidStreamErrorFrameRetryability -v`
Expected: FAIL — `Retryable = false, want true` for the three retryable types.

- [ ] **Step 3: Implement**

In `internal/llm/anthropic/provider.go`, `decode`'s `case "error":` becomes:

```go
		case "error":
			apiErr := &llm.APIError{Message: "stream error"}
			if data.Error != nil {
				apiErr.Code = data.Error.Type
				apiErr.Message = data.Error.Message
				apiErr.Retryable = retryableErrorType(data.Error.Type)
			}
			yield(llm.StreamEvent{}, apiErr)
			return
```

and add at the bottom of the file:

```go
// retryableErrorType classifies mid-stream error-frame types: transient server
// conditions are retryable by re-requesting the step; everything else
// (invalid_request_error, authentication_error, ...) is terminal.
func retryableErrorType(t string) bool {
	switch t {
	case "overloaded_error", "api_error", "rate_limit_error":
		return true
	}
	return false
}
```

- [ ] **Step 4: Run tests; expect pass**

Run: `go test ./internal/llm/anthropic/ -v`
Expected: PASS (all, including existing tests).

- [ ] **Step 5: Commit**

```bash
git add internal/llm/anthropic/provider.go internal/llm/anthropic/provider_test.go
git commit -m "fix(anthropic): classify mid-stream error frames as retryable by type"
```

### Task 1.2: `streamWithRetry` in the agent loop

- [ ] **Step 1: Write the failing tests**

Append to `internal/agent/agent_test.go` (add `"time"` to its imports):

```go
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
```

Add imports to the test file: `"fmt"`, `"time"`, `"harness/internal/sse"`.

- [ ] **Step 2: Run them; expect failure**

Run: `go test ./internal/agent/ -run TestMidStream -v`
Expected: compile FAIL — `a.sleep` undefined. (The cancellation tests `TestCancellationMidStream*` are the not-retried-on-cancel regression and must still pass when done.)

- [ ] **Step 3: Implement**

In `internal/agent/agent.go`:

Add imports `"time"` and `"harness/internal/retry"`. Add to the consts:

```go
// streamRetries is the per-step mid-stream retry budget: a step whose stream
// fails after the first byte may be re-requested this many times (spec §2).
// Retries do not consume the maxSteps budget.
const streamRetries = 2
```

Add a `sleep func(time.Duration)` field to `Agent` and set `sleep: time.Sleep` in `New`. Then add:

```go
// streamWithRetry runs stream, re-requesting the step from scratch when it
// fails mid-flight with a retryable error. Partial output from a failed
// attempt is never committed to the transcript; wasted carries the usage
// failed attempts reported (paid for, so counted) — it never drives the
// compaction trigger.
func (a *Agent) streamWithRetry(ctx context.Context, req llm.Request, sink EventSink) (res stepResult, wasted llm.Usage, err error) {
	for attempt := 0; ; attempt++ {
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
```

In `RunTurn`, replace:

```go
		res, err := a.stream(ctx, req, sink)
		steps++
		total = add(total, res.usage)
		lastInput = res.usage.InputTokens
```

with:

```go
		res, wasted, err := a.streamWithRetry(ctx, req, sink)
		steps++
		total = add(total, add(res.usage, wasted))
		lastInput = res.usage.InputTokens
```

- [ ] **Step 4: Run tests; expect pass**

Run: `go test ./internal/agent/ -v`
Expected: PASS — including the pre-existing cancellation tests (cancellation must not be retried).

- [ ] **Step 5: Commit**

```bash
git add internal/agent/agent.go internal/agent/agent_test.go
git commit -m "feat(agent): retry steps whose stream fails mid-flight"
```

### Task 1.3: Defensive usage merge (#6)

- [ ] **Step 1: Write the failing tests**

Append to `internal/agent/agent_test.go`:

```go
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
```

Append to `internal/agent/compact_test.go` a matching test for `summarize` (follow that file's existing setup helpers for building an over-threshold transcript; the essential assertion):

```go
func TestSummarizeUsageSurvivesZeroedDoneFrame(t *testing.T) {
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{
			{Kind: llm.EventUsage, Usage: &llm.Usage{InputTokens: 55, OutputTokens: 5}},
			textDelta("summary"),
		},
		Stop: llm.StopEndTurn,
	})
	a := newAgent(fp, tools.Default(), Options{})
	_, usage, err := a.summarize(context.Background(), []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: "old"}}},
	})
	if err != nil {
		t.Fatalf("summarize: %v", err)
	}
	if usage.InputTokens != 55 || usage.OutputTokens != 5 {
		t.Errorf("usage = %+v, want 55 in / 5 out preserved", usage)
	}
}
```

- [ ] **Step 2: Run them; expect failure**

Run: `go test ./internal/agent/ -run 'TestZeroedFinalUsage|TestSummarizeUsage' -v`
Expected: FAIL — usage fields zeroed by the Done frame overwrite.

- [ ] **Step 3: Implement**

In `internal/agent/agent.go` add:

```go
// mergeUsage merges a cumulative usage snapshot into acc element-wise. The
// provider contract says snapshots are cumulative; max keeps a zeroed or
// partial late frame from erasing earlier numbers (spec §3).
func mergeUsage(acc, in llm.Usage) llm.Usage {
	return llm.Usage{
		InputTokens:      max(acc.InputTokens, in.InputTokens),
		OutputTokens:     max(acc.OutputTokens, in.OutputTokens),
		CacheReadTokens:  max(acc.CacheReadTokens, in.CacheReadTokens),
		CacheWriteTokens: max(acc.CacheWriteTokens, in.CacheWriteTokens),
	}
}
```

In `Agent.stream`, both usage assignments change from `res.usage = *ev.Usage` to `res.usage = mergeUsage(res.usage, *ev.Usage)` (the `EventUsage` case and the `EventDone` case).

In `compact.go` `summarize`, both `usage = *ev.Usage` assignments become `usage = mergeUsage(usage, *ev.Usage)`.

(The spec's "verify `stream_options.include_usage`" item is already satisfied: `internal/llm/openai/wire.go:138` sets it unconditionally — nothing to do.)

In `internal/llm/provider.go`, extend the `Stream` doc comment:

```go
	// Stream runs one model call. The iterator yields events until a Done
	// event or a terminal error (yielded at most once, last). Consumer break
	// or ctx cancellation aborts the underlying HTTP request.
	//
	// Usage events carry cumulative snapshots of the whole call, never
	// deltas; consumers may merge them with element-wise max.
	Stream(ctx context.Context, req Request) iter.Seq2[StreamEvent, error]
```

- [ ] **Step 4: Run tests; expect pass**

Run: `go test ./internal/agent/ ./internal/llm/... -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/agent/agent.go internal/agent/compact.go internal/llm/provider.go internal/agent/agent_test.go internal/agent/compact_test.go
git commit -m "fix(agent): merge usage snapshots defensively instead of overwriting"
```

### Task 1.4: Lane gate + roadmap

- [ ] **Step 1: Full gate**

Run: `go build ./... && go vet ./... && go test ./...`
Expected: all PASS.

- [ ] **Step 2: Move roadmap items #2 and #6 to Done**

In `docs/roadmap.md`: delete entries 2 ("Mid-stream retry") and 6 ("Defensive usage accounting") from their sections and append to **Done** (keep numbering style):

```markdown
2. **Mid-stream retry.** The agent loop re-requests a step from scratch when
   its stream fails after the first byte (truncated body, retryable error
   frame, transport reset), 2 retries per step with backoff. Failed-attempt
   usage still counts toward the turn total.
3. **Defensive usage accounting.** Usage events are documented as cumulative
   snapshots and merged element-wise (max) in the loop and compaction, so a
   zeroed or partial late frame cannot erase earlier numbers.
```

- [ ] **Step 3: Commit**

```bash
git add docs/roadmap.md
git commit -m "docs(roadmap): move mid-stream retry and usage accounting to done"
```

---

## Lane 2 (Wave 1): Dispatch timeout ceiling (#5)

**Files:**
- Modify: `internal/tools/tool.go` (`Registry`, `Dispatch`)
- Test: Create `internal/tools/dispatchtimeout_test.go`

Spec section 6. Read `internal/tools/tool.go` in full first.

### Task 2.1: Timeout ceiling in Dispatch

- [ ] **Step 1: Write the failing tests**

Create `internal/tools/dispatchtimeout_test.go`:

```go
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
```

- [ ] **Step 2: Run them; expect failure**

Run: `go test ./internal/tools/ -run TestDispatch -v`
Expected: compile FAIL — `SetDispatchTimeout` undefined.

- [ ] **Step 3: Implement**

In `internal/tools/tool.go`: add `"time"` to imports; add to `Registry`:

```go
type Registry struct {
	order           []string
	tools           map[string]Tool
	dispatchTimeout time.Duration // 0 = defaultDispatchTimeout
}
```

Add near the top:

```go
// defaultDispatchTimeout caps any single tool call: the largest tool
// self-limit (run_command/exec cap timeout_seconds at 600s) plus a one-minute
// grace, so the ceiling never fires first for well-behaved tools. It exists to
// stop tools with no self-limit from hanging the turn (spec §6).
const defaultDispatchTimeout = 11 * time.Minute

// SetDispatchTimeout overrides the per-call ceiling applied by Dispatch.
// Non-positive values reset to the default.
func (r *Registry) SetDispatchTimeout(d time.Duration) { r.dispatchTimeout = d }
```

Replace the body of `Dispatch` from the `defer func() { recover... }()` onward (keeping the unknown-tool and empty-input handling above it) with a goroutine + select; the panic recover moves into the goroutine so a panicking tool cannot crash the process:

```go
func (r *Registry) Dispatch(parent context.Context, call llm.ToolCall) (res llm.ToolResult) {
	res.ForID = call.ID

	t, ok := r.tools[call.Name]
	if !ok {
		res.Text = fmt.Sprintf("error: unknown tool %q", call.Name)
		res.IsError = true
		return res
	}

	input := call.Input
	if len(input) == 0 {
		input = json.RawMessage("{}")
	}

	timeout := r.dispatchTimeout
	if timeout <= 0 {
		timeout = defaultDispatchTimeout
	}
	ctx, cancel := context.WithTimeout(parent, timeout)
	defer cancel()

	type outcome struct {
		out string
		err error
	}
	done := make(chan outcome, 1) // buffered: an abandoned Run can still send and exit
	go func() {
		defer func() {
			if rec := recover(); rec != nil {
				log.Printf("tool %q panicked: %v", call.Name, rec)
				done <- outcome{err: fmt.Errorf("tool panicked: %v", rec)}
			}
		}()
		out, err := t.Run(ctx, input)
		done <- outcome{out: out, err: err}
	}()

	var out string
	var err error
	select {
	case o := <-done:
		out, err = o.out, o.err
	case <-ctx.Done():
		// The Run goroutine is abandoned (standard cost of a timeout shim);
		// its eventual send lands in the buffered channel and is dropped.
		if parent.Err() != nil {
			res.Text = "error: " + parent.Err().Error()
		} else {
			res.Text = fmt.Sprintf("error: tool timed out after %s", timeout)
		}
		res.IsError = true
		return res
	}

	if err != nil {
		// A well-behaved tool returning because the ceiling expired reports a
		// timeout, not a bare context error; outer cancellation stays as-is.
		if errors.Is(err, context.DeadlineExceeded) && parent.Err() == nil {
			res.Text = fmt.Sprintf("error: tool timed out after %s", timeout)
		} else if _, bad := err.(*invalidArgsError); bad {
			res.Text = "error: invalid arguments: " + err.Error()
		} else if isJSONError(err) {
			res.Text = "error: invalid arguments: " + err.Error()
		} else {
			res.Text = "error: " + err.Error()
		}
		res.IsError = true
		return res
	}

	res.Text = truncate(out)
	return res
}
```

Add `"errors"` to the imports. Note the old top-level `defer recover` is gone — the goroutine's recover replaces it; the "tool panicked" result text now arrives through the error path, so update the existing panic test's expected text in `tool_test.go` if it asserts the exact string (it should match `error: tool panicked: ...` — the same prefix as before).

- [ ] **Step 4: Run tests; expect pass**

Run: `go test ./internal/tools/ -v`
Expected: PASS — including the pre-existing panic-recovery and error-mapping tests in `tool_test.go`.

- [ ] **Step 5: Commit**

```bash
git add internal/tools/tool.go internal/tools/dispatchtimeout_test.go internal/tools/tool_test.go
git commit -m "feat(tools): enforce a dispatch-level timeout ceiling on every tool call"
```

### Task 2.2: Lane gate + roadmap

- [ ] **Step 1: Full gate**

Run: `go build ./... && go vet ./... && go test ./...`
Expected: all PASS.

- [ ] **Step 2: Move roadmap item #5 to Done**

In `docs/roadmap.md`, remove entry 5 and append to **Done**:

```markdown
4. **Per-tool-call timeout ceiling in `Dispatch`.** Every tool call gets an
   11-minute ceiling (largest self-limit + grace); Dispatch unblocks even for
   tools that ignore ctx, returning a timed-out is_error result.
```

- [ ] **Step 3: Commit**

```bash
git add docs/roadmap.md
git commit -m "docs(roadmap): move dispatch timeout ceiling to done"
```

---

## Lane 3 (Wave 1): Anthropic tools cache breakpoint (#7)

**Files:**
- Modify: `internal/llm/anthropic/wire.go` (`buildRequest`)
- Test: `internal/llm/anthropic/request_test.go`

Spec section 7. Read `internal/llm/anthropic/wire.go` and `request_test.go` first.

### Task 3.1: Breakpoint on the last tool

- [ ] **Step 1: Write the failing test**

Append to `internal/llm/anthropic/request_test.go`:

```go
func TestBuildRequestToolsCacheBreakpoint(t *testing.T) {
	req := llm.Request{
		Model: "m",
		Tools: []llm.ToolSchema{
			{Name: "a", Parameters: json.RawMessage(`{}`)},
			{Name: "b", Parameters: json.RawMessage(`{}`)},
		},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: "hi"}}},
		},
	}
	w := buildRequest(req, 200_000)

	if w.Tools[0].CacheControl != nil {
		t.Error("first tool must not carry cache_control")
	}
	if w.Tools[1].CacheControl == nil || w.Tools[1].CacheControl.Type != "ephemeral" {
		t.Errorf("last tool must carry the ephemeral breakpoint, got %+v", w.Tools[1].CacheControl)
	}
}

func TestBuildRequestNoToolsNoBreakpointPanic(t *testing.T) {
	req := llm.Request{
		Model: "m",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: "hi"}}},
		},
	}
	w := buildRequest(req, 200_000)
	if len(w.Tools) != 0 {
		t.Fatalf("unexpected tools: %+v", w.Tools)
	}
}
```

- [ ] **Step 2: Run them; expect failure**

Run: `go test ./internal/llm/anthropic/ -run TestBuildRequestTools -v`
Expected: FAIL — last tool has no cache_control.

- [ ] **Step 3: Implement**

In `internal/llm/anthropic/wire.go` `buildRequest`, after the tools loop:

```go
	for _, t := range req.Tools {
		w.Tools = append(w.Tools, wireTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.Parameters,
		})
	}
	// Third breakpoint (of the 4 allowed): the tool-schema array is the static
	// prefix; caching it separately survives system-prompt changes such as a
	// run-mode switch (spec §7).
	if n := len(w.Tools); n > 0 {
		w.Tools[n-1].CacheControl = ephemeral
	}
```

Update the comment above `buildRequest` ("cache_control breakpoints are placed on the system block and the last content block of the final message") to mention all three placements.

- [ ] **Step 4: Run tests; expect pass**

Run: `go test ./internal/llm/anthropic/ -v`
Expected: PASS. If `TestBuildRequestGolden` includes tools, update its golden expectation: the last tool entry gains `"cache_control":{"type":"ephemeral"}`.

- [ ] **Step 5: Commit**

```bash
git add internal/llm/anthropic/wire.go internal/llm/anthropic/request_test.go
git commit -m "feat(anthropic): add cache breakpoint after the tool-schema array"
```

### Task 3.2: Lane gate + roadmap

- [ ] **Step 1: Full gate**

Run: `go build ./... && go vet ./... && go test ./...`
Expected: all PASS.

- [ ] **Step 2: Move roadmap item #7 to Done**

Remove entry 7 from `docs/roadmap.md`, append to **Done**:

```markdown
5. **Anthropic cache-breakpoint tuning.** A third breakpoint after the static
   tool-schema array preserves the cached tools segment across system-prompt
   changes (e.g. run-mode switches).
```

- [ ] **Step 3: Commit**

```bash
git add docs/roadmap.md
git commit -m "docs(roadmap): move cache-breakpoint tuning to done"
```

---

## Lane 4 (Wave 1): gitignore-aware grep via git (#9)

**Files:**
- Modify: `internal/tools/grep.go`
- Modify: `docs/design.md` (§9.3 grep prose)
- Test: `internal/tools/grep_test.go`

Spec section 9. Read `internal/tools/grep.go` and `grep_test.go` first.

### Task 4.1: git-backed file listing

- [ ] **Step 1: Write the failing tests**

Append to `internal/tools/grep_test.go` (add imports `"os/exec"`, `"path/filepath"`):

```go
// initGitRepo turns dir into a git repo; skips the test when git is missing.
func initGitRepo(t *testing.T, dir string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "t@example.com"},
		{"config", "user.name", "t"},
	} {
		cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
}

func writeFileT(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestGrepRespectsGitignore(t *testing.T) {
	dir := t.TempDir()
	initGitRepo(t, dir)
	writeFileT(t, filepath.Join(dir, ".gitignore"), "secret.txt\nbuildout/\n")
	writeFileT(t, filepath.Join(dir, "kept.txt"), "needle here\n")
	writeFileT(t, filepath.Join(dir, "secret.txt"), "needle here\n")
	writeFileT(t, filepath.Join(dir, "buildout", "gen.txt"), "needle here\n")
	writeFileT(t, filepath.Join(dir, "sub", ".gitignore"), "local.txt\n")
	writeFileT(t, filepath.Join(dir, "sub", "local.txt"), "needle here\n")
	writeFileT(t, filepath.Join(dir, "sub", "kept.txt"), "needle here\n")

	out, err := grep{}.Run(context.Background(),
		json.RawMessage(`{"pattern":"needle","path":"`+dir+`"}`))
	if err != nil {
		t.Fatalf("grep: %v", err)
	}
	for _, want := range []string{"kept.txt:1:needle here", "sub/kept.txt:1:"} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
	for _, banned := range []string{"secret.txt", "buildout", "local.txt"} {
		if strings.Contains(out, banned) {
			t.Errorf("ignored file %q leaked into output:\n%s", banned, out)
		}
	}
}

func TestGrepNoIgnoreSearchesEverything(t *testing.T) {
	dir := t.TempDir()
	initGitRepo(t, dir)
	writeFileT(t, filepath.Join(dir, ".gitignore"), "secret.txt\n")
	writeFileT(t, filepath.Join(dir, "secret.txt"), "needle here\n")

	out, err := grep{}.Run(context.Background(),
		json.RawMessage(`{"pattern":"needle","path":"`+dir+`","no_ignore":true}`))
	if err != nil {
		t.Fatalf("grep: %v", err)
	}
	if !strings.Contains(out, "secret.txt:1:needle here") {
		t.Errorf("no_ignore should search ignored files:\n%s", out)
	}
}

func TestGrepNonRepoFallsBackToDenylist(t *testing.T) {
	dir := t.TempDir() // no git repo here
	writeFileT(t, filepath.Join(dir, "a.txt"), "needle here\n")
	writeFileT(t, filepath.Join(dir, "node_modules", "dep.js"), "needle here\n")

	out, err := grep{}.Run(context.Background(),
		json.RawMessage(`{"pattern":"needle","path":"`+dir+`"}`))
	if err != nil {
		t.Fatalf("grep: %v", err)
	}
	if !strings.Contains(out, "a.txt:1:needle here") {
		t.Errorf("missing a.txt match:\n%s", out)
	}
	if strings.Contains(out, "node_modules") {
		t.Errorf("denylist not applied in fallback:\n%s", out)
	}
}

func TestGrepGitListingHonorsGlob(t *testing.T) {
	dir := t.TempDir()
	initGitRepo(t, dir)
	writeFileT(t, filepath.Join(dir, "a.go"), "needle\n")
	writeFileT(t, filepath.Join(dir, "a.txt"), "needle\n")

	out, err := grep{}.Run(context.Background(),
		json.RawMessage(`{"pattern":"needle","path":"`+dir+`","glob":"*.go"}`))
	if err != nil {
		t.Fatalf("grep: %v", err)
	}
	if !strings.Contains(out, "a.go:1:needle") || strings.Contains(out, "a.txt") {
		t.Errorf("glob filter wrong:\n%s", out)
	}
}
```

If `grep_test.go` already defines a file-writing helper, use it instead of adding `writeFileT`.

- [ ] **Step 2: Run them; expect failure**

Run: `go test ./internal/tools/ -run TestGrep -v`
Expected: `TestGrepRespectsGitignore` FAIL — ignored files leak; the walker has no gitignore knowledge. The other three PASS already: they pin behavior (no_ignore semantics, non-repo fallback, glob filtering) that the change must not regress, and `TestGrepGitListingHonorsGlob` only starts exercising the git-listing path once it exists.

- [ ] **Step 3: Implement**

In `internal/tools/grep.go`:

1. Schema — add to `properties`:

```json
    "no_ignore": {"type": "boolean", "description": "Search ignored files too (gitignore filtering is the default inside git repos)."}
```

2. Description:

```go
func (grep) Description() string {
	return "Search file contents with a Go (RE2) regular expression. Recurses from a path; respects .gitignore inside git repos; prints path:line:text."
}
```

3. Args struct — add `NoIgnore bool \`json:"no_ignore"\``.

4. Add the lister (imports: `"os/exec"`, `"slices"`):

```go
// gitListFiles lists tracked plus untracked-but-not-ignored files under root
// (git grep --untracked semantics), paths relative to root, sorted. ok is
// false when root is not in a git work tree or git is unavailable; the caller
// falls back to the denylist walk. Ignore semantics — nesting, negation,
// global excludes — are git's own, which is the point (spec §9).
func gitListFiles(ctx context.Context, root string) ([]string, bool) {
	cmd := exec.CommandContext(ctx, "git", "-C", root,
		"ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", ".")
	out, err := cmd.Output()
	if err != nil {
		return nil, false
	}
	var files []string
	for _, f := range bytes.Split(out, []byte{0}) {
		if len(f) > 0 {
			files = append(files, string(f))
		}
	}
	slices.Sort(files)
	return files, true
}
```

5. In `Run`, restructure the directory branch: extract the existing `filepath.WalkDir` block into a helper `walkGrep` on the same receiver-less style (same body, unchanged), then:

```go
	if !info.IsDir() {
		if err := grepFile(ctx, root, root, re, emit); err != nil {
			return "", err
		}
	} else {
		listed := false
		if !args.NoIgnore {
			if files, ok := gitListFiles(ctx, root); ok {
				listed = true
				for _, rel := range files {
					if ctx.Err() != nil {
						return "", ctx.Err()
					}
					if args.Glob != "" {
						if ok, _ := path.Match(args.Glob, filepath.Base(rel)); !ok {
							continue
						}
					}
					display := filepath.Join(filepath.Base(root), rel)
					if err := grepFile(ctx, filepath.Join(root, rel), display, re, emit); err != nil {
						return "", err
					}
					if truncated {
						break
					}
				}
			}
		}
		if !listed {
			if err := walkGrep(ctx, root, args.Glob, re, emit, &truncated); err != nil {
				return "", err
			}
		}
	}
```

`walkGrep` signature and body (verbatim move of the current WalkDir block):

```go
// walkGrep is the denylist walk used outside git repos or with no_ignore.
func walkGrep(ctx context.Context, root, glob string, re *regexp.Regexp, emit func(string, int, string) bool, truncated *bool) error {
	walkErr := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if d != nil && d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if d.IsDir() {
			if p != root && grepDenylist[d.Name()] {
				return fs.SkipDir
			}
			return nil
		}
		if !d.Type().IsRegular() {
			return nil
		}
		if glob != "" {
			ok, _ := path.Match(glob, d.Name())
			if !ok {
				return nil
			}
		}
		rel, rerr := filepath.Rel(root, p)
		if rerr != nil {
			rel = p
		}
		rel = filepath.Join(filepath.Base(root), rel)
		if cerr := grepFile(ctx, p, rel, re, emit); cerr != nil {
			return cerr
		}
		if *truncated {
			return errStopWalk
		}
		return nil
	})
	if walkErr != nil && walkErr != errStopWalk {
		return walkErr
	}
	return nil
}
```

Note the closure detail: `truncated` is set inside `emit`; the walk helper reads it through the pointer, the listing loop reads the variable directly.

- [ ] **Step 4: Run tests; expect pass**

Run: `go test ./internal/tools/ -v`
Expected: PASS — all new tests plus every pre-existing grep test (format, caps, binary skip, single-file, denylist).

- [ ] **Step 5: Commit**

```bash
git add internal/tools/grep.go internal/tools/grep_test.go
git commit -m "feat(tools): make grep gitignore-aware via git ls-files, with no_ignore opt-out"
```

### Task 4.2: Design-doc prose + lane gate + roadmap

- [ ] **Step 1: Update `docs/design.md` §9.3**

Find the grep section (§9.3) and update its file-selection prose: inside a git work tree the candidate set is `git ls-files --cached --others --exclude-standard` (tracked + untracked-but-not-ignored); the fixed denylist walk remains the fallback outside repos, on git failure, or with `no_ignore: true`. Keep the rest (RE2, caps, binary sniff) as written.

- [ ] **Step 2: Full gate**

Run: `go build ./... && go vet ./... && go test ./...`
Expected: all PASS.

- [ ] **Step 3: Move roadmap item #9 to Done; commit**

Remove entry 9 from `docs/roadmap.md`, append to **Done**:

```markdown
6. **gitignore-aware grep.** Inside git repos, grep's candidate files come
   from `git ls-files --cached --others --exclude-standard`, delegating all
   ignore semantics to git; the denylist walk remains the non-repo /
   `no_ignore` fallback. The RE2 contract and output caps are unchanged.
```

```bash
git add docs/design.md docs/roadmap.md
git commit -m "docs: describe git-backed grep listing; move roadmap item to done"
```

---

## Lane 5 (Wave 2): Proactive compaction (#3) + maxSteps auto-continue (#4)

**Files:**
- Modify: `internal/agent/agent.go` (`RunTurn`, `Options`)
- Modify: `internal/config/config.go`
- Modify: `cmd/harness/main.go` (~line 342, `agent.Options` literal)
- Test: `internal/agent/compact_test.go`, `internal/agent/agent_test.go`, `internal/config/config_test.go`

Spec sections 4 and 5. Branch from main after wave 1 merges (lane 1 has rewritten parts of `RunTurn` — read it fresh; the step loop will contain `streamWithRetry`).

### Task 5.1: Proactive (mid-turn) compaction trigger

- [ ] **Step 1: Write the failing tests**

Append to `internal/agent/compact_test.go`:

```go
// seedTurns returns n complete small turns so compaction has history to fold.
func seedTurns(n int) []llm.Message {
	var msgs []llm.Message
	for i := 0; i < n; i++ {
		msgs = append(msgs,
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: "q"}}},
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: "a"}}},
		)
	}
	return msgs
}

func TestProactiveCompactionMidTurn(t *testing.T) {
	// Window 1000 tokens -> trigger at 780 tokens (3120 bytes estimated).
	// The tool result is 8000 bytes, so the estimate crosses the threshold
	// before step 2's request is built.
	big := strings.Repeat("x", 8000)
	tool := &recordTool{name: "blob", run: func(_ context.Context, _ json.RawMessage) (string, error) {
		return big, nil
	}}
	reg := &tools.Registry{}
	reg.Register(tool)

	fp := llmtest.New("fake",
		llmtest.Step{ // step 1: ask for the ballooning tool
			Events: []llm.StreamEvent{toolDone(0, "c1", "blob", `{}`)},
			Stop:   llm.StopToolUse,
			Usage:  llm.Usage{InputTokens: 10, OutputTokens: 2},
		},
		llmtest.Step{ // the mid-turn summary call
			Events: []llm.StreamEvent{textDelta("the summary")},
			Stop:   llm.StopEndTurn,
			Usage:  llm.Usage{InputTokens: 50, OutputTokens: 5},
		},
		llmtest.Step{ // step 2 proper, against the compacted transcript
			Events: []llm.StreamEvent{textDelta("done")},
			Stop:   llm.StopEndTurn,
			Usage:  llm.Usage{InputTokens: 20, OutputTokens: 3},
		},
	)
	a := newAgent(fp, reg, Options{ContextWindow: 1000})
	a.SetTranscript(seedTurns(5))
	sink := &recordSink{}

	if err := a.RunTurn(context.Background(), "go", sink); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	mustValid(t, a.Transcript())

	if len(fp.Requests) != 3 {
		t.Fatalf("provider called %d times, want 3 (step, summary, step)", len(fp.Requests))
	}
	// The post-compaction request starts with the summary message.
	first := fp.Requests[2].Messages[0]
	if !strings.HasPrefix(first.Content[0].Text, summaryHeader) {
		t.Errorf("post-compaction request should start with the summary, got %q", first.Content[0].Text)
	}
	var compacted bool
	for _, n := range sink.notices {
		if strings.Contains(n, "compacted:") {
			compacted = true
		}
	}
	if !compacted {
		t.Errorf("no compaction notice, notices=%v", sink.notices)
	}
	// Summary-call usage folds into the turn total (10+50+20 inputs).
	if got := sink.turnUsage[0].Usage.InputTokens; got != 80 {
		t.Errorf("turn input tokens = %d, want 80", got)
	}
}

func TestNoMidTurnCompactionUnderThreshold(t *testing.T) {
	tool := &recordTool{name: "small", run: func(_ context.Context, _ json.RawMessage) (string, error) {
		return "tiny", nil
	}}
	reg := &tools.Registry{}
	reg.Register(tool)

	fp := llmtest.New("fake",
		llmtest.Step{
			Events: []llm.StreamEvent{toolDone(0, "c1", "small", `{}`)},
			Stop:   llm.StopToolUse,
		},
		llmtest.Step{Events: []llm.StreamEvent{textDelta("done")}, Stop: llm.StopEndTurn},
	)
	a := newAgent(fp, reg, Options{ContextWindow: 1_000_000})
	a.SetTranscript(seedTurns(5))
	sink := &recordSink{}

	if err := a.RunTurn(context.Background(), "go", sink); err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	if len(fp.Requests) != 2 {
		t.Errorf("provider called %d times, want 2 (no summary call)", len(fp.Requests))
	}
	for _, n := range sink.notices {
		if strings.Contains(n, "compacted:") {
			t.Errorf("unexpected compaction: %v", sink.notices)
		}
	}
}
```

- [ ] **Step 2: Run them; expect failure**

Run: `go test ./internal/agent/ -run 'TestProactiveCompaction|TestNoMidTurnCompaction' -v`
Expected: `TestProactiveCompactionMidTurn` FAIL — only 2 provider calls and the transcript overflows unsummarized; the under-threshold test may pass (regression pin).

- [ ] **Step 3: Implement**

In `RunTurn`, at the top of the step loop (before the `req := llm.Request{...}` literal):

```go
	for steps < a.maxSteps {
		// Proactive trigger (spec §4): a turn whose tool results balloon the
		// context compacts before the next request, not after the turn. The
		// estimate catches growth the last reported count knows nothing about.
		if trigger := max(lastInput, estimateTokens(a.transcript)); trigger*100 >= a.window()*compactThresholdPct {
			if compUsage, err := a.Compact(ctx, sink); err == nil {
				total = add(total, compUsage)
				// The old reported count no longer describes the compacted
				// transcript and would re-trigger every step.
				lastInput = 0
			}
		}

		req := llm.Request{
```

And make `lastInput` a context-size signal (spec §4) — replace:

```go
		lastInput = res.usage.InputTokens
```

with:

```go
		// Context-size signal, not billing: cached tokens occupy the window too.
		lastInput = res.usage.InputTokens + res.usage.CacheReadTokens + res.usage.CacheWriteTokens
```

- [ ] **Step 4: Run tests; expect pass**

Run: `go test ./internal/agent/ -v`
Expected: PASS, including all pre-existing compaction tests.

- [ ] **Step 5: Commit**

```bash
git add internal/agent/agent.go internal/agent/compact_test.go
git commit -m "feat(agent): compact proactively inside the step loop"
```

### Task 5.2: `on_max_steps` config knob

- [ ] **Step 1: Write the failing config tests**

Append to `internal/config/config_test.go` (follow the file's existing Load-test helper conventions):

```go
func TestOnMaxStepsResolution(t *testing.T) {
	getenv := func(string) string { return "" }

	c, err := Load(nil, getenv, "")
	if err != nil {
		t.Fatal(err)
	}
	if c.OnMaxSteps != "stop" {
		t.Errorf("default OnMaxSteps = %q, want \"stop\"", c.OnMaxSteps)
	}

	c, err = Load([]string{"-on-max-steps", "continue"}, getenv, "")
	if err != nil {
		t.Fatal(err)
	}
	if c.OnMaxSteps != "continue" {
		t.Errorf("flag OnMaxSteps = %q, want \"continue\"", c.OnMaxSteps)
	}

	env := func(k string) string {
		if k == "HARNESS_ON_MAX_STEPS" {
			return "continue"
		}
		return ""
	}
	c, err = Load(nil, env, "")
	if err != nil {
		t.Fatal(err)
	}
	if c.OnMaxSteps != "continue" {
		t.Errorf("env OnMaxSteps = %q, want \"continue\"", c.OnMaxSteps)
	}

	if _, err := Load([]string{"-on-max-steps", "bogus"}, getenv, ""); err == nil {
		t.Error("invalid on-max-steps value should error")
	}
}
```

- [ ] **Step 2: Run; expect failure**

Run: `go test ./internal/config/ -run TestOnMaxSteps -v`
Expected: compile FAIL — `OnMaxSteps` undefined.

- [ ] **Step 3: Implement**

In `internal/config/config.go`:

- `Config`: add under "Loop / model limits":

```go
	OnMaxSteps string // -on-max-steps: "stop" (default) or "continue"
```

- `fileConfig`: add `OnMaxSteps string \`json:"on_max_steps"\``.
- `flags` struct: add `onMaxSteps *string`; in `newFlagSet`:

```go
	f.onMaxSteps = fs.String("on-max-steps", "", "when the step budget is hit: stop (default) or continue (up to 3 fresh budgets)")
```

- In `Load`, next to the MaxSteps resolution:

```go
	c.OnMaxSteps = strings.ToLower(strings.TrimSpace(resolveString(set["on-max-steps"], *f.onMaxSteps,
		getenv("HARNESS_ON_MAX_STEPS"), fc.OnMaxSteps, "stop")))
	if c.OnMaxSteps != "stop" && c.OnMaxSteps != "continue" {
		return Config{}, fmt.Errorf("invalid -on-max-steps %q (valid: stop, continue)", c.OnMaxSteps)
	}
```

- [ ] **Step 4: Run; expect pass**

Run: `go test ./internal/config/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/config/config.go internal/config/config_test.go
git commit -m "feat(config): add on_max_steps knob (stop|continue)"
```

### Task 5.3: Auto-continue in the agent loop

- [ ] **Step 1: Write the failing test**

Append to `internal/agent/agent_test.go`:

```go
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
```

(`TestMaxStepsStop` already pins the default-off behavior.)

- [ ] **Step 2: Run; expect failure**

Run: `go test ./internal/agent/ -run TestAutoContinue -v`
Expected: compile FAIL — `AutoContinue` undefined.

- [ ] **Step 3: Implement**

In `internal/agent/agent.go`:

- `Options`: add

```go
	// AutoContinue grants up to maxAutoContinues fresh step budgets when the
	// model still wants tools at the cap, instead of stopping (spec §5).
	AutoContinue bool
```

- `Agent`: add `autoContinue bool`; set from opts in `New`.
- Add const:

```go
// maxAutoContinues bounds AutoContinue budgets per turn so a pathological
// loop still terminates (spec §5).
const maxAutoContinues = 3
```

- In `RunTurn`, introduce a budget that replaces both uses of `a.maxSteps` in loop control:

```go
	budget := a.maxSteps
	continues := 0
	for steps < budget {
```

and the post-dispatch check becomes:

```go
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
```

- [ ] **Step 4: Run; expect pass**

Run: `go test ./internal/agent/ -v`
Expected: PASS, including `TestMaxStepsStop`.

- [ ] **Step 5: Wire main**

In `cmd/harness/main.go` (~line 342), the `agent.Options` literal gains:

```go
		AutoContinue:  cfg.OnMaxSteps == "continue",
```

Run: `go build ./... && go test ./cmd/...`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/agent/agent.go internal/agent/agent_test.go cmd/harness/main.go
git commit -m "feat(agent): optional auto-continue past the step budget"
```

### Task 5.4: Lane gate + roadmap

- [ ] **Step 1: Full gate**

Run: `go build ./... && go vet ./... && go test ./...`
Expected: all PASS.

- [ ] **Step 2: Move roadmap items #3 and #4 to Done; commit**

Remove entries 3 and 4 from `docs/roadmap.md`, append to **Done**:

```markdown
7. **Proactive compaction.** The step loop estimates the transcript before
   every request and compacts mid-turn at the same 78% threshold; the
   reported-tokens signal now counts cached tokens too.
8. **maxSteps auto-continue.** `on_max_steps = continue` grants up to 3 fresh
   step budgets before stopping; default behavior is unchanged.
```

```bash
git add docs/roadmap.md
git commit -m "docs(roadmap): move proactive compaction and auto-continue to done"
```

---

## Lane 6 (Wave 3): Parallel read-only dispatch (#8)

**Files:**
- Modify: `internal/tools/tool.go` (interface + `AllReadOnly`), every `Tool` implementation
- Modify: `internal/agent/agent.go` (dispatch section of `RunTurn`)
- Modify: `docs/design.md` (§8 dispatch prose)
- Test: `internal/agent/agent_test.go`, `internal/tools/tool_test.go`

Spec section 8. Branch from main after wave 2 merges. Extending the `Tool` interface breaks every implementor until all are updated — the compiler is the checklist.

### Task 6.1: `ReadOnly()` on the Tool interface

- [ ] **Step 1: Extend the interface**

In `internal/tools/tool.go`:

```go
type Tool interface {
	Name() string
	Description() string     // model-facing, one line
	Schema() json.RawMessage // JSON Schema for the input object
	// ReadOnly reports that Run never mutates workspace or repo state, so
	// calls may dispatch concurrently with other read-only calls (spec §8).
	ReadOnly() bool
	Run(ctx context.Context, input json.RawMessage) (string, error)
}
```

- [ ] **Step 2: Implement on every tool**

Find all implementations: `go build ./... 2>&1` now lists every type missing `ReadOnly` — or `grep -rn ") Schema() json.RawMessage" --include="*.go"`. Add one line per production tool:

```go
func (readFile) ReadOnly() bool      { return true }   // readfile.go
func (listDir) ReadOnly() bool       { return true }   // listdir.go
func (grep) ReadOnly() bool          { return true }   // grep.go
func (edit) ReadOnly() bool          { return false }  // edit.go
func (writeFile) ReadOnly() bool     { return false }  // writefile.go
func (applyPatch) ReadOnly() bool    { return false }  // applypatch.go
func (runCommand) ReadOnly() bool    { return false }  // runcommand.go
func (execTool) ReadOnly() bool      { return false }  // exec.go
func (gitTool) ReadOnly() bool       { return false }  // git.go
func (webFetch) ReadOnly() bool      { return true }   // webfetch.go (GET; no workspace mutation)
func (gitReadonly) ReadOnly() bool   { return true }   // gitreadonly.go
func (*writeTmpFile) ReadOnly() bool { return false }  // writetmpfile.go (match its existing receiver form)
```

Check each tool's existing receiver form (value vs pointer) and match it. Test fakes also need the method: `recordTool` in `internal/agent/agent_test.go` gains a `readOnly bool` field and `func (t *recordTool) ReadOnly() bool { return t.readOnly }`; any stub tools in `tool_test.go`, `dispatchtimeout_test.go`, and elsewhere (`go vet ./...` finds them) get an explicit `ReadOnly() bool` too.

Add to `internal/tools/tool.go`:

```go
// AllReadOnly reports whether every call resolves to a read-only tool.
// Unknown names count as not read-only: they dispatch to an error result,
// and serializing them is the conservative choice.
func (r *Registry) AllReadOnly(calls []llm.ToolCall) bool {
	for _, c := range calls {
		t, ok := r.tools[c.Name]
		if !ok || !t.ReadOnly() {
			return false
		}
	}
	return true
}
```

- [ ] **Step 3: Build + test**

Run: `go build ./... && go test ./...`
Expected: PASS (behavior unchanged so far).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(tools): declare read-only on every tool and registry"
```

### Task 6.2: Concurrent dispatch for all-read-only steps

- [ ] **Step 1: Write the failing tests**

Append to `internal/agent/agent_test.go` (add import `"sync"`):

```go
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
```

`recordTool.inputs` is appended from concurrent goroutines now — make `Run` lock: add a `mu sync.Mutex` field to `recordTool` and guard the `t.inputs` append.

- [ ] **Step 2: Run; expect failure**

Run: `go test ./internal/agent/ -run 'TestAllReadOnlyStep|TestMixedStep' -v`
Expected: `TestAllReadOnlyStepDispatchesConcurrently` FAIL — barrier timeout error results (~2s) under sequential dispatch. `TestMixedStepStaysSequential` should PASS (regression pin).

- [ ] **Step 3: Implement**

In `internal/agent/agent.go` (add import `"sync"`): add

```go
// maxParallelTools bounds concurrent read-only dispatch (spec §8).
const maxParallelTools = 8
```

Replace the dispatch block in `RunTurn`:

```go
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
```

with:

```go
		results := a.dispatchCalls(ctx, res.toolCalls, sink)
```

and add:

```go
// dispatchCalls runs one step's tool calls: concurrently when the step is
// all-read-only with 2+ calls, sequentially otherwise. Sink events and the
// returned blocks are in emission order either way, and the sink is only
// ever called from this goroutine (spec §8).
func (a *Agent) dispatchCalls(ctx context.Context, calls []llm.ToolCall, sink EventSink) []llm.ContentBlock {
	blocks := make([]llm.ContentBlock, len(calls))

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
		}
		return blocks
	}

	for i, call := range calls {
		sink.ToolStart(call)
		r := a.tools.Dispatch(ctx, call)
		sink.ToolResult(r)
		blocks[i] = resultBlock(r)
	}
	return blocks
}

func resultBlock(r llm.ToolResult) llm.ContentBlock {
	return llm.ContentBlock{
		Kind:        llm.BlockToolResult,
		ResultForID: r.ForID,
		ResultText:  r.Text,
		ResultError: r.IsError,
	}
}
```

- [ ] **Step 4: Run with the race detector; expect pass**

Run: `go test -race ./internal/agent/ ./internal/tools/ -v`
Expected: PASS, no data races. (If `-race` flags `recordTool.inputs`, the mutex from Step 1 is missing.)

- [ ] **Step 5: Commit**

```bash
git add internal/agent/agent.go internal/agent/agent_test.go
git commit -m "feat(agent): dispatch all-read-only tool steps concurrently"
```

### Task 6.3: Design prose + lane gate + roadmap

- [ ] **Step 1: Update `docs/design.md` §8**

Where §8.1 describes sequential tool execution in emission order, add: a step whose calls are all read-only (2+ of them) dispatches concurrently, bounded at 8; results, sink events, and transcript blocks remain in emission order; mixed steps stay sequential.

- [ ] **Step 2: Full gate**

Run: `go build ./... && go vet ./... && go test -race ./...`
Expected: all PASS.

- [ ] **Step 3: Move roadmap item #8 to Done; commit**

Remove entry 8 from `docs/roadmap.md` (and the now-empty "Flagged" section if nothing remains), append to **Done**:

```markdown
9. **Parallel dispatch of read-only tool calls.** Steps whose calls are all
   read-only dispatch concurrently (bounded at 8); ordering of sink events,
   results, and transcript blocks is unchanged. Mixed steps stay sequential.
```

```bash
git add docs/design.md docs/roadmap.md
git commit -m "docs: describe parallel read-only dispatch; move roadmap item to done"
```

---

## Final integration (orchestrator)

- [ ] After each wave: review lane diffs, `git merge --no-ff` each branch into main, run `go build ./... && go vet ./... && go test -race ./...` on main.
- [ ] After wave 3: confirm `docs/roadmap.md` shows items 2–9 under Done and the open sections are empty; reconcile Done-list numbering across lanes (lanes 1–4 edit it concurrently — expect a trivial merge conflict in `docs/roadmap.md`; resolve by keeping all entries and renumbering).
- [ ] Run the smoke checklist in `docs/smoke.md` against a live provider if credentials are available.
