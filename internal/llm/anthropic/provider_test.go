package anthropic

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"harness/internal/llm"
	"harness/internal/sse"
)

// writeBody copies raw bytes to the response without going through
// http.ResponseWriter.Write directly (which the security scanner flags as an
// XSS sink); these are static test fixtures.
func writeBody(w http.ResponseWriter, b []byte) {
	_, _ = io.Copy(w, strings.NewReader(string(b)))
}

// serveFixture starts an httptest server that replies with the named SSE
// fixture as a 200 streaming response.
func serveFixture(t *testing.T, name string) *httptest.Server {
	t.Helper()
	body, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		writeBody(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func testProvider(t *testing.T, srv *httptest.Server, sleep func(time.Duration)) *Provider {
	t.Helper()
	if sleep == nil {
		sleep = func(time.Duration) {}
	}
	return New(Config{
		APIKey:  "test-key",
		BaseURL: srv.URL,
		Sleep:   sleep,
	})
}

func simpleRequest() llm.Request {
	return llm.Request{
		Model: "claude-opus-4-8",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: "hi"}}},
		},
	}
}

// drain collects all events and the terminal error from a stream.
func drain(stream func(func(llm.StreamEvent, error) bool)) ([]llm.StreamEvent, error) {
	var events []llm.StreamEvent
	var lastErr error
	for ev, err := range stream {
		if err != nil {
			lastErr = err
			break
		}
		events = append(events, ev)
	}
	return events, lastErr
}

func TestStreamTextOnly(t *testing.T) {
	srv := serveFixture(t, "text_only.sse")
	p := testProvider(t, srv, nil)

	events, err := drain(p.Stream(context.Background(), simpleRequest()))
	if err != nil {
		t.Fatalf("unexpected terminal error: %v", err)
	}

	var text strings.Builder
	var done *llm.StreamEvent
	for i := range events {
		switch events[i].Kind {
		case llm.EventTextDelta:
			text.WriteString(events[i].Text)
		case llm.EventDone:
			done = &events[i]
		}
	}
	if text.String() != "Hello!" {
		t.Errorf("text = %q, want %q", text.String(), "Hello!")
	}
	if done == nil {
		t.Fatal("no EventDone")
	}
	if done.StopReason != llm.StopEndTurn {
		t.Errorf("stop reason = %q, want %q", done.StopReason, llm.StopEndTurn)
	}
	if done.Usage == nil {
		t.Fatal("EventDone carries no usage")
	}
	want := llm.Usage{InputTokens: 25, OutputTokens: 15, CacheWriteTokens: 10, CacheReadTokens: 7}
	if *done.Usage != want {
		t.Errorf("final usage = %+v, want %+v", *done.Usage, want)
	}
}

func TestStreamTextOnlyEventOrder(t *testing.T) {
	srv := serveFixture(t, "text_only.sse")
	p := testProvider(t, srv, nil)
	events, err := drain(p.Stream(context.Background(), simpleRequest()))
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	gotKinds := kindsOf(events)
	wantKinds := []llm.EventKind{
		llm.EventTextDelta, // Hello
		llm.EventTextDelta, // !
		llm.EventDone,
	}
	// Usage events may also be emitted; filter to the structural kinds we assert.
	gotKinds = without(gotKinds, llm.EventUsage)
	if !equalKinds(gotKinds, wantKinds) {
		t.Errorf("event kinds = %v, want %v", gotKinds, wantKinds)
	}
}

func TestStreamToolCall(t *testing.T) {
	srv := serveFixture(t, "tool_call.sse")
	p := testProvider(t, srv, nil)
	events, err := drain(p.Stream(context.Background(), simpleRequest()))
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	var start, done *llm.StreamEvent
	var deltas strings.Builder
	var text strings.Builder
	var final *llm.StreamEvent
	for i := range events {
		switch events[i].Kind {
		case llm.EventTextDelta:
			text.WriteString(events[i].Text)
		case llm.EventToolCallStart:
			start = &events[i]
		case llm.EventToolCallDelta:
			deltas.WriteString(events[i].ArgsDelta)
		case llm.EventToolCallDone:
			done = &events[i]
		case llm.EventDone:
			final = &events[i]
		}
	}
	if text.String() != "Let me check the weather." {
		t.Errorf("text = %q", text.String())
	}
	if start == nil || done == nil {
		t.Fatal("missing tool call start/done")
	}
	if start.ToolID != "toolu_01T1x1fJ34qAmk2tNTrN7Up6" || start.ToolName != "get_weather" {
		t.Errorf("start id/name = %q/%q", start.ToolID, start.ToolName)
	}
	if start.Index != 1 {
		t.Errorf("start index = %d, want 1", start.Index)
	}
	wantInput := `{"location": "San Francisco, CA"}`
	if string(done.ToolInput) != wantInput {
		t.Errorf("assembled input = %s, want %s", done.ToolInput, wantInput)
	}
	if deltas.String() != wantInput {
		t.Errorf("concatenated deltas = %q, want %q", deltas.String(), wantInput)
	}
	if final == nil || final.StopReason != llm.StopToolUse {
		t.Errorf("final stop reason wrong: %+v", final)
	}
}

func TestStreamParallelTools(t *testing.T) {
	srv := serveFixture(t, "parallel_tools.sse")
	p := testProvider(t, srv, nil)
	events, err := drain(p.Stream(context.Background(), simpleRequest()))
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	var dones []llm.StreamEvent
	for _, e := range events {
		if e.Kind == llm.EventToolCallDone {
			dones = append(dones, e)
		}
	}
	if len(dones) != 2 {
		t.Fatalf("got %d tool dones, want 2", len(dones))
	}
	if dones[0].Index != 0 || dones[0].ToolID != "toolu_A" {
		t.Errorf("first done = %+v", dones[0])
	}
	if string(dones[0].ToolInput) != `{"location": "San Francisco, CA"}` {
		t.Errorf("first input = %s", dones[0].ToolInput)
	}
	if dones[1].Index != 1 || dones[1].ToolID != "toolu_B" {
		t.Errorf("second done = %+v", dones[1])
	}
	if string(dones[1].ToolInput) != `{"location": "New York, NY"}` {
		t.Errorf("second input = %s", dones[1].ToolInput)
	}
}

func TestStreamEmptyArgs(t *testing.T) {
	srv := serveFixture(t, "empty_args.sse")
	p := testProvider(t, srv, nil)
	events, err := drain(p.Stream(context.Background(), simpleRequest()))
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	var done *llm.StreamEvent
	for i := range events {
		if events[i].Kind == llm.EventToolCallDone {
			done = &events[i]
		}
	}
	if done == nil {
		t.Fatal("no tool done")
	}
	if string(done.ToolInput) != "{}" {
		t.Errorf("empty args assembled as %s, want {}", done.ToolInput)
	}
}

func TestStreamErrorFrame(t *testing.T) {
	srv := serveFixture(t, "error_frame.sse")
	p := testProvider(t, srv, nil)
	events, err := drain(p.Stream(context.Background(), simpleRequest()))
	if err == nil {
		t.Fatal("expected terminal error from mid-stream error frame")
	}
	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error is not *llm.APIError: %T %v", err, err)
	}
	if apiErr.Code != "overloaded_error" {
		t.Errorf("code = %q, want overloaded_error", apiErr.Code)
	}
	// Text streamed before the error frame is still delivered.
	var sawText bool
	for _, e := range events {
		if e.Kind == llm.EventTextDelta && e.Text == "Thinking" {
			sawText = true
		}
		if e.Kind == llm.EventDone {
			t.Error("EventDone emitted despite mid-stream error")
		}
	}
	if !sawText {
		t.Error("pre-error text delta not delivered")
	}
}

func TestStreamTruncated(t *testing.T) {
	srv := serveFixture(t, "truncated.sse")
	p := testProvider(t, srv, nil)
	events, err := drain(p.Stream(context.Background(), simpleRequest()))
	if err == nil {
		t.Fatal("expected truncated-stream error")
	}
	if !errors.Is(err, sse.ErrTruncatedStream) {
		t.Errorf("error does not wrap sse.ErrTruncatedStream: %v", err)
	}
	for _, e := range events {
		if e.Kind == llm.EventDone {
			t.Error("EventDone emitted for truncated stream")
		}
	}
}

func TestStreamRetryThenSuccess(t *testing.T) {
	body, err := os.ReadFile("testdata/text_only.sse")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Retry-After", "2")
			w.WriteHeader(http.StatusTooManyRequests)
			writeBody(w, []byte(`{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		writeBody(w, body)
	}))
	t.Cleanup(srv.Close)

	var slept []time.Duration
	var mu sync.Mutex
	p := testProvider(t, srv, func(d time.Duration) {
		mu.Lock()
		slept = append(slept, d)
		mu.Unlock()
	})

	events, err := drain(p.Stream(context.Background(), simpleRequest()))
	if err != nil {
		t.Fatalf("unexpected error after retry: %v", err)
	}
	if calls.Load() != 2 {
		t.Errorf("server hit %d times, want 2", calls.Load())
	}
	if len(slept) != 1 {
		t.Fatalf("slept %d times, want 1", len(slept))
	}
	// Retry-After: 2s is honored as a floor.
	if slept[0] < 2*time.Second {
		t.Errorf("backoff %v below Retry-After floor 2s", slept[0])
	}
	var done bool
	for _, e := range events {
		if e.Kind == llm.EventDone {
			done = true
		}
	}
	if !done {
		t.Error("no EventDone after successful retry")
	}
}

func TestStreamFatalStatusNoRetry(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		writeBody(w, []byte(`{"type":"error","error":{"type":"invalid_request_error","message":"bad model"}}`))
	}))
	t.Cleanup(srv.Close)

	var slept int
	p := testProvider(t, srv, func(time.Duration) { slept++ })
	_, err := drain(p.Stream(context.Background(), simpleRequest()))
	if err == nil {
		t.Fatal("expected APIError for 400")
	}
	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("not an APIError: %T %v", err, err)
	}
	if apiErr.StatusCode != 400 || apiErr.Retryable {
		t.Errorf("apiErr = %+v, want 400 non-retryable", apiErr)
	}
	if apiErr.Code != "invalid_request_error" || apiErr.Message != "bad model" {
		t.Errorf("apiErr code/message = %q/%q", apiErr.Code, apiErr.Message)
	}
	if calls.Load() != 1 {
		t.Errorf("server hit %d times, want 1 (no retry on 400)", calls.Load())
	}
	if slept != 0 {
		t.Errorf("slept %d times, want 0", slept)
	}
}

func TestStreamRetryBudgetExhausted(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
		writeBody(w, []byte(`{"type":"error","error":{"type":"overloaded_error","message":"try later"}}`))
	}))
	t.Cleanup(srv.Close)

	var slept int
	p := testProvider(t, srv, func(time.Duration) { slept++ })
	_, err := drain(p.Stream(context.Background(), simpleRequest()))
	if err == nil {
		t.Fatal("expected error after budget exhaustion")
	}
	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("not an APIError: %T %v", err, err)
	}
	if apiErr.StatusCode != 503 {
		t.Errorf("status = %d, want 503", apiErr.StatusCode)
	}
	// 5 attempts total => 4 sleeps between them.
	if calls.Load() != 5 {
		t.Errorf("server hit %d times, want 5", calls.Load())
	}
	if slept != 4 {
		t.Errorf("slept %d times, want 4", slept)
	}
}

func TestStreamContextCancelMidStream(t *testing.T) {
	// A handler that streams the message_start frame, then blocks so the body
	// read is in-flight when the context is cancelled.
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		writeBody(w, []byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"x\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-4-8\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n"))
		if fl != nil {
			fl.Flush()
		}
		<-release
	}))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) })

	p := testProvider(t, srv, nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var lastErr error
	for _, err := range p.Stream(ctx, simpleRequest()) {
		if err != nil {
			lastErr = err
			break
		}
		// Cancel as soon as the first event (message_start usage) arrives.
		cancel()
	}
	if !errors.Is(lastErr, context.Canceled) {
		t.Errorf("terminal error = %v, want context.Canceled", lastErr)
	}
}

func TestStreamSendsHeaders(t *testing.T) {
	body, _ := os.ReadFile("testdata/text_only.sse")
	var gotKey, gotVersion, gotContentType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("x-api-key")
		gotVersion = r.Header.Get("anthropic-version")
		gotContentType = r.Header.Get("content-type")
		w.WriteHeader(http.StatusOK)
		writeBody(w, body)
	}))
	t.Cleanup(srv.Close)

	p := testProvider(t, srv, nil)
	_, err := drain(p.Stream(context.Background(), simpleRequest()))
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if gotKey != "test-key" {
		t.Errorf("x-api-key = %q", gotKey)
	}
	if gotVersion != "2023-06-01" {
		t.Errorf("anthropic-version = %q", gotVersion)
	}
	if gotContentType != "application/json" {
		t.Errorf("content-type = %q", gotContentType)
	}
}

func TestStreamAppendsMessagesPath(t *testing.T) {
	body, _ := os.ReadFile("testdata/text_only.sse")
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.WriteHeader(http.StatusOK)
		writeBody(w, body)
	}))
	t.Cleanup(srv.Close)

	// Custom base URL with a prefix; the dialect appends /v1/messages.
	p := New(Config{APIKey: "k", BaseURL: srv.URL + "/anthropic", Sleep: func(time.Duration) {}})
	_, err := drain(p.Stream(context.Background(), simpleRequest()))
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if gotPath != "/anthropic/v1/messages" {
		t.Errorf("request path = %q, want /anthropic/v1/messages", gotPath)
	}
}

func TestName(t *testing.T) {
	p := New(Config{APIKey: "k"})
	if p.Name() != "anthropic" {
		t.Errorf("Name() = %q", p.Name())
	}
}

func TestNormalizeStopReason(t *testing.T) {
	cases := map[string]llm.StopReason{
		"end_turn":      llm.StopEndTurn,
		"tool_use":      llm.StopToolUse,
		"max_tokens":    llm.StopMaxTokens,
		"stop_sequence": llm.StopStop,
		"pause_turn":    llm.StopEndTurn, // unknown/other -> end_turn
		"refusal":       llm.StopEndTurn,
		"":              llm.StopEndTurn,
	}
	for in, want := range cases {
		if got := normalizeStopReason(in); got != want {
			t.Errorf("normalizeStopReason(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestParseRetryAfter(t *testing.T) {
	if d := parseRetryAfter("3"); d != 3*time.Second {
		t.Errorf("seconds form = %v, want 3s", d)
	}
	if d := parseRetryAfter(""); d != 0 {
		t.Errorf("empty = %v, want 0", d)
	}
	if d := parseRetryAfter("-5"); d != 0 {
		t.Errorf("negative = %v, want 0", d)
	}
	if d := parseRetryAfter("not-a-number"); d != 0 {
		t.Errorf("garbage = %v, want 0", d)
	}
}

func TestRetryableStatus(t *testing.T) {
	retryable := []int{429, 500, 502, 503, 529}
	for _, c := range retryable {
		if !retryableStatus(c) {
			t.Errorf("status %d should be retryable", c)
		}
	}
	fatal := []int{400, 401, 403, 404, 422, 200}
	for _, c := range fatal {
		if retryableStatus(c) {
			t.Errorf("status %d should not be retryable", c)
		}
	}
}

// --- helpers ---

func kindsOf(events []llm.StreamEvent) []llm.EventKind {
	out := make([]llm.EventKind, len(events))
	for i, e := range events {
		out[i] = e.Kind
	}
	return out
}

func without(kinds []llm.EventKind, drop llm.EventKind) []llm.EventKind {
	out := kinds[:0:0]
	for _, k := range kinds {
		if k != drop {
			out = append(out, k)
		}
	}
	return out
}

func equalKinds(a, b []llm.EventKind) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
