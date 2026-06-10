package ui

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"harness/internal/agent"
	"harness/internal/llm"
)

// fixedClock returns successive instants spaced by step, so duration math in the
// usage line is deterministic without sleeping (design §13).
func fixedClock(start time.Time, step time.Duration) func() time.Time {
	t := start
	first := true
	return func() time.Time {
		if first {
			first = false
			return t
		}
		t = t.Add(step)
		return t
	}
}

func TestToolSummaryLine(t *testing.T) {
	var out, errw bytes.Buffer
	r := NewRenderer(&out, &errw, RenderOptions{})

	r.ToolStart(llm.ToolCall{
		ID:    "c1",
		Name:  "grep",
		Input: json.RawMessage(`{"pattern":"func main","path":"."}`),
	})
	r.ToolResult(llm.ToolResult{
		ForID: "c1",
		Text:  "a.go:1:func main\nb.go:2:func main\n",
	})

	got := errw.String()
	if out.Len() != 0 {
		t.Errorf("tool lines must go to errw, not out; out=%q", out.String())
	}
	if !strings.HasPrefix(got, "[grep]") {
		t.Errorf("summary should start with [grep], got %q", got)
	}
	if !strings.Contains(got, `pattern="func main"`) {
		t.Errorf("summary should quote the pattern arg, got %q", got)
	}
	if !strings.Contains(got, "path=.") {
		t.Errorf("summary should show path arg, got %q", got)
	}
	if !strings.Contains(got, "→") {
		t.Errorf("summary should show the arrow separator, got %q", got)
	}
	if !strings.Contains(got, "2 lines") {
		t.Errorf("summary should report 2 lines, got %q", got)
	}
}

func TestToolSummaryErrorMarked(t *testing.T) {
	var out, errw bytes.Buffer
	r := NewRenderer(&out, &errw, RenderOptions{})
	r.ToolStart(llm.ToolCall{ID: "e1", Name: "edit", Input: json.RawMessage(`{"path":"x"}`)})
	r.ToolResult(llm.ToolResult{ForID: "e1", Text: "error: old_string not found in x", IsError: true})

	got := errw.String()
	if !strings.Contains(got, "error") {
		t.Errorf("error result should surface the error text, got %q", got)
	}
}

func TestVerboseAddsSnippet(t *testing.T) {
	var out, errw bytes.Buffer
	r := NewRenderer(&out, &errw, RenderOptions{Verbose: true})
	r.ToolStart(llm.ToolCall{ID: "c1", Name: "read_file", Input: json.RawMessage(`{"path":"a.go"}`)})
	body := "line1\nline2\nline3\nline4\nline5\nline6\nline7\n"
	r.ToolResult(llm.ToolResult{ForID: "c1", Text: body})

	got := errw.String()
	if !strings.Contains(got, "line1") || !strings.Contains(got, "line5") {
		t.Errorf("verbose should include the first ~5 lines, got %q", got)
	}
	if strings.Contains(got, "line6") {
		t.Errorf("verbose should cap the snippet at ~5 lines, got %q", got)
	}
}

func TestUsageLineKnownModelShowsCost(t *testing.T) {
	var out, errw bytes.Buffer
	start := time.Date(2026, 6, 9, 0, 0, 0, 0, time.UTC)
	r := NewRenderer(&out, &errw, RenderOptions{
		Model: "claude-opus-4-8",
		Registry: llm.NewRegistry(map[string]llm.ModelInfo{
			"claude-opus-4-8": {
				ContextWindow: 1_000_000,
				Price:         llm.Price{Input: 5.0, Output: 25.0},
			},
		}),
		Now: fixedClock(start, 4300*time.Millisecond),
	})
	r.StartTurn()
	r.TurnComplete(agent.TurnUsage{
		Steps: 3,
		Usage: llm.Usage{InputTokens: 12400, OutputTokens: 1800},
	})

	got := errw.String()
	if out.Len() != 0 {
		t.Errorf("usage line must go to errw, not out; out=%q", out.String())
	}
	if !strings.Contains(got, "[turn:") {
		t.Errorf("usage line should be bracketed, got %q", got)
	}
	if !strings.Contains(got, "3 steps") {
		t.Errorf("usage line should show step count, got %q", got)
	}
	if !strings.Contains(got, "12.4k in") || !strings.Contains(got, "1.8k out") {
		t.Errorf("usage line should show token counts, got %q", got)
	}
	if !strings.Contains(got, "$") {
		t.Errorf("known model should show a cost, got %q", got)
	}
	if !strings.Contains(got, "4.3s") {
		t.Errorf("usage line should show elapsed duration, got %q", got)
	}
}

func TestUsageLineUnknownModelOmitsCost(t *testing.T) {
	var out, errw bytes.Buffer
	start := time.Date(2026, 6, 9, 0, 0, 0, 0, time.UTC)
	r := NewRenderer(&out, &errw, RenderOptions{
		Model: "some-local-llama",
		Now:   fixedClock(start, time.Second),
	})
	r.StartTurn()
	r.TurnComplete(agent.TurnUsage{Steps: 1, Usage: llm.Usage{InputTokens: 100, OutputTokens: 10}})

	got := errw.String()
	if strings.Contains(got, "$") {
		t.Errorf("unknown model must omit cost, got %q", got)
	}
}

func TestColorSuppressedWhenNotTTY(t *testing.T) {
	var out, errw bytes.Buffer
	r := NewRenderer(&out, &errw, RenderOptions{Color: false})
	r.ToolStart(llm.ToolCall{ID: "c1", Name: "list_dir", Input: json.RawMessage(`{"path":"."}`)})
	r.ToolResult(llm.ToolResult{ForID: "c1", Text: "a\nb\n"})
	if strings.Contains(errw.String(), "\x1b[") {
		t.Errorf("no ANSI escapes when color disabled, got %q", errw.String())
	}
}

func TestColorEmittedWhenEnabled(t *testing.T) {
	var out, errw bytes.Buffer
	r := NewRenderer(&out, &errw, RenderOptions{Color: true})
	r.ToolStart(llm.ToolCall{ID: "c1", Name: "list_dir", Input: json.RawMessage(`{"path":"."}`)})
	r.ToolResult(llm.ToolResult{ForID: "c1", Text: "a\nb\n"})
	if !strings.Contains(errw.String(), "\x1b[") {
		t.Errorf("expected ANSI dim escapes when color enabled, got %q", errw.String())
	}
}

func TestTextDeltaGoesToStdout(t *testing.T) {
	var out, errw bytes.Buffer
	r := NewRenderer(&out, &errw, RenderOptions{})
	r.TextDelta("hello ")
	r.TextDelta("world")
	if out.String() != "hello world" {
		t.Errorf("assistant text should stream raw to out, got %q", out.String())
	}
	if errw.Len() != 0 {
		t.Errorf("assistant text must not touch errw, got %q", errw.String())
	}
}
