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
		Input: json.RawMessage(`{"args":["-R","-n","func main","."]}`),
	})
	r.ToolResult(llm.ToolResult{
		ForID: "c1",
		Text:  "a.go:1:func main\nb.go:2:func main\n",
	})

	got := errw.String()
	if out.Len() != 0 {
		t.Errorf("tool lines must go to errw, not out; out=%q", out.String())
	}
	if !strings.Contains(got, "[tool: grep started") {
		t.Errorf("tool start should be reported, got %q", got)
	}
	if !strings.Contains(got, "[grep]") {
		t.Errorf("summary should include [grep], got %q", got)
	}
	if !strings.Contains(got, `args=["-R","-n","func main","."]`) {
		t.Errorf("summary should show argv-style args, got %q", got)
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

func TestToolSummaryFinishesAssistantLine(t *testing.T) {
	var out, errw bytes.Buffer
	r := NewRenderer(&out, &errw, RenderOptions{})

	r.TextDelta("calling a tool")
	r.ToolStart(llm.ToolCall{ID: "c1", Name: "list_dir", Input: json.RawMessage(`{"path":"."}`)})
	r.ToolResult(llm.ToolResult{ForID: "c1", Text: "a\nb\n"})

	if got := out.String(); got != "calling a tool\n" {
		t.Errorf("tool summary should force a newline after assistant text, got %q", got)
	}
	if got := errw.String(); !strings.Contains(got, "[list_dir]") {
		t.Errorf("tool summary should still go to errw, got %q", got)
	}
}

func TestToolSummaryDoesNotDoubleSpaceAfterAssistantNewline(t *testing.T) {
	var out, errw bytes.Buffer
	r := NewRenderer(&out, &errw, RenderOptions{})

	r.TextDelta("calling a tool\n")
	r.ToolStart(llm.ToolCall{ID: "c1", Name: "list_dir", Input: json.RawMessage(`{"path":"."}`)})
	r.ToolResult(llm.ToolResult{ForID: "c1", Text: "a\nb\n"})

	if got := out.String(); got != "calling a tool\n" {
		t.Errorf("tool summary should not add a second newline, got %q", got)
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
		t.Errorf("TurnComplete should not write a newline before usage with no assistant text, got out=%q", out.String())
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

func TestTurnCompleteWritesTrailingNewline(t *testing.T) {
	var out, errw bytes.Buffer
	r := NewRenderer(&out, &errw, RenderOptions{})
	r.StartTurn()
	r.TextDelta("hello world")
	r.TurnComplete(agent.TurnUsage{Steps: 1})

	got := out.String()
	if !strings.HasSuffix(got, "\n") {
		t.Errorf("TurnComplete should write a trailing newline to out, got %q", got)
	}
	// The trailing newline should appear after the text.
	if !strings.Contains(got, "hello world\n") {
		t.Errorf("trailing newline must come after assistant text, got %q", got)
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

func TestModelStepStartGoesToStderr(t *testing.T) {
	var out, errw bytes.Buffer
	r := NewRenderer(&out, &errw, RenderOptions{})

	r.ModelStepStart(2, 1, agent.ContextEstimate{})
	r.ModelStepStart(2, 3, agent.ContextEstimate{})

	if out.Len() != 0 {
		t.Errorf("model progress must not touch stdout, got %q", out.String())
	}
	got := errw.String()
	if !strings.Contains(got, "[model: step 2 waiting]") {
		t.Errorf("missing step wait line, got %q", got)
	}
	if !strings.Contains(got, "[model: step 2 retry 2 waiting]") {
		t.Errorf("missing retry wait line, got %q", got)
	}
}

func TestToolUseStreamEnabledWritesProgressOnlyToStderr(t *testing.T) {
	var out, errw bytes.Buffer
	r := NewRenderer(&out, &errw, RenderOptions{ToolStream: true})

	r.ToolUseStart(llm.ToolCall{ID: "call_1", Name: "read_file"})
	r.ToolUseDelta(0, `{"path":`)
	r.ToolUseDelta(0, `"a.go"}`)
	r.Notice("[done]")

	if out.Len() != 0 {
		t.Errorf("tool-call stream must not touch stdout, got %q", out.String())
	}
	got := errw.String()
	if !strings.Contains(got, "[tool-call: read_file id=call_1]") {
		t.Errorf("missing tool-call start line, got %q", got)
	}
	if strings.Contains(got, "[tool-call args]") || strings.Contains(got, `{"path"`) {
		t.Errorf("tool-call args should not dump raw JSON, got %q", got)
	}
	if !strings.Contains(got, "[done]") {
		t.Errorf("notice should still render after ignored argument deltas, got %q", got)
	}
}

func TestEditToolCallDoesNotDumpLargeJSONArgs(t *testing.T) {
	var out, errw bytes.Buffer
	r := NewRenderer(&out, &errw, RenderOptions{ToolStream: true})

	input := json.RawMessage(`{"path":"internal/ui/repl.go","old_string":"line1\nline2\nline3","new_string":"line1\nline two changed\nline3"}`)
	r.ToolUseStart(llm.ToolCall{ID: "call_edit", Name: "edit"})
	r.ToolUseDelta(0, `{"path":"internal/ui/repl.go","old_string":"line1\nline2\nline3",`)
	r.ToolUseDelta(0, `"new_string":"line1\nline two changed\nline3"}`)
	r.ToolStart(llm.ToolCall{ID: "call_edit", Name: "edit", Input: input})
	r.ToolResult(llm.ToolResult{
		ForID:   "call_edit",
		Text:    "error: old_string not found in internal/ui/repl.go",
		IsError: true,
	})

	got := errw.String()
	if out.Len() != 0 {
		t.Errorf("tool-call stream must not touch stdout, got %q", out.String())
	}
	if strings.Contains(got, "[tool-call args]") || strings.Contains(got, `{"path":"internal/ui/repl.go"`) {
		t.Errorf("large edit args should not be dumped as raw JSON, got %q", got)
	}
	for _, want := range []string{
		"[tool-call: edit id=call_edit]",
		"[tool: edit started",
		"path=internal/ui/repl.go",
		`old_string="line1\nline2\nline3"`,
		`new_string="line1\nline two changed\nline3"`,
		"[edit]",
		"error: error: old_string not found in internal/ui/repl.go",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("stderr missing %q:\n%s", want, got)
		}
	}
}

func TestToolUseStreamDisabledSuppressesRawArgs(t *testing.T) {
	var out, errw bytes.Buffer
	r := NewRenderer(&out, &errw, RenderOptions{ToolStream: false})

	r.ToolUseStart(llm.ToolCall{ID: "call_1", Name: "read_file"})
	r.ToolUseDelta(0, `{"path":"a.go"}`)

	if out.Len() != 0 {
		t.Errorf("disabled tool stream must not touch stdout, got %q", out.String())
	}
	if errw.Len() != 0 {
		t.Errorf("disabled tool stream must not touch stderr, got %q", errw.String())
	}
}
