package ui

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"harness/internal/llm"
	"harness/internal/llm/llmtest"
)

func TestOneShotAssistantTextOnStdoutNoiseOnStderr(t *testing.T) {
	var out, errw bytes.Buffer
	tool := toolStep("read_file", `{"path":"a.go"}`, "c1")
	fp := llmtest.New("fake",
		llmtest.Step{
			Events: []llm.StreamEvent{textDelta("reading file "), tool},
			Stop:   llm.StopToolUse,
			Usage:  llm.Usage{InputTokens: 10, OutputTokens: 4},
		},
		llmtest.Step{
			Events: []llm.StreamEvent{textDelta("the answer is 42")},
			Stop:   llm.StopEndTurn,
			Usage:  llm.Usage{InputTokens: 20, OutputTokens: 6},
		},
	)
	app := newTestApp(t, &out, &errw, fp)

	code := OneShot(app, "do it")
	if code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if !strings.Contains(out.String(), "the answer is 42") {
		t.Errorf("assistant text should be on stdout, out=%q", out.String())
	}
	if strings.Contains(out.String(), "[read_file]") || strings.Contains(out.String(), "[turn:") {
		t.Errorf("tool summaries and usage must not pollute stdout, out=%q", out.String())
	}
	if !strings.Contains(errw.String(), "[read_file]") {
		t.Errorf("tool summary should be on stderr, errw=%q", errw.String())
	}
	if !strings.Contains(errw.String(), "[turn:") {
		t.Errorf("usage line should be on stderr, errw=%q", errw.String())
	}
}

func TestOneShotSavesSessionAndRunsOneTurn(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("done")},
		Stop:   llm.StopEndTurn,
	})
	app := newTestApp(t, &out, &errw, fp)

	if code := OneShot(app, "go"); code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if len(fp.Requests) != 1 {
		t.Errorf("one-shot should run exactly one turn, got %d requests", len(fp.Requests))
	}
	if _, err := os.Stat(app.SessionPath); err != nil {
		t.Errorf("one-shot should save the session: %v", err)
	}
}

// TestOneShotSaveFailureWarned is the regression test for the one-shot save
// error being silently swallowed: OneShot used to return ExitOK and print
// nothing when the session save failed, losing the transcript with no signal.
// A failed save must warn to errw (design §11/§12 — visible failure beats silent
// data loss).
func TestOneShotSaveFailureWarned(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("done")},
		Stop:   llm.StopEndTurn,
	})
	app := newTestApp(t, &out, &errw, fp)
	app.SessionPath = unsavablePath(t)

	// The turn itself succeeds; only the save fails. Exit code is unchanged (the
	// turn ran), but the failure must be surfaced.
	if code := OneShot(app, "go"); code != ExitOK {
		t.Fatalf("turn succeeded, exit code should be 0, got %d; errw=%q", code, errw.String())
	}
	if !strings.Contains(errw.String(), "save failed") {
		t.Errorf("failed one-shot save must warn to errw, got %q", errw.String())
	}
}

func TestOneShotProviderErrorExit1(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake", llmtest.Step{Err: errContext("upstream 500")})
	app := newTestApp(t, &out, &errw, fp)

	code := OneShot(app, "go")
	if code != 1 {
		t.Errorf("provider error should exit 1, got %d", code)
	}
	if !strings.Contains(strings.ToLower(errw.String()), "error") {
		t.Errorf("error should be reported to stderr, errw=%q", errw.String())
	}
}

func TestBuildPromptDash(t *testing.T) {
	got, err := BuildPrompt("-", strings.NewReader("from stdin"), true)
	if err != nil {
		t.Fatalf("BuildPrompt: %v", err)
	}
	if got != "from stdin" {
		t.Errorf("`-p -` should read the whole prompt from stdin, got %q", got)
	}
}

func TestBuildPromptFlagAndStdinConcatenate(t *testing.T) {
	got, err := BuildPrompt("summarize:", strings.NewReader("the notes"), true)
	if err != nil {
		t.Fatalf("BuildPrompt: %v", err)
	}
	if got != "summarize:\nthe notes" {
		t.Errorf("flag text then stdin should concatenate, got %q", got)
	}
}

func TestBuildPromptFlagOnlyWhenNoStdin(t *testing.T) {
	got, err := BuildPrompt("just the flag", nil, false)
	if err != nil {
		t.Fatalf("BuildPrompt: %v", err)
	}
	if got != "just the flag" {
		t.Errorf("flag-only prompt should pass through, got %q", got)
	}
}

// toolStep builds a complete tool-call Done event for one-shot tests.
func toolStep(name, input, id string) llm.StreamEvent {
	return llm.StreamEvent{
		Kind:      llm.EventToolCallDone,
		ToolID:    id,
		ToolName:  name,
		ToolInput: []byte(input),
	}
}
