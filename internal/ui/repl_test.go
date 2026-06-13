package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"harness/internal/agent"
	"harness/internal/llm"
	"harness/internal/llm/llmtest"
	"harness/internal/session"
	"harness/internal/tools"
)

func textDelta(s string) llm.StreamEvent {
	return llm.StreamEvent{Kind: llm.EventTextDelta, Text: s}
}

func newTestApp(t *testing.T, out, errw *bytes.Buffer, fp *llmtest.FakeProvider) *App {
	t.Helper()
	stateDir := t.TempDir()
	a := agent.New(fp, tools.Default(), agent.Options{Model: "claude-opus-4-8"})
	a.SetSystem("you are a test")
	a.SetSleep(func(time.Duration) {}) // no real time in tests
	r := NewRenderer(out, errw, RenderOptions{Model: "claude-opus-4-8", ToolStream: true})
	return &App{
		Agent:       a,
		Renderer:    r,
		Out:         out,
		Errw:        errw,
		Provider:    "anthropic",
		Model:       "claude-opus-4-8",
		BaseURL:     "https://api.anthropic.com/v1",
		System:      "you are a test",
		SessionPath: filepath.Join(stateDir, "session"),
		StateDir:    stateDir,
	}
}

func TestREPLHelpPromptExit(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("the answer")},
		Stop:   llm.StopEndTurn,
		Usage:  llm.Usage{InputTokens: 10, OutputTokens: 3},
	})
	app := newTestApp(t, &out, &errw, fp)

	in := strings.NewReader("/help\nwhat is 2+2?\n/exit\n")
	code := Run(in, app, nil)

	if code != 0 {
		t.Errorf("exit code = %d, want 0", code)
	}
	if !strings.Contains(errw.String(), "/help") || !strings.Contains(errw.String(), "/exit") {
		t.Errorf("/help should list commands, errw=%q", errw.String())
	}
	if len(fp.Requests) != 1 {
		t.Errorf("agent should be invoked once for the single prompt, got %d requests", len(fp.Requests))
	}
	if !strings.Contains(out.String(), "the answer") {
		t.Errorf("assistant text should reach stdout, out=%q", out.String())
	}
}

func TestREPLSavesSessionAfterTurn(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("hi")},
		Stop:   llm.StopEndTurn,
		Usage:  llm.Usage{InputTokens: 5, OutputTokens: 1},
	})
	app := newTestApp(t, &out, &errw, fp)
	path := app.SessionPath

	in := strings.NewReader("hello\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("session should be saved to %s: %v", path, err)
	}
	data, _ := os.ReadFile(filepath.Join(path, "state.json"))
	if !strings.Contains(string(data), "hello") {
		t.Errorf("saved session should contain the user prompt, got %s", data)
	}
}

func TestREPLClearResetsAndRotates(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake",
		llmtest.Step{Events: []llm.StreamEvent{textDelta("one")}, Stop: llm.StopEndTurn},
		llmtest.Step{Events: []llm.StreamEvent{textDelta("two")}, Stop: llm.StopEndTurn},
	)
	app := newTestApp(t, &out, &errw, fp)
	origPath := app.SessionPath

	in := strings.NewReader("first prompt\n/clear\nsecond prompt\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}

	// After /clear the transcript holds only the second turn (user+assistant).
	msgs := app.Agent.Transcript()
	if err := llm.ValidateTranscript(msgs); err != nil {
		t.Fatalf("transcript invalid after clear: %v", err)
	}
	if len(msgs) != 2 {
		t.Fatalf("after /clear transcript should hold only the second turn, got %d messages", len(msgs))
	}
	if msgs[0].Content[0].Text != "second prompt" {
		t.Errorf("transcript should start at the post-clear prompt, got %q", msgs[0].Content[0].Text)
	}

	// /clear rotates to a fresh session path.
	if app.SessionPath == origPath {
		t.Errorf("/clear should rotate to a fresh session file, still %s", origPath)
	}
	if !strings.Contains(errw.String(), "/clear") && !strings.Contains(errw.String(), "cleared") {
		t.Errorf("/clear should acknowledge, errw=%q", errw.String())
	}
}

func TestREPLUnknownCommand(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake")
	app := newTestApp(t, &out, &errw, fp)

	in := strings.NewReader("/bogus\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if !strings.Contains(errw.String(), "/bogus") || !strings.Contains(strings.ToLower(errw.String()), "unknown") {
		t.Errorf("unknown command should be reported, errw=%q", errw.String())
	}
	if len(fp.Requests) != 0 {
		t.Errorf("unknown command must not invoke the agent, got %d requests", len(fp.Requests))
	}
}

func TestREPLLiteralSlashEscape(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("ok")},
		Stop:   llm.StopEndTurn,
	})
	app := newTestApp(t, &out, &errw, fp)

	in := strings.NewReader("//not-a-command\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if len(fp.Requests) != 1 {
		t.Fatalf("// escape should send a prompt, got %d requests", len(fp.Requests))
	}
	// The leading slash is restored; the doubled slash is the escape.
	sent := app.Agent.Transcript()[0].Content[0].Text
	if sent != "/not-a-command" {
		t.Errorf("escaped prompt = %q, want %q", sent, "/not-a-command")
	}
}

func TestREPLBracketedPasteSubmittedAsSingleLiteralPrompt(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("ok")},
		Stop:   llm.StopEndTurn,
	})
	app := newTestApp(t, &out, &errw, fp)

	pasted := "/exit is pasted text\nsecond line\nthird line"
	in := strings.NewReader(bracketedPasteStart + pasted + bracketedPasteEnd + "\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if len(fp.Requests) != 1 {
		t.Fatalf("bracketed paste should send one prompt, got %d requests", len(fp.Requests))
	}
	sent := app.Agent.Transcript()[0].Content[0].Text
	if sent != pasted {
		t.Errorf("pasted prompt = %q, want %q", sent, pasted)
	}
}

func TestREPLAcceptsPromptLongerThanScannerLimit(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("ok")},
		Stop:   llm.StopEndTurn,
	})
	app := newTestApp(t, &out, &errw, fp)

	prompt := strings.Repeat("x", 4*1024*1024+1)
	in := strings.NewReader(prompt + "\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if len(fp.Requests) != 1 {
		t.Fatalf("long prompt should send one request, got %d", len(fp.Requests))
	}
	sent := app.Agent.Transcript()[0].Content[0].Text
	if sent != prompt {
		t.Fatalf("long prompt length = %d, want %d", len(sent), len(prompt))
	}
}

func TestREPLUsageCumulative(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake",
		llmtest.Step{Events: []llm.StreamEvent{textDelta("a")}, Stop: llm.StopEndTurn, Usage: llm.Usage{InputTokens: 100, OutputTokens: 10}},
		llmtest.Step{Events: []llm.StreamEvent{textDelta("b")}, Stop: llm.StopEndTurn, Usage: llm.Usage{InputTokens: 200, OutputTokens: 20}},
	)
	app := newTestApp(t, &out, &errw, fp)

	in := strings.NewReader("p1\np2\n/usage\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	got := errw.String()
	// Cumulative: 300 in / 30 out across both turns.
	if !strings.Contains(got, "300") || !strings.Contains(got, "30 out") {
		t.Errorf("/usage should show cumulative tokens, errw=%q", got)
	}
}

func TestREPLToolsCommandListsBuiltInMCPAndDisabledTools(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake")
	app := newTestApp(t, &out, &errw, fp)
	reg := tools.Default()
	reg.Register(mcpRefreshTool{name: "mcp__search__lookup"})
	reg.Register(mcpRefreshTool{name: "mcp__files__read"})
	app.Agent.SetTools(reg)
	app.DisabledTools = []tools.DisabledTool{
		{Name: "rg", Reason: `"rg" binary not found`},
	}

	in := strings.NewReader("/tools\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if len(fp.Requests) != 0 {
		t.Fatalf("/tools should not invoke the agent, got %d requests", len(fp.Requests))
	}
	got := errw.String()
	for _, want := range []string{
		"built-in tools:",
		"  read_file - Read a file from disk.",
		"mcp tools:",
		"  [files]",
		"    mcp__files__read - refreshed tool",
		"  [search]",
		"    mcp__search__lookup - refreshed tool",
		"disabled tools:",
		`  rg  ("rg" binary not found)`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("/tools output missing %q:\n%s", want, got)
		}
	}
}

func TestREPLUsageLineSeedsFromSavedUsage(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake",
		llmtest.Step{Events: []llm.StreamEvent{textDelta("a")}, Stop: llm.StopEndTurn, Usage: llm.Usage{InputTokens: 50, OutputTokens: 5}},
	)
	app := newTestApp(t, &out, &errw, fp)
	app.SetUsage(session.UsageTotals{Usage: llm.Usage{InputTokens: 300, OutputTokens: 30}})

	in := strings.NewReader("p1\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	got := errw.String()
	if !strings.Contains(got, "50 (350) in") {
		t.Errorf("usage line should include seeded input total, errw=%q", got)
	}
	if !strings.Contains(got, "5 (35) out") {
		t.Errorf("usage line should include seeded output total, errw=%q", got)
	}
}

func TestREPLClearResetsUsageLineCumulative(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake",
		llmtest.Step{Events: []llm.StreamEvent{textDelta("a")}, Stop: llm.StopEndTurn, Usage: llm.Usage{InputTokens: 100, OutputTokens: 10}},
		llmtest.Step{Events: []llm.StreamEvent{textDelta("b")}, Stop: llm.StopEndTurn, Usage: llm.Usage{InputTokens: 200, OutputTokens: 20}},
	)
	app := newTestApp(t, &out, &errw, fp)

	in := strings.NewReader("p1\n/clear\np2\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	got := errw.String()
	if !strings.Contains(got, "100 (100) in") {
		t.Errorf("first turn should show its own cumulative input, errw=%q", got)
	}
	if !strings.Contains(got, "200 (200) in") {
		t.Errorf("post-clear turn should reset cumulative input, errw=%q", got)
	}
	if strings.Contains(got, "200 (300) in") {
		t.Errorf("post-clear turn leaked pre-clear input total, errw=%q", got)
	}
}

func TestREPLUsageLineIncludesCompactUsage(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake",
		llmtest.Step{Events: []llm.StreamEvent{textDelta("CANNED SUMMARY")}, Stop: llm.StopEndTurn, Usage: llm.Usage{InputTokens: 9100, OutputTokens: 400}},
		llmtest.Step{Events: []llm.StreamEvent{textDelta("after compact")}, Stop: llm.StopEndTurn, Usage: llm.Usage{InputTokens: 100, OutputTokens: 10}},
	)
	app := newTestApp(t, &out, &errw, fp)

	var seed []llm.Message
	for i := 0; i < 10; i++ {
		label := string(rune('a' + i))
		seed = append(seed,
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: label + " q"}}},
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: label + " a"}}},
		)
	}
	app.Agent.SetTranscript(seed)

	in := strings.NewReader("/compact\np1\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	got := errw.String()
	if !strings.Contains(got, "100 (9.2k) in") {
		t.Errorf("post-compact turn should include compact input usage in cumulative total, errw=%q", got)
	}
	if !strings.Contains(got, "10 (410) out") {
		t.Errorf("post-compact turn should include compact output usage in cumulative total, errw=%q", got)
	}
}

func TestREPLCompactCommand(t *testing.T) {
	var out, errw bytes.Buffer
	// The only model call here is the summary call /compact triggers.
	fp := llmtest.New("fake",
		llmtest.Step{Events: []llm.StreamEvent{textDelta("CANNED SUMMARY")}, Stop: llm.StopEndTurn, Usage: llm.Usage{InputTokens: 9100, OutputTokens: 400}},
	)
	app := newTestApp(t, &out, &errw, fp)

	// Seed enough whole turns that there is something older than the last four
	// to summarize.
	var seed []llm.Message
	for i := 0; i < 10; i++ {
		label := string(rune('a' + i))
		seed = append(seed,
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: label + " q"}}},
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: label + " a"}}},
		)
	}
	app.Agent.SetTranscript(seed)

	in := strings.NewReader("/compact\n/usage\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}

	msgs := app.Agent.Transcript()
	if err := llm.ValidateTranscript(msgs); err != nil {
		t.Fatalf("transcript invalid after /compact: %v", err)
	}
	if len(msgs) != 1+8 {
		t.Fatalf("/compact should collapse to summary + last 4 turns (9 msgs), got %d", len(msgs))
	}
	got := errw.String()
	if !strings.Contains(got, "compacted") {
		t.Errorf("/compact should print a compaction report, errw=%q", got)
	}
	// The summary call's tokens must fold into the cumulative session totals.
	if !strings.Contains(got, "9100") || !strings.Contains(got, "400 out") {
		t.Errorf("/usage should include the summary call usage after /compact, errw=%q", got)
	}
	// The summary call was actually issued (the only model call here).
	if len(fp.Requests) != 1 {
		t.Errorf("/compact should issue exactly the summary call, got %d requests", len(fp.Requests))
	}
}

func TestREPLModelCommand(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake")
	app := newTestApp(t, &out, &errw, fp)
	app.AvailableModels = []string{"gpt-5.5", "claude-opus-4-8"}

	in := strings.NewReader("/model\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	got := errw.String()
	if !strings.Contains(got, "anthropic") || !strings.Contains(got, "claude-opus-4-8") || !strings.Contains(got, "api.anthropic.com") {
		t.Errorf("/model should print provider/model/base-url, errw=%q", got)
	}
	if !strings.Contains(got, "available models:") || !strings.Contains(got, "gpt-5.5") {
		t.Errorf("/model should list available models, errw=%q", got)
	}
}

func TestREPLModelCommandSwitchesNextTurn(t *testing.T) {
	var out, errw bytes.Buffer
	initial := llmtest.New("initial")
	switched := llmtest.New("switched", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("switched reply")},
		Stop:   llm.StopEndTurn,
	})
	app := newTestApp(t, &out, &errw, initial)
	app.SwitchModel = func(model string) (ModelSelection, error) {
		if model != "gpt-5.5" {
			t.Fatalf("switch model = %q, want gpt-5.5", model)
		}
		return ModelSelection{
			Provider: "openai",
			Model:    model,
			BaseURL:  "https://api.openai.com/v1",
			Runtime:  switched,
		}, nil
	}

	in := strings.NewReader("/model gpt-5.5\nhello\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if len(initial.Requests) != 0 {
		t.Fatalf("initial provider should not receive the post-switch turn, got %d requests", len(initial.Requests))
	}
	if len(switched.Requests) != 1 {
		t.Fatalf("switched provider requests = %d, want 1", len(switched.Requests))
	}
	if switched.Requests[0].Model != "gpt-5.5" {
		t.Fatalf("post-switch request model = %q, want gpt-5.5", switched.Requests[0].Model)
	}
	if app.Provider != "openai" || app.Model != "gpt-5.5" {
		t.Fatalf("app provider/model = %s/%s, want openai/gpt-5.5", app.Provider, app.Model)
	}
	if !strings.Contains(errw.String(), "model switched") {
		t.Errorf("switch should be acknowledged, errw=%q", errw.String())
	}
}

func TestREPLModeCommandLists(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake")
	app := newTestApp(t, &out, &errw, fp)
	app.Mode = "plan"
	app.AvailableModes = []string{"auto", "independent", "plan"}

	in := strings.NewReader("/mode\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	got := errw.String()
	for _, name := range []string{"auto", "independent", "plan"} {
		if !strings.Contains(got, name) {
			t.Errorf("/mode should list %q, errw=%q", name, got)
		}
	}
	if !strings.Contains(got, "plan (current)") {
		t.Errorf("/mode should mark the current mode, errw=%q", got)
	}
}

func TestREPLModeCommandSwitchesNextTurn(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("ok")},
		Stop:   llm.StopEndTurn,
	})
	app := newTestApp(t, &out, &errw, fp)
	planTools, err := tools.Catalog().Subset([]string{"read_file", "grep"})
	if err != nil {
		t.Fatalf("subset: %v", err)
	}
	app.SwitchMode = func(name string) (ModeSelection, error) {
		if name != "plan" {
			t.Fatalf("switch mode = %q, want plan", name)
		}
		return ModeSelection{Name: "plan", Tools: planTools, System: "PLAN MODE PROMPT"}, nil
	}

	in := strings.NewReader("/mode plan\nhello\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if app.Mode != "plan" {
		t.Errorf("app.Mode = %q, want plan", app.Mode)
	}
	if app.System != "PLAN MODE PROMPT" {
		t.Errorf("app.System should update so saves capture it, got %q", app.System)
	}
	if !strings.Contains(errw.String(), "mode switched: plan") {
		t.Errorf("switch should be acknowledged, errw=%q", errw.String())
	}
	// The post-switch turn must advertise only the plan tool set.
	if len(fp.Requests) != 1 {
		t.Fatalf("provider requests = %d, want 1", len(fp.Requests))
	}
	names := make([]string, len(fp.Requests[0].Tools))
	for i, s := range fp.Requests[0].Tools {
		names[i] = s.Name
	}
	if len(names) != 2 || names[0] != "read_file" || names[1] != "grep" {
		t.Errorf("post-switch request should advertise [read_file grep], got %v", names)
	}
}

func TestREPLModeCommandUnknownReportsError(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake")
	app := newTestApp(t, &out, &errw, fp)
	app.Mode = "auto"
	app.SwitchMode = func(name string) (ModeSelection, error) {
		return ModeSelection{}, errors.New(`unknown mode "bogus" (available: auto, plan)`)
	}

	in := strings.NewReader("/mode bogus\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if !strings.Contains(errw.String(), "mode switch failed") {
		t.Errorf("unknown mode should report failure, errw=%q", errw.String())
	}
	if app.Mode != "auto" {
		t.Errorf("failed switch should not change the mode, got %q", app.Mode)
	}
}

func TestREPLSaveToPath(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("hi")},
		Stop:   llm.StopEndTurn,
	})
	app := newTestApp(t, &out, &errw, fp)
	alt := filepath.Join(t.TempDir(), "alt.json")

	in := strings.NewReader("hello\n/save " + alt + "\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if _, err := os.Stat(alt); err != nil {
		t.Fatalf("/save <file> should write to the given path: %v", err)
	}
}

func TestREPLEOFSavesAndExitsZero(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("hi")},
		Stop:   llm.StopEndTurn,
	})
	app := newTestApp(t, &out, &errw, fp)

	// No trailing /exit: stream ends (EOF) after one prompt.
	in := strings.NewReader("hello\n")
	if code := Run(in, app, nil); code != 0 {
		t.Errorf("^D/EOF should exit 0, got %d", code)
	}
	if _, err := os.Stat(app.SessionPath); err != nil {
		t.Errorf("EOF should save the session: %v", err)
	}
}

func TestREPLProviderErrorReported(t *testing.T) {
	var out, errw bytes.Buffer
	// A plain (non-API, non-cancel) error is retryable, so it must persist
	// across the whole per-step budget (1 + 2 retries) to surface to errw.
	fail := llmtest.Step{Err: errContext("boom")}
	fp := llmtest.New("fake", fail, fail, fail)
	app := newTestApp(t, &out, &errw, fp)

	in := strings.NewReader("hello\n/exit\n")
	// A turn error in the REPL is reported but does not end the session.
	if code := Run(in, app, nil); code != 0 {
		t.Errorf("REPL should survive a turn error and exit 0 via /exit, got %d", code)
	}
	if !strings.Contains(strings.ToLower(errw.String()), "error") {
		t.Errorf("turn error should be reported to errw, got %q", errw.String())
	}
}

func TestREPLEscapeEscapeCancelsActiveTurn(t *testing.T) {
	var out, errw bytes.Buffer
	inTurn := make(chan struct{})
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("partial")},
		Stop:   llm.StopEndTurn,
		Usage:  llm.Usage{InputTokens: 5, OutputTokens: 1},
		Block: func(ctx context.Context) {
			close(inTurn)
			<-ctx.Done()
		},
	})
	app := newTestApp(t, &out, &errw, fp)
	exitRequested := make(chan struct{}, 1)
	app.Interrupt = agent.NewInterruptWatcher(make(chan os.Signal), app.clock(), func() {
		exitRequested <- struct{}{}
	})

	pr, pw := io.Pipe()
	defer pr.Close()
	defer pw.Close()
	codeCh := make(chan int, 1)
	go func() { codeCh <- Run(pr, app, nil) }()

	writePipe(t, pw, "first\n")
	select {
	case <-inTurn:
	case <-time.After(time.Second):
		t.Fatal("turn did not start")
	}
	writePipe(t, pw, "\x1b\x1b/exit\n")
	_ = pw.Close()

	code := waitRun(t, codeCh)
	if code != ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if !strings.Contains(errw.String(), "[cancelled]") {
		t.Fatalf("Esc-Esc should render cancellation, errw=%q", errw.String())
	}
	select {
	case <-exitRequested:
		t.Fatal("Esc-Esc must cancel the turn without requesting process exit")
	default:
	}
}

func TestREPLTypeaheadDuringActiveTurnRunsAfterTurn(t *testing.T) {
	var out, errw bytes.Buffer
	inTurn := make(chan struct{})
	releaseTurn := make(chan struct{})
	fp := llmtest.New("fake",
		llmtest.Step{
			Events: []llm.StreamEvent{textDelta("first answer")},
			Stop:   llm.StopEndTurn,
			Usage:  llm.Usage{InputTokens: 5, OutputTokens: 2},
			Block: func(ctx context.Context) {
				close(inTurn)
				<-releaseTurn
			},
		},
		llmtest.Step{
			Events: []llm.StreamEvent{textDelta("second answer")},
			Stop:   llm.StopEndTurn,
			Usage:  llm.Usage{InputTokens: 6, OutputTokens: 2},
		},
	)
	app := newTestApp(t, &out, &errw, fp)

	pr, pw := io.Pipe()
	defer pr.Close()
	defer pw.Close()
	codeCh := make(chan int, 1)
	go func() { codeCh <- Run(pr, app, nil) }()

	writePipe(t, pw, "first\n")
	select {
	case <-inTurn:
	case <-time.After(time.Second):
		t.Fatal("turn did not start")
	}
	writePipe(t, pw, "second\n/exit\n")
	_ = pw.Close()
	close(releaseTurn)

	code := waitRun(t, codeCh)
	if code != ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if len(fp.Requests) != 2 {
		t.Fatalf("typeahead prompt should run after the blocked turn, got %d requests", len(fp.Requests))
	}
	var prompts []string
	for _, msg := range app.Agent.Transcript() {
		if msg.Role == llm.RoleUser && len(msg.Content) == 1 && msg.Content[0].Kind == llm.BlockText {
			prompts = append(prompts, msg.Content[0].Text)
		}
	}
	if strings.Join(prompts, "|") != "first|second" {
		t.Fatalf("user prompts = %q, want first|second", strings.Join(prompts, "|"))
	}
}

// TestREPLInputReadErrorWarned covers the lint fix: a non-EOF read error from
// stdin must be surfaced (warned to errw) rather than silently treated as a clean
// end of input. The scanner stops on the error; Run reports it and exits 0
// (there is nothing more to read, but the user should know why).
func TestREPLInputReadErrorWarned(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("hi")},
		Stop:   llm.StopEndTurn,
	})
	app := newTestApp(t, &out, &errw, fp)

	in := &erroringReader{data: []byte("hello\n"), err: errContext("disk gone")}
	code := Run(in, app, nil)
	if code != ExitOK {
		t.Fatalf("read error should still exit 0, got %d; errw=%q", code, errw.String())
	}
	got := errw.String()
	if !strings.Contains(strings.ToLower(got), "input") || !strings.Contains(got, "disk gone") {
		t.Errorf("input read error should be warned to errw, got %q", got)
	}
	// The session is still saved on this exit path.
	if _, err := os.Stat(app.SessionPath); err != nil {
		t.Errorf("read-error exit should save the session: %v", err)
	}
}

// unsavablePath returns a SessionPath whose parent is a regular file, so
// session.Save's os.MkdirAll fails — a deterministic stand-in for the ordinary
// disk-full / read-only / permission faults that make an automatic save fail.
func unsavablePath(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	blocker := filepath.Join(dir, "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatalf("write blocker: %v", err)
	}
	// blocker is a file, so MkdirAll(blocker/sub) cannot create the parent.
	return filepath.Join(blocker, "sub", "session")
}

// TestREPLAutoSaveFailureWarned is the regression test for after-every-turn
// auto-save errors being silently swallowed (design §11/§12: a visible failure
// beats silent data loss). A failed save must warn to errw, not vanish.
func TestREPLAutoSaveFailureWarned(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("hi")},
		Stop:   llm.StopEndTurn,
	})
	app := newTestApp(t, &out, &errw, fp)
	app.SessionPath = unsavablePath(t)

	// One prompt then /exit; the after-turn auto-save fails first.
	in := strings.NewReader("hello\n/exit\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("REPL should still exit 0, got %d; errw=%q", code, errw.String())
	}
	if !strings.Contains(errw.String(), "save failed") {
		t.Errorf("failed auto-save must warn to errw, got %q", errw.String())
	}
}

// TestREPLCompactSaveFailureWarned covers the /compact save path, the sixth
// automatic-save site: after a forced compaction the collapsed transcript must
// be saved, and a failed save must warn rather than leave a stale file silently.
func TestREPLCompactSaveFailureWarned(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake",
		llmtest.Step{Events: []llm.StreamEvent{textDelta("CANNED SUMMARY")}, Stop: llm.StopEndTurn, Usage: llm.Usage{InputTokens: 100, OutputTokens: 10}},
	)
	app := newTestApp(t, &out, &errw, fp)
	app.SessionPath = unsavablePath(t)

	var seed []llm.Message
	for i := 0; i < 10; i++ {
		label := string(rune('a' + i))
		seed = append(seed,
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: label + " q"}}},
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: label + " a"}}},
		)
	}
	app.Agent.SetTranscript(seed)

	// /compact compacts and saves; the save fails and must warn. The failure does
	// not abort the REPL.
	in := strings.NewReader("/compact\n")
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("REPL should exit 0 on EOF, got %d; errw=%q", code, errw.String())
	}
	if !strings.Contains(errw.String(), "save failed") {
		t.Errorf("failed /compact save must warn to errw, got %q", errw.String())
	}
}

// TestREPLExitSaveFailureWarned covers the /exit save path: if the final save
// fails, the user must be told the on-disk session is stale rather than exiting
// as if it were saved.
func TestREPLExitSaveFailureWarned(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake")
	app := newTestApp(t, &out, &errw, fp)
	app.SessionPath = unsavablePath(t)

	in := strings.NewReader("/exit\n") // no turn; only the /exit save runs
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("/exit should exit 0, got %d; errw=%q", code, errw.String())
	}
	if !strings.Contains(errw.String(), "save failed") {
		t.Errorf("failed /exit save must warn to errw, got %q", errw.String())
	}
}

// TestREPLEOFSaveFailureWarned covers the EOF (^D) exit-save path.
func TestREPLEOFSaveFailureWarned(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake")
	app := newTestApp(t, &out, &errw, fp)
	app.SessionPath = unsavablePath(t)

	in := strings.NewReader("") // immediate EOF, no prompt
	if code := Run(in, app, nil); code != 0 {
		t.Fatalf("EOF should exit 0, got %d; errw=%q", code, errw.String())
	}
	if !strings.Contains(errw.String(), "save failed") {
		t.Errorf("failed EOF save must warn to errw, got %q", errw.String())
	}
}

// erroringReader returns its data once, then a non-EOF error (not io.EOF), so the
// scanner stops with a real read error rather than clean end-of-input.
type erroringReader struct {
	data []byte
	off  int
	err  error
}

func (r *erroringReader) Read(p []byte) (int, error) {
	if r.off < len(r.data) {
		n := copy(p, r.data[r.off:])
		r.off += n
		return n, nil
	}
	return 0, r.err
}

func writePipe(t *testing.T, w *io.PipeWriter, s string) {
	t.Helper()
	done := make(chan error, 1)
	go func() {
		_, err := w.Write([]byte(s))
		done <- err
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("pipe write %q: %v", s, err)
		}
	case <-time.After(time.Second):
		t.Fatalf("pipe write %q timed out", s)
	}
}

func waitRun(t *testing.T, codeCh <-chan int) int {
	t.Helper()
	select {
	case code := <-codeCh:
		return code
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not return")
	}
	return 0
}

// errContext is a sentinel non-cancellation error for provider-error tests.
type errContextT string

func (e errContextT) Error() string { return string(e) }
func errContext(s string) error     { return errContextT(s) }

// The terminal reset must go to /dev/tty (and only when one exists), never to
// Errw: a piped or redirected stderr must receive no escape sequences. This
// regression-tests the removal of the \033c (RIS) write before the first
// prompt, which also cleared the user's screen and scrollback.
func TestREPLWritesNoEscapeSequencesToErrw(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake")
	app := newTestApp(t, &out, &errw, fp)

	code := Run(strings.NewReader("/exit\n"), app, nil)

	if code != 0 {
		t.Errorf("exit code = %d, want 0", code)
	}
	if s := errw.String(); strings.ContainsRune(s, '\x1b') {
		t.Errorf("errw contains escape bytes: %q", s)
	}
}

// mcpRefreshTool is a minimal Tool used to prove the RefreshMCP hook's returned
// registry was applied to the agent before the turn.
type mcpRefreshTool struct{ name string }

func (m mcpRefreshTool) Name() string            { return m.name }
func (m mcpRefreshTool) Description() string     { return "refreshed tool" }
func (m mcpRefreshTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (m mcpRefreshTool) ReadOnly() bool          { return false }
func (m mcpRefreshTool) Run(context.Context, json.RawMessage) (string, error) {
	return "ok", nil
}

// TestREPLRefreshMCPAppliedBeforeTurn asserts the REPL consults RefreshMCP at
// the idle-prompt boundary, swaps in the returned tools (visible in the next
// request's advertised tool list), and renders the notice.
func TestREPLRefreshMCPAppliedBeforeTurn(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("done")},
		Stop:   llm.StopEndTurn,
	})
	app := newTestApp(t, &out, &errw, fp)
	app.Mode = "auto"

	refreshed := &tools.Registry{}
	refreshed.Register(mcpRefreshTool{name: "mcp__test__fresh"})

	var gotMode string
	calls := 0
	app.RefreshMCP = func(mode string) (*tools.Registry, string) {
		calls++
		gotMode = mode
		return refreshed, "[mcp: tool list updated; 1 tools]"
	}

	if code := Run(strings.NewReader("hello\n/exit\n"), app, nil); code != 0 {
		t.Fatalf("exit = %d, want 0; errw=%q", code, errw.String())
	}
	if calls != 1 {
		t.Errorf("RefreshMCP called %d times, want 1", calls)
	}
	if gotMode != "auto" {
		t.Errorf("RefreshMCP mode = %q, want auto", gotMode)
	}
	if len(fp.Requests) != 1 {
		t.Fatalf("want 1 request, got %d", len(fp.Requests))
	}
	var advertised bool
	for _, ts := range fp.Requests[0].Tools {
		if ts.Name == "mcp__test__fresh" {
			advertised = true
		}
	}
	if !advertised {
		t.Errorf("refreshed tool not advertised to the model: %+v", fp.Requests[0].Tools)
	}
	if !strings.Contains(errw.String(), "tool list updated") {
		t.Errorf("refresh notice not rendered: %q", errw.String())
	}
}

// TestREPLRefreshMCPNoChangeKeepsTools confirms a nil-registry hook result is a
// no-op: the turn still runs and no notice is rendered.
func TestREPLRefreshMCPNoChangeKeepsTools(t *testing.T) {
	var out, errw bytes.Buffer
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{textDelta("done")},
		Stop:   llm.StopEndTurn,
	})
	app := newTestApp(t, &out, &errw, fp)
	called := false
	app.RefreshMCP = func(string) (*tools.Registry, string) {
		called = true
		return nil, ""
	}
	if code := Run(strings.NewReader("hi\n/exit\n"), app, nil); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if !called {
		t.Errorf("RefreshMCP should still be consulted")
	}
	if strings.Contains(errw.String(), "tool list updated") {
		t.Errorf("no notice expected on no-change, got %q", errw.String())
	}
}
