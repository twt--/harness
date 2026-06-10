package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"harness/internal/llm"
	"harness/internal/llm/factory"
	"harness/internal/llm/llmtest"
	"harness/internal/session"
	"harness/internal/ui"
)

// fakeProviderEnv builds an environment whose provider is the scripted fake, so
// run is exercised without real network calls. stateDir/HOME are pinned to a
// temp dir so auto-save paths are deterministic.
func fakeProviderEnv(t *testing.T, args []string, fp *llmtest.FakeProvider, stdin string) (environment, *bytes.Buffer, *bytes.Buffer, func(string) string) {
	t.Helper()
	dir := t.TempDir()
	getenv := func(k string) string {
		switch k {
		case "HOME":
			return dir
		case "XDG_STATE_HOME":
			return filepath.Join(dir, "state")
		default:
			return ""
		}
	}
	var out, errw bytes.Buffer
	env := environment{
		args:       args,
		stdin:      strings.NewReader(stdin),
		stdout:     &out,
		stderr:     &errw,
		getenv:     getenv,
		now:        func() time.Time { return time.Date(2026, 6, 9, 12, 0, 0, 0, time.UTC) },
		colorTTY:   false,
		stdinPiped: false,
		sigCh:      nil, // no signal handling in tests
		newProvider: func(factory.Options) (llm.Provider, error) {
			return fp, nil
		},
	}
	return env, &out, &errw, getenv
}

func TestRunOneShotAssistantToStdout(t *testing.T) {
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "42"}},
		Stop:   llm.StopEndTurn,
		Usage:  llm.Usage{InputTokens: 5, OutputTokens: 1},
	})
	env, out, errw, _ := fakeProviderEnv(t, []string{"-model", "claude-opus-4-8", "-p", "what is the answer"}, fp, "")

	code := run(env)
	if code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if !strings.Contains(out.String(), "42") {
		t.Errorf("assistant text should be on stdout, out=%q", out.String())
	}
	if !strings.Contains(errw.String(), "session:") {
		t.Errorf("session path should be printed at startup on stderr, errw=%q", errw.String())
	}
	if len(fp.Requests) != 1 {
		t.Errorf("one-shot runs exactly one turn, got %d requests", len(fp.Requests))
	}
	// Wiring gap #1: the resolved model must reach the provider request.
	if fp.Requests[0].Model != "claude-opus-4-8" {
		t.Errorf("request model = %q, want claude-opus-4-8", fp.Requests[0].Model)
	}
}

// TestRunEnvBlockReportsAbsoluteCwd is the regression test for the env block
// emitting `cwd: .` instead of the absolute working directory (design §8.5).
// main must populate EnvOptions.Dir via os.Getwd so the system prompt the model
// receives names a real absolute path it can reason about.
func TestRunEnvBlockReportsAbsoluteCwd(t *testing.T) {
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "ok"}},
		Stop:   llm.StopEndTurn,
		Usage:  llm.Usage{InputTokens: 1, OutputTokens: 1},
	})
	env, _, errw, _ := fakeProviderEnv(t, []string{"-model", "claude-opus-4-8", "-p", "hi"}, fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if len(fp.Requests) != 1 {
		t.Fatalf("want 1 request, got %d", len(fp.Requests))
	}
	system := fp.Requests[0].System

	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if !filepath.IsAbs(wd) {
		t.Fatalf("test precondition: cwd %q is not absolute", wd)
	}
	if strings.Contains(system, "cwd: .\n") {
		t.Errorf("system prompt reports cwd as the literal \".\"; system=%q", system)
	}
	if !strings.Contains(system, "cwd: "+wd+"\n") {
		t.Errorf("system prompt should report the absolute cwd %q; system=%q", wd, system)
	}
}

// TestRunHelpFlagExitsZeroWithUsage covers the design §10 help path: -h/--help is
// a request, not a usage error. It prints a usage screen naming every §10 flag
// and exits 0 (the prior defect exited 2 with a terse "flag: help requested").
func TestRunHelpFlagExitsZeroWithUsage(t *testing.T) {
	flags := []string{
		"-p", "-provider", "-model", "-base-url", "-system", "-system-override",
		"-no-env", "-resume", "-session", "-max-steps", "-context-window",
		"-v", "-no-color", "-config", "-setup",
	}
	for _, arg := range []string{"-h", "--help"} {
		fp := llmtest.New("fake")
		env, out, errw, _ := fakeProviderEnv(t, []string{arg}, fp, "")
		code := run(env)
		if code != ui.ExitOK {
			t.Fatalf("run(%q) exit = %d, want 0; errw=%q", arg, code, errw.String())
		}
		// Usage goes to stdout (it is the requested output, not an error).
		text := out.String()
		for _, f := range flags {
			if !strings.Contains(text, f) {
				t.Errorf("run(%q) usage missing flag %q:\n%s", arg, f, text)
			}
		}
		if len(fp.Requests) != 0 {
			t.Errorf("run(%q) should not call the provider, got %d requests", arg, len(fp.Requests))
		}
	}
}

func TestRunSetupCreatesDefaultConfigAndProviderConfig(t *testing.T) {
	fp := llmtest.New("fake")
	stdin := strings.Join([]string{
		"openrouter",
		"https://openrouter.ai/api/v1",
		"openai",
		"sk-openrouter",
		"openai/gpt-5.5",
	}, "\n") + "\n"
	env, out, errw, getenv := fakeProviderEnv(t, []string{"--setup"}, fp, stdin)

	code := run(env)
	if code != ui.ExitOK {
		t.Fatalf("setup exit = %d, want 0; errw=%q", code, errw.String())
	}
	if len(fp.Requests) != 0 {
		t.Fatalf("setup should not call provider, got %d requests", len(fp.Requests))
	}

	configPath := filepath.Join(getenv("HOME"), ".config", "harness", "config.json")
	providerPath := filepath.Join(getenv("HOME"), ".config", "harness", "openrouter.json")
	if !strings.Contains(out.String(), configPath) || !strings.Contains(out.String(), providerPath) {
		t.Fatalf("setup output should name written files, got %q", out.String())
	}

	var mainCfg struct {
		Provider        string   `json:"provider"`
		Model           string   `json:"model"`
		ProviderConfigs []string `json:"provider_configs"`
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read generated config: %v", err)
	}
	if err := json.Unmarshal(data, &mainCfg); err != nil {
		t.Fatalf("decode generated config: %v", err)
	}
	if mainCfg.Provider != "openrouter" || mainCfg.Model != "openai/gpt-5.5" {
		t.Fatalf("generated config = %+v", mainCfg)
	}
	if len(mainCfg.ProviderConfigs) != 1 || mainCfg.ProviderConfigs[0] != "openrouter.json" {
		t.Fatalf("provider configs = %#v", mainCfg.ProviderConfigs)
	}

	var providerCfg struct {
		Name    string `json:"name"`
		APIType string `json:"api_type"`
		BaseURL string `json:"base_url"`
		APIKey  string `json:"api_key"`
		Models  []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	data, err = os.ReadFile(providerPath)
	if err != nil {
		t.Fatalf("read generated provider config: %v", err)
	}
	if err := json.Unmarshal(data, &providerCfg); err != nil {
		t.Fatalf("decode generated provider config: %v", err)
	}
	if providerCfg.Name != "openrouter" || providerCfg.APIType != "openai" ||
		providerCfg.BaseURL != "https://openrouter.ai/api/v1" || providerCfg.APIKey != "sk-openrouter" {
		t.Fatalf("generated provider config = %+v", providerCfg)
	}
	if len(providerCfg.Models) != 1 || providerCfg.Models[0].Name != "openai/gpt-5.5" {
		t.Fatalf("generated models = %#v", providerCfg.Models)
	}
}

func TestRunSetupRefusesExistingDefaultConfig(t *testing.T) {
	fp := llmtest.New("fake")
	env, _, errw, getenv := fakeProviderEnv(t, []string{"--setup"}, fp, "")
	configDir := filepath.Join(getenv("HOME"), ".config", "harness")
	if err := os.MkdirAll(configDir, 0o700); err != nil {
		t.Fatalf("mkdir config dir: %v", err)
	}
	configPath := filepath.Join(configDir, "config.json")
	if err := os.WriteFile(configPath, []byte(`{"model":"existing"}`), 0o600); err != nil {
		t.Fatalf("write existing config: %v", err)
	}

	code := run(env)
	if code != ui.ExitUsage {
		t.Fatalf("setup with existing config exit = %d, want 2", code)
	}
	if !strings.Contains(errw.String(), "already exists") || !strings.Contains(errw.String(), configPath) {
		t.Fatalf("stderr should name existing config, got %q", errw.String())
	}
}

func TestRunMissingModelUsageError(t *testing.T) {
	fp := llmtest.New("fake")
	env, _, errw, _ := fakeProviderEnv(t, []string{"-p", "hi"}, fp, "")
	code := run(env)
	if code != ui.ExitUsage {
		t.Errorf("missing model should exit 2, got %d", code)
	}
	if !strings.Contains(strings.ToLower(errw.String()), "model") {
		t.Errorf("error should mention the missing model, errw=%q", errw.String())
	}
}

func TestRunProviderConfigSelectsAPITypeAndConnectionSettings(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	providerPath := filepath.Join(dir, "openrouter.json")
	if err := os.WriteFile(cfgPath, []byte(`{
  "provider": "openrouter",
  "model": "openai/gpt-5.1",
  "provider_configs": ["openrouter.json"]
}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := os.WriteFile(providerPath, []byte(`{
  "name": "openrouter",
  "api_type": "openai",
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "sk-openrouter-file",
  "models": [
    {"name":"openai/gpt-5.1","context_window":1000000,"price":{"input":2,"output":8}}
  ]
}`), 0o644); err != nil {
		t.Fatalf("write provider config: %v", err)
	}

	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "ok"}},
		Stop:   llm.StopEndTurn,
	})
	env, _, errw, _ := fakeProviderEnv(t, []string{"-config", cfgPath, "-p", "hi"}, fp, "")
	var got factory.Options
	env.newProvider = func(opts factory.Options) (llm.Provider, error) {
		got = opts
		return fp, nil
	}

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit = %d, want 0; errw=%q", code, errw.String())
	}
	if got.Provider != "openai" {
		t.Fatalf("factory provider = %q, want api_type openai", got.Provider)
	}
	if got.BaseURL != "https://openrouter.ai/api/v1" {
		t.Fatalf("factory base URL = %q", got.BaseURL)
	}
	if got.APIKey != "sk-openrouter-file" {
		t.Fatalf("factory API key = %q", got.APIKey)
	}
	if got.ContextWindow != 1_000_000 {
		t.Fatalf("factory context window = %d, want 1000000", got.ContextWindow)
	}
}

func TestRunMissingProviderConfigWarnsAndSkips(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	if err := os.WriteFile(cfgPath, []byte(`{
  "provider": "openai",
  "model": "gpt-5.1",
  "base_url": "http://localhost:11434/v1",
  "provider_configs": ["missing.json"]
}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "ok"}},
		Stop:   llm.StopEndTurn,
	})
	env, _, errw, _ := fakeProviderEnv(t, []string{"-config", cfgPath, "-p", "hi"}, fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit = %d, want 0; errw=%q", code, errw.String())
	}
	if !strings.Contains(errw.String(), "warning") || !strings.Contains(errw.String(), "missing.json") {
		t.Fatalf("stderr should warn and name missing config, got %q", errw.String())
	}
}

func TestRunBadFlagUsageError(t *testing.T) {
	fp := llmtest.New("fake")
	env, _, _, _ := fakeProviderEnv(t, []string{"-model", "x", "-nonsense"}, fp, "")
	if code := run(env); code != ui.ExitUsage {
		t.Errorf("unknown flag should exit 2, got %d", code)
	}
}

func TestRunOneShotProviderErrorExit1(t *testing.T) {
	fp := llmtest.New("fake", llmtest.Step{Err: &runtimeErr{"upstream"}})
	env, _, errw, _ := fakeProviderEnv(t, []string{"-model", "gpt-5.5", "-p", "go"}, fp, "")
	if code := run(env); code != ui.ExitRuntime {
		t.Errorf("provider error should exit 1, got %d; errw=%q", code, errw.String())
	}
}

// TestRunResumeFlagsWinWarning covers wiring gap #2: when -resume's session file
// disagrees with the flags' provider/model, the flags win and a warning is
// rendered to stderr.
func TestRunResumeFlagsWinWarning(t *testing.T) {
	dir := t.TempDir()
	sessPath := filepath.Join(dir, "prior.json")
	prior := session.Session{
		Version:  session.Version,
		Provider: "openai",
		Model:    "gpt-5.5",
		Created:  time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC),
		System:   "prior system",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: "earlier"}}},
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: "reply"}}},
		},
	}
	data, _ := json.Marshal(prior)
	if err := os.WriteFile(sessPath, data, 0o644); err != nil {
		t.Fatal(err)
	}

	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "ok"}},
		Stop:   llm.StopEndTurn,
	})
	env, _, errw, _ := fakeProviderEnv(t,
		[]string{"-model", "claude-opus-4-8", "-provider", "anthropic", "-resume", sessPath, "-p", "continue"},
		fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("resume one-shot should exit 0, got %d; errw=%q", code, errw.String())
	}
	w := errw.String()
	if !strings.Contains(w, "openai") || !strings.Contains(w, "flags win") {
		t.Errorf("expected a provider override warning, errw=%q", w)
	}
	if !strings.Contains(w, "gpt-5.5") || !strings.Contains(w, "claude-opus-4-8") {
		t.Errorf("expected a model override warning, errw=%q", w)
	}
	// The resumed transcript was carried into the new turn's request.
	if len(fp.Requests) != 1 {
		t.Fatalf("want 1 request, got %d", len(fp.Requests))
	}
	first := fp.Requests[0].Messages[0]
	if first.Content[0].Text != "earlier" {
		t.Errorf("resumed transcript should be re-sent, first message = %q", first.Content[0].Text)
	}
}

func TestRunOneShotConcatenatesFlagAndStdin(t *testing.T) {
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "done"}},
		Stop:   llm.StopEndTurn,
	})
	env, _, _, _ := fakeProviderEnv(t, []string{"-model", "gpt-5.5", "-p", "summarize:"}, fp, "the notes")
	env.stdinPiped = true

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit = %d", code)
	}
	got := fp.Requests[0].Messages[0].Content[0].Text
	if got != "summarize:\nthe notes" {
		t.Errorf("flag and piped stdin should concatenate, got %q", got)
	}
}

func TestRunSavesSessionToDefaultPath(t *testing.T) {
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "ok"}},
		Stop:   llm.StopEndTurn,
	})
	env, _, errw, getenv := fakeProviderEnv(t, []string{"-model", "claude-opus-4-8", "-p", "hi"}, fp, "")
	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit = %d", code)
	}
	// The default auto-save dir lives under XDG_STATE_HOME/harness/sessions.
	sessionsDir := filepath.Join(getenv("XDG_STATE_HOME"), "harness", "sessions")
	entries, err := os.ReadDir(sessionsDir)
	if err != nil || len(entries) == 0 {
		t.Fatalf("expected a saved session under %s: %v (errw=%q)", sessionsDir, err, errw.String())
	}
}

// TestRunSigintExitDuringTurnNoRace exercises the SIGINT-exit-while-a-turn-is-in-
// flight path through run() with a non-nil injected signal channel. The first ^C
// cancels the in-flight turn; a second ^C within the double-press window requests
// exit. The REPL goroutine completes the cancelled turn (its per-turn save and
// usage update) and then performs the final exit save itself, with no concurrent
// writer. Run under -race this is the regression guard for the data race that the
// previous main-side concurrent exit save produced (design §8.4): the run() exit
// wiring is exercised under the race detector, and the SIGINT exit code is 130.
func TestRunSigintExitDuringTurnNoRace(t *testing.T) {
	inTurn := make(chan struct{}) // closed when the turn's stream is in flight
	stdinBlock := make(chan struct{})
	t.Cleanup(func() { close(stdinBlock) }) // unblock the leftover scanner read
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "partial"}},
		Stop:   llm.StopEndTurn,
		Usage:  llm.Usage{InputTokens: 7, OutputTokens: 2},
		Block: func(ctx context.Context) {
			close(inTurn)
			<-ctx.Done() // released by the first ^C cancelling the turn
		},
	})

	dir := t.TempDir()
	getenv := func(k string) string {
		switch k {
		case "HOME":
			return dir
		case "XDG_STATE_HOME":
			return filepath.Join(dir, "state")
		default:
			return ""
		}
	}
	sigCh := make(chan os.Signal, 2)
	var out, errw bytes.Buffer
	env := environment{
		args:     []string{"-model", "claude-opus-4-8"},
		stdin:    &pausingReader{line: []byte("trigger a turn\n"), block: stdinBlock},
		stdout:   &out,
		stderr:   &errw,
		getenv:   getenv,
		now:      func() time.Time { return time.Date(2026, 6, 9, 12, 0, 0, 0, time.UTC) },
		colorTTY: false,
		sigCh:    sigCh,
		newProvider: func(factory.Options) (llm.Provider, error) {
			return fp, nil
		},
	}

	codeCh := make(chan int, 1)
	go func() { codeCh <- run(env) }()

	<-inTurn
	// First ^C cancels the in-flight turn; the second requests exit. The REPL
	// goroutine finishes the cancelled turn (saving + accumulating usage) before
	// acting on the exit request, so there is no concurrent save.
	sigCh <- syscall.SIGINT
	sigCh <- syscall.SIGINT

	code := <-codeCh
	if code != ui.ExitInterrupt {
		t.Fatalf("SIGINT exit should return 130, got %d; errw=%q", code, errw.String())
	}
}

// pausingReader feeds one line, then blocks Read until block is closed. It keeps
// the REPL alive (no premature EOF) while the test drives signals, so the SIGINT
// exit path is what ends the REPL rather than end-of-input.
type pausingReader struct {
	line  []byte
	off   int
	block <-chan struct{}
}

func (r *pausingReader) Read(p []byte) (int, error) {
	if r.off < len(r.line) {
		n := copy(p, r.line[r.off:])
		r.off += n
		return n, nil
	}
	<-r.block
	return 0, io.EOF
}

type runtimeErr struct{ s string }

func (e *runtimeErr) Error() string { return e.s }
