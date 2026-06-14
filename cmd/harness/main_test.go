package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"testing"
	"time"

	"harness/internal/llm"
	"harness/internal/llm/llmtest"
	"harness/internal/modelproxy/protocol"
	"harness/internal/session"
	"harness/internal/tools"
	"harness/internal/ui"
)

// fakeProviderEnv builds an environment whose provider is the scripted fake, so
// run is exercised without real network calls. stateDir/HOME are pinned to a
// temp dir so auto-save paths are deterministic.
func fakeProviderEnv(t *testing.T, args []string, fp *llmtest.FakeProvider, stdin string) (environment, *bytes.Buffer, *bytes.Buffer, func(string) string) {
	env, out, errw, getenv, _ := fakeProviderEnvWithProxy(t, args, fp, stdin)
	return env, out, errw, getenv
}

func fakeProviderEnvWithProxy(t *testing.T, args []string, fp *llmtest.FakeProvider, stdin string) (environment, *bytes.Buffer, *bytes.Buffer, func(string) string, *fakeModelProxy) {
	t.Helper()
	proxy := newFakeModelProxy(t, fp)
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
		args:       append(append([]string{}, args...), "-model-proxy-url", proxy.URL()),
		stdin:      strings.NewReader(stdin),
		stdout:     &out,
		stderr:     &errw,
		getenv:     getenv,
		now:        func() time.Time { return time.Date(2026, 6, 9, 12, 0, 0, 0, time.UTC) },
		colorTTY:   false,
		stdinPiped: false,
		sigCh:      nil, // no signal handling in tests
	}
	return env, &out, &errw, getenv, proxy
}

type fakeModelProxy struct {
	t        *testing.T
	fp       *llmtest.FakeProvider
	server   *httptest.Server
	catalog  protocol.Catalog
	requests []protocol.StreamRequest
}

func newFakeModelProxy(t *testing.T, fp *llmtest.FakeProvider) *fakeModelProxy {
	t.Helper()
	proxy := &fakeModelProxy{
		t:  t,
		fp: fp,
		catalog: protocol.Catalog{
			Providers: []protocol.Provider{
				{
					ID:   "anthropic",
					Name: "Anthropic",
					Models: []protocol.Model{{
						ID:            "claude-opus-4-8",
						ContextWindow: 1_000_000,
					}},
				},
				{
					ID:   "openai",
					Name: "OpenAI",
					Models: []protocol.Model{{
						ID:            "gpt-5.5",
						ContextWindow: 1_050_000,
						Price:         llm.Price{Input: 5, Output: 30, CacheRead: 0.5},
					}},
				},
				{
					ID:   "openrouter",
					Name: "OpenRouter",
					Models: []protocol.Model{{
						ID:            "openai/gpt-5.5",
						ContextWindow: 1_050_000,
						Price:         llm.Price{Input: 5, Output: 30, CacheRead: 0.5},
						Reasoning: &llm.ReasoningInfo{
							Supported: true,
							Options:   []llm.ReasoningOption{{Type: "effort", Values: []string{"low", "medium", "high"}}},
						},
					}},
				},
			},
		},
	}
	proxy.server = httptest.NewServer(proxy)
	t.Cleanup(proxy.server.Close)
	return proxy
}

func (p *fakeModelProxy) URL() string { return p.server.URL }

func (p *fakeModelProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch {
	case r.Method == http.MethodGet && r.URL.Path == "/v1/models":
		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(p.catalog)
	case r.Method == http.MethodPost && r.URL.Path == "/v1/stream":
		var req protocol.StreamRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		p.requests = append(p.requests, req)
		w.Header().Set("content-type", protocol.ContentTypeNDJSON)
		enc := json.NewEncoder(w)
		flusher, _ := w.(http.Flusher)
		for ev, err := range p.fp.Stream(r.Context(), req.Request) {
			if err != nil {
				_ = enc.Encode(protocol.StreamEnvelope{Error: protocol.ErrorFrom(err)})
				if flusher != nil {
					flusher.Flush()
				}
				return
			}
			event := ev
			_ = enc.Encode(protocol.StreamEnvelope{Event: &event})
			if flusher != nil {
				flusher.Flush()
			}
		}
	default:
		http.NotFound(w, r)
	}
}

// okStep is the canned single-step script most wiring tests use: one "ok"
// text delta, then end_turn.
func okStep() llmtest.Step {
	return llmtest.Step{
		Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "ok"}},
		Stop:   llm.StopEndTurn,
	}
}

// okStepWithUsage is okStep with reported token counts attached.
func okStepWithUsage(in, out int) llmtest.Step {
	s := okStep()
	s.Usage = llm.Usage{InputTokens: in, OutputTokens: out}
	return s
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

func TestRunTimestampModes(t *testing.T) {
	cases := []struct {
		name       string
		args       []string
		wantStatus string
		wantNot    string
	}{
		{name: "default short", args: nil, wantStatus: "[12:00:00 model:"},
		{name: "full", args: []string{"-timestamps=full"}, wantStatus: "[2026-06-09 12:00:00 model:"},
		{name: "long alias", args: []string{"-timestamps=long"}, wantStatus: "[2026-06-09 12:00:00 model:"},
		{name: "none", args: []string{"-timestamps=none"}, wantStatus: "[model:", wantNot: "12:00:00"},
		{name: "no timestamps alias", args: []string{"-no-timestamps"}, wantStatus: "[model:", wantNot: "12:00:00"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fp := llmtest.New("fake", llmtest.Step{
				Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "42"}},
				Stop:   llm.StopEndTurn,
			})
			args := append([]string{"-model", "claude-opus-4-8", "-p", "what is the answer"}, tc.args...)
			env, out, errw, _ := fakeProviderEnv(t, args, fp, "")
			if code := run(env); code != ui.ExitOK {
				t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
			}
			if out.String() != "42\n" {
				t.Fatalf("stdout = %q, want raw assistant text", out.String())
			}
			if strings.Contains(errw.String(), "12:00:00 session:") || strings.Contains(errw.String(), "12:00:00 provider:") {
				t.Fatalf("startup diagnostics should not be timestamped: %q", errw.String())
			}
			if !strings.Contains(errw.String(), tc.wantStatus) {
				t.Fatalf("stderr %q missing %q", errw.String(), tc.wantStatus)
			}
			if tc.wantNot != "" && strings.Contains(errw.String(), tc.wantNot) {
				t.Fatalf("stderr %q should not contain %q", errw.String(), tc.wantNot)
			}
		})
	}
}

func TestRunREPLModelCommandSwitchesProvider(t *testing.T) {
	fp := llmtest.New("fake", okStep())
	env, _, errw, _, proxy := fakeProviderEnvWithProxy(t,
		[]string{"-model", "claude-opus-4-8"},
		fp,
		"/model gpt-5.5\nhello\n/exit\n",
	)

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if len(proxy.requests) != 1 || proxy.requests[0].Provider != "openai" || proxy.requests[0].Request.Model != "gpt-5.5" {
		t.Fatalf("proxy requests = %+v, want one openai/gpt-5.5 request", proxy.requests)
	}
	if !strings.Contains(errw.String(), "model switched") {
		t.Errorf("switch should be acknowledged, errw=%q", errw.String())
	}
}

func TestRunREPLModelCommandAcceptsProviderQualifiedModel(t *testing.T) {
	fp := llmtest.New("fake", okStep())
	env, _, errw, _, proxy := fakeProviderEnvWithProxy(t,
		[]string{"-model", "claude-opus-4-8"},
		fp,
		"/model openrouter:openai/gpt-5.5\nhello\n/exit\n",
	)

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if len(proxy.requests) != 1 || proxy.requests[0].Provider != "openrouter" || proxy.requests[0].Request.Model != "openai/gpt-5.5" {
		t.Fatalf("proxy requests = %+v, want one openrouter/openai/gpt-5.5 request", proxy.requests)
	}
}

func TestRunREPLModelCommandPromptsConfiguredProviderAndModel(t *testing.T) {
	fp := llmtest.New("fake", okStep())
	env, _, errw, _, proxy := fakeProviderEnvWithProxy(t,
		[]string{"-model", "claude-opus-4-8"},
		fp,
		"/model\nopenrouter\nopenai/gpt-5.5\nhello\n/exit\n",
	)

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if len(proxy.requests) != 1 || proxy.requests[0].Provider != "openrouter" || proxy.requests[0].Request.Model != "openai/gpt-5.5" {
		t.Fatalf("proxy requests = %+v, want one openrouter-local request", proxy.requests)
	}
	stderr := errw.String()
	if !strings.Contains(stderr, "Providers 1-3 of 3") || !strings.Contains(stderr, "Models for openrouter") ||
		!strings.Contains(stderr, "model switched") {
		t.Fatalf("/model should render provider/model picker and acknowledge switch, stderr=%q", stderr)
	}
}

// TestRunEnvBlockReportsAbsoluteCwd is the regression test for the env block
// emitting `cwd: .` instead of the absolute working directory (design §8.5).
// main must populate EnvOptions.Dir via os.Getwd so the system prompt the model
// receives names a real absolute path it can reason about.
func TestRunEnvBlockReportsAbsoluteCwd(t *testing.T) {
	fp := llmtest.New("fake", okStepWithUsage(1, 1))
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
		"-p", "-provider", "-model", "-model-proxy-url", "-system", "-system-override",
		"-no-env", "-resume", "-session", "-max-turns", "-default-context-window", "-context-window",
		"-reasoning-effort", "-agent", "-v", "-tool-stream", "-q", "-quiet", "-log-level", "-no-color", "-config", "-prompt",
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

func TestRunPromptsForModelAndSavesConfigWhenModelMissing(t *testing.T) {
	fp := llmtest.New("fake", okStep())
	env, _, errw, getenv := fakeProviderEnv(t, nil, fp, "2\n1\n/exit\n")

	code := run(env)
	if code != ui.ExitOK {
		t.Fatalf("exit = %d, want ok; errw=%q", code, errw.String())
	}
	if len(fp.Requests) != 0 {
		t.Fatalf("provider should not be called before a prompt, got %d requests", len(fp.Requests))
	}
	configPath := filepath.Join(getenv("HOME"), ".config", "harness", "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	var got struct {
		Provider string `json:"provider"`
		Model    string `json:"model"`
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("decode config: %v\n%s", err, data)
	}
	if got.Provider != "openai" || got.Model != "gpt-5.5" {
		t.Fatalf("saved provider/model = %q/%q, want openai/gpt-5.5\n%s", got.Provider, got.Model, data)
	}
	if !strings.Contains(errw.String(), "Select a provider and model") {
		t.Fatalf("stderr should show startup picker, got %q", errw.String())
	}
}

func TestRunReasoningEffortRejectedWhenProxyCatalogSaysUnsupported(t *testing.T) {
	fp := llmtest.New("fake")
	env, _, errw, _, proxy := fakeProviderEnvWithProxy(t, []string{"-provider", "openai", "-model", "gpt-4o", "-reasoning-effort", "high", "-p", "hi"}, fp, "")
	for i := range proxy.catalog.Providers {
		if proxy.catalog.Providers[i].ID == "openai" {
			proxy.catalog.Providers[i].Models = append(proxy.catalog.Providers[i].Models, protocol.Model{
				ID:            "gpt-4o",
				ContextWindow: 128000,
				Reasoning:     &llm.ReasoningInfo{Supported: false},
			})
		}
	}

	if code := run(env); code != ui.ExitUsage {
		t.Fatalf("exit = %d, want usage error; errw=%q", code, errw.String())
	}
	if len(fp.Requests) != 0 {
		t.Fatalf("provider should not be called after validation failure, got %d requests", len(fp.Requests))
	}
	if !strings.Contains(errw.String(), "does not support reasoning effort") {
		t.Fatalf("stderr should explain unsupported effort, got %q", errw.String())
	}
}

func TestRunReasoningEffortRejectedWhenProxyValueUnsupported(t *testing.T) {
	fp := llmtest.New("fake")
	env, _, errw, _, proxy := fakeProviderEnvWithProxy(t, []string{"-provider", "openai", "-model", "gpt-5-pro", "-reasoning-effort", "xhigh", "-p", "hi"}, fp, "")
	for i := range proxy.catalog.Providers {
		if proxy.catalog.Providers[i].ID == "openai" {
			proxy.catalog.Providers[i].Models = append(proxy.catalog.Providers[i].Models, protocol.Model{
				ID:            "gpt-5-pro",
				ContextWindow: 400000,
				Reasoning: &llm.ReasoningInfo{
					Supported: true,
					Options:   []llm.ReasoningOption{{Type: "effort", Values: []string{"high"}}},
				},
			})
		}
	}

	if code := run(env); code != ui.ExitUsage {
		t.Fatalf("exit = %d, want usage error; errw=%q", code, errw.String())
	}
	if len(fp.Requests) != 0 {
		t.Fatalf("provider should not be called after validation failure, got %d requests", len(fp.Requests))
	}
	if !strings.Contains(errw.String(), `supported: high`) {
		t.Fatalf("stderr should list supported effort values, got %q", errw.String())
	}
}

func TestRunContextWindowOverrideStillWins(t *testing.T) {
	fp := llmtest.New("fake", okStep())
	env, _, errw, _ := fakeProviderEnv(t, []string{
		"-provider", "openrouter",
		"-model", "openai/gpt-5.5",
		"-context-window", "64000",
		"-p", "hi",
	}, fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit = %d, want 0; errw=%q", code, errw.String())
	}
	if len(fp.Requests) != 1 || fp.Requests[0].Model != "openai/gpt-5.5" {
		t.Fatalf("requests = %+v", fp.Requests)
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
	// A plain (non-API, non-cancel) provider error is retryable, so it must
	// recur through the whole per-model-turn budget (1 + 2 retries) to surface as the
	// turn-fatal exit-1 it models.
	fail := llmtest.Step{Err: &runtimeErr{"upstream"}}
	fp := llmtest.New("fake", fail, fail, fail)
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
	sessPath := filepath.Join(dir, "prior")
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
	if err := prior.Save(sessPath); err != nil {
		t.Fatal(err)
	}

	fp := llmtest.New("fake", okStep())
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
	fp := llmtest.New("fake", okStep())
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

func TestRunSessionReplaySubcommand(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "session")
	if err := session.AppendEvent(dir, session.Event{Type: session.EventUser, Turn: 1, Text: "hello"}); err != nil {
		t.Fatalf("append event: %v", err)
	}
	if err := session.AppendEvent(dir, session.Event{Type: session.EventAssistantDelta, Turn: 1, Text: "world"}); err != nil {
		t.Fatalf("append event: %v", err)
	}
	var out, errw bytes.Buffer
	code := run(environment{
		args:   []string{"session", "replay", dir},
		stdout: &out,
		stderr: &errw,
		getenv: func(string) string { return "" },
		now:    time.Now,
	})
	if code != ui.ExitOK {
		t.Fatalf("exit = %d; stderr=%q", code, errw.String())
	}
	if !strings.Contains(out.String(), "> hello") || !strings.Contains(out.String(), "world") {
		t.Fatalf("unexpected replay output: %q", out.String())
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
	proxy := newFakeModelProxy(t, fp)

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
		args:     []string{"-model", "claude-opus-4-8", "-model-proxy-url", proxy.URL()},
		stdin:    &pausingReader{line: []byte("trigger a turn\n"), block: stdinBlock},
		stdout:   &out,
		stderr:   &errw,
		getenv:   getenv,
		now:      func() time.Time { return time.Date(2026, 6, 9, 12, 0, 0, 0, time.UTC) },
		colorTTY: false,
		sigCh:    sigCh,
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

func TestLoadAgentsMD_Missing(t *testing.T) {
	dir := t.TempDir()
	content, err := loadAgentsMD(dir)
	if err != nil {
		t.Fatalf("loadAgentsMD should not error on missing file: %v", err)
	}
	if content != "" {
		t.Errorf("loadAgentsMD should return empty string for missing file, got %q", content)
	}
}

func TestLoadAgentsMD_Present(t *testing.T) {
	dir := t.TempDir()
	expected := "# Project Rules\n\nAlways write tests."
	if err := os.WriteFile(filepath.Join(dir, "AGENTS.md"), []byte(expected), 0o644); err != nil {
		t.Fatalf("write AGENTS.md: %v", err)
	}
	content, err := loadAgentsMD(dir)
	if err != nil {
		t.Fatalf("loadAgentsMD should not error: %v", err)
	}
	if content != expected {
		t.Errorf("loadAgentsMD returned %q, want %q", content, expected)
	}
}

func TestLoadAgentsMD_EmptyDir(t *testing.T) {
	content, err := loadAgentsMD("")
	if err != nil {
		t.Fatalf("loadAgentsMD should not error on empty dir: %v", err)
	}
	if content != "" {
		t.Errorf("loadAgentsMD should return empty string for empty dir, got %q", content)
	}
}

// runInDirSystemPrompt runs a one-shot turn from dir (the chdir is load-bearing:
// AGENTS.md auto-discovery reads the real working directory) and returns the
// system prompt the fake provider received.
func runInDirSystemPrompt(t *testing.T, dir string) string {
	t.Helper()
	originalDir, err := os.Getwd()
	if err != nil {
		t.Fatalf("get cwd: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("chdir: %v", err)
	}
	defer os.Chdir(originalDir)

	fp := llmtest.New("fake", okStepWithUsage(1, 1))
	env, _, errw, _ := fakeProviderEnv(t, []string{"-model", "claude-opus-4-8", "-p", "hi"}, fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if len(fp.Requests) != 1 {
		t.Fatalf("want 1 request, got %d", len(fp.Requests))
	}
	return fp.Requests[0].System
}

func TestRunAgentsMDDiscovery(t *testing.T) {
	agentsMD := "# Custom Rules\n\nUse camelCase variables."
	cases := []struct {
		name         string
		writeAgents  bool
		wantContains []string
	}{
		{name: "included when present", writeAgents: true, wantContains: []string{agentsMD}},
		{name: "builtin prompt when missing", writeAgents: false, wantContains: []string{"You are a coding agent", "<env>"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if tc.writeAgents {
				if err := os.WriteFile(filepath.Join(dir, "AGENTS.md"), []byte(agentsMD), 0o644); err != nil {
					t.Fatalf("write AGENTS.md: %v", err)
				}
			}
			system := runInDirSystemPrompt(t, dir)
			for _, want := range tc.wantContains {
				if !strings.Contains(system, want) {
					t.Errorf("system prompt should contain %q; system=%q", want, system)
				}
			}
		})
	}
}

// toolNames extracts the advertised tool names from a recorded request.
func toolNames(req llm.Request) []string {
	names := make([]string, len(req.Tools))
	for i, s := range req.Tools {
		names[i] = s.Name
	}
	return names
}

// Default (auto) agent advertises the default tool set plus delegate and carries
// no agent-specific section.
func TestRunDefaultAgentTools(t *testing.T) {
	fp := llmtest.New("fake", okStepWithUsage(1, 1))
	env, _, errw, _ := fakeProviderEnv(t, []string{"-model", "claude-opus-4-8", "-p", "hi"}, fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	want := expectedDefaultToolNames()
	if got := toolNames(fp.Requests[0]); !slices.Equal(got, want) {
		t.Errorf("default agent tools = %v, want %v", got, want)
	}
	if strings.Contains(fp.Requests[0].System, "plan agent") || strings.Contains(fp.Requests[0].System, "independent agent") {
		t.Errorf("default agent should carry no agent section; system=%q", fp.Requests[0].System)
	}
}

func TestRunDelegateToolUsesCurrentAgentTools(t *testing.T) {
	fp := llmtest.New("fake",
		llmtest.Step{
			Events: []llm.StreamEvent{{
				Kind:      llm.EventToolCallDone,
				ToolID:    "call_delegate",
				ToolName:  "delegate",
				ToolInput: json.RawMessage(`{"task":"inspect only"}`),
			}},
			Stop:  llm.StopToolUse,
			Usage: llm.Usage{InputTokens: 10, OutputTokens: 2},
		},
		llmtest.Step{
			Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "child report"}},
			Stop:   llm.StopEndTurn,
			Usage:  llm.Usage{InputTokens: 30, OutputTokens: 7},
		},
		llmtest.Step{
			Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "parent done"}},
			Stop:   llm.StopEndTurn,
			Usage:  llm.Usage{InputTokens: 20, OutputTokens: 4},
		},
	)
	env, out, errw, _ := fakeProviderEnv(t, []string{"-model", "claude-opus-4-8", "-p", "hi"}, fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if !strings.Contains(out.String(), "parent done") {
		t.Fatalf("parent final text missing from stdout: %q", out.String())
	}
	if len(fp.Requests) != 3 {
		t.Fatalf("provider requests = %d, want parent/tool, child, parent/final", len(fp.Requests))
	}
	if !slices.Contains(toolNames(fp.Requests[0]), "delegate") {
		t.Fatalf("parent request did not advertise delegate: %v", toolNames(fp.Requests[0]))
	}
	childTools := toolNames(fp.Requests[1])
	wantChildTools := expectedDefaultToolNames()
	if !slices.Equal(childTools, wantChildTools) {
		t.Fatalf("child request tools = %v, want current agent tools %v", childTools, wantChildTools)
	}
	if got := fp.Requests[1].Messages[0].Content[0].Text; got != "inspect only" {
		t.Fatalf("child task = %q", got)
	}
	if !strings.Contains(errw.String(), "delegate] task=\"inspect only\"") {
		t.Fatalf("delegate tool result was not rendered: %q", errw.String())
	}
	if !strings.Contains(errw.String(), "60 (60) in / 13 (13) out") {
		t.Fatalf("turn usage should include parent and child model calls, stderr=%q", errw.String())
	}
}

func TestRunDelegateUsesSwitchedModelAndAgent(t *testing.T) {
	fp := llmtest.New("fake",
		llmtest.Step{
			Events: []llm.StreamEvent{{
				Kind:      llm.EventToolCallDone,
				ToolID:    "call_delegate",
				ToolName:  "delegate",
				ToolInput: json.RawMessage(`{"task":"inspect after switches"}`),
			}},
			Stop: llm.StopToolUse,
		},
		llmtest.Step{
			Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "child report"}},
			Stop:   llm.StopEndTurn,
		},
		llmtest.Step{
			Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "parent done"}},
			Stop:   llm.StopEndTurn,
		},
	)
	env, _, errw, _ := fakeProviderEnv(t,
		[]string{"-model", "claude-opus-4-8"},
		fp,
		"/model gpt-5.5\n/agent plan\nhi\n/exit\n",
	)

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if len(fp.Requests) != 3 {
		t.Fatalf("provider requests = %d, want 3", len(fp.Requests))
	}
	child := fp.Requests[1]
	if child.Model != "gpt-5.5" {
		t.Fatalf("delegate child model = %q, want switched model", child.Model)
	}
	if !strings.Contains(child.System, "plan agent") {
		t.Fatalf("delegate child system should include switched agent prompt, system=%q", child.System)
	}
}

func TestRunLogsUnavailableToolsAtLaunch(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	fp := llmtest.New("fake", okStepWithUsage(1, 1))
	env, _, errw, _ := fakeProviderEnv(t, []string{"-model", "claude-opus-4-8", "-p", "hi"}, fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	got := errw.String()
	for _, want := range []string{
		`[warn] [cli_tools] Tool "rg" is disabled. Reason: "rg" binary not found.`,
		`[warn] [cli_tools] Tool "git" is disabled. Reason: "git" binary not found.`,
		`[warn] [cli_tools] Tool "git_readonly" is disabled. Reason: "git" binary not found.`,
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("stderr missing %q:\n%s", want, got)
		}
	}
	for _, name := range []string{"rg", "git", "git_readonly"} {
		if slices.Contains(toolNames(fp.Requests[0]), name) {
			t.Fatalf("request advertised unavailable tool %q: %v", name, toolNames(fp.Requests[0]))
		}
	}
}

func TestRunQuietSuppressesUnavailableToolWarnings(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	fp := llmtest.New("fake", okStepWithUsage(1, 1))
	env, _, errw, _ := fakeProviderEnv(t, []string{"-model", "claude-opus-4-8", "--quiet", "-p", "hi"}, fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if strings.Contains(errw.String(), "[cli_tools]") {
		t.Fatalf("quiet should suppress disabled-tool warnings, stderr=%q", errw.String())
	}
}

func TestRunLogLevelSuppressesUnavailableToolWarnings(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	fp := llmtest.New("fake", okStepWithUsage(1, 1))
	env, _, errw, _ := fakeProviderEnv(t, []string{"-model", "claude-opus-4-8", "--log-level", "error", "-p", "hi"}, fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if strings.Contains(errw.String(), "[cli_tools]") {
		t.Fatalf("log-level error should suppress warn diagnostics, stderr=%q", errw.String())
	}
}

// Plan agent advertises only its read-only tool set and includes its prompt.
func TestRunPlanAgentRestrictsToolsAndAddsPrompt(t *testing.T) {
	fp := llmtest.New("fake", okStepWithUsage(1, 1))
	env, _, errw, _ := fakeProviderEnv(t, []string{"-model", "claude-opus-4-8", "-agent", "plan", "-p", "hi"}, fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	want := expectedPlanToolNames()
	if got := toolNames(fp.Requests[0]); !slices.Equal(got, want) {
		t.Errorf("plan agent tools = %v, want %v", got, want)
	}
	if !strings.Contains(fp.Requests[0].System, "plan agent") {
		t.Errorf("plan agent system prompt should include the plan section; system=%q", fp.Requests[0].System)
	}
}

// An unknown agent is a startup usage error that lists the available agents.
func TestRunUnknownAgentIsUsageError(t *testing.T) {
	fp := llmtest.New("fake")
	env, _, errw, _ := fakeProviderEnv(t, []string{"-model", "claude-opus-4-8", "-agent", "bogus", "-p", "hi"}, fp, "")

	if code := run(env); code != ui.ExitUsage {
		t.Fatalf("exit code = %d, want ExitUsage; errw=%q", code, errw.String())
	}
	got := errw.String()
	if !strings.Contains(got, "bogus") || !strings.Contains(got, "auto") || !strings.Contains(got, "plan") {
		t.Errorf("error should name the bad agent and list valid ones, errw=%q", got)
	}
	if len(fp.Requests) != 0 {
		t.Errorf("no turn should run for an unknown agent, got %d requests", len(fp.Requests))
	}
}

// A config agent entry overriding only the prompt keeps the built-in tool list.
func TestRunConfigAgentPromptOverrideKeepsTools(t *testing.T) {
	fp := llmtest.New("fake", okStepWithUsage(1, 1))
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	if err := os.WriteFile(cfgPath, []byte(`{"agent":"plan","agents":{"plan":{"prompt":"CUSTOM PLAN GUIDANCE"}}}`), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	env, _, errw, _ := fakeProviderEnv(t, []string{"-config", cfgPath, "-model", "claude-opus-4-8", "-p", "hi"}, fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	want := expectedPlanToolNames()
	if got := toolNames(fp.Requests[0]); !slices.Equal(got, want) {
		t.Errorf("plan tools should be preserved by a prompt-only override = %v, want %v", got, want)
	}
	if !strings.Contains(fp.Requests[0].System, "CUSTOM PLAN GUIDANCE") {
		t.Errorf("custom plan prompt should be used; system=%q", fp.Requests[0].System)
	}
}

func TestRunConfigAgentCanSetProviderAndModel(t *testing.T) {
	fp := llmtest.New("fake", okStepWithUsage(1, 1))
	cfgPath := filepath.Join(t.TempDir(), "config.json")
	cfg := `{
		"agent":"style",
		"agents":{
			"style":{
				"description":"Style review",
				"provider":"openai",
				"model":"gpt-5.5",
				"allowed_tools":["read_file"],
				"prompt":"STYLE REVIEW PROMPT"
			}
		}
	}`
	if err := os.WriteFile(cfgPath, []byte(cfg), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	env, _, errw, _, proxy := fakeProviderEnvWithProxy(t, []string{"-config", cfgPath, "-p", "hi"}, fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if len(proxy.requests) != 1 || proxy.requests[0].Provider != "openai" || proxy.requests[0].Request.Model != "gpt-5.5" {
		t.Fatalf("proxy requests = %+v, want openai/gpt-5.5", proxy.requests)
	}
	if got := toolNames(proxy.requests[0].Request); !slices.Equal(got, []string{"read_file"}) {
		t.Fatalf("agent tools = %v, want [read_file]", got)
	}
	if !strings.Contains(proxy.requests[0].Request.System, "STYLE REVIEW PROMPT") {
		t.Fatalf("agent prompt missing from system: %q", proxy.requests[0].Request.System)
	}
}

func TestRunREPLAgentListShowsProviderModelConfig(t *testing.T) {
	fp := llmtest.New("fake")
	cfgPath := filepath.Join(t.TempDir(), "config.json")
	cfg := `{
		"agents":{
			"security":{
				"description":"Security review",
				"provider":"openai",
				"model":"gpt-5.5",
				"allowed_tools":["read_file"],
				"prompt":"SECURITY"
			}
		}
	}`
	if err := os.WriteFile(cfgPath, []byte(cfg), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	env, _, errw, _ := fakeProviderEnv(t, []string{"-config", cfgPath, "-model", "claude-opus-4-8"}, fp, "/agent\n/exit\n")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	got := errw.String()
	if !strings.Contains(got, "security       [openai/gpt-5.5] - Security review") {
		t.Fatalf("/agent output missing configured provider/model, stderr=%q", got)
	}
	if !strings.Contains(got, "current agent: auto [anthropic/claude-opus-4-8]") ||
		!strings.Contains(got, "auto (current) [inherit current]") {
		t.Fatalf("/agent output missing inherited provider/model, stderr=%q", got)
	}
	if len(fp.Requests) != 0 {
		t.Fatalf("/agent listing should not call model, got %d requests", len(fp.Requests))
	}
}

func TestRunDelegateNamedAgentUsesDefinition(t *testing.T) {
	fp := llmtest.New("fake",
		llmtest.Step{
			Events: []llm.StreamEvent{{
				Kind:      llm.EventToolCallDone,
				ToolID:    "call_delegate",
				ToolName:  "delegate",
				ToolInput: json.RawMessage(`{"task":"check style","agent":"style"}`),
			}},
			Stop: llm.StopToolUse,
		},
		llmtest.Step{
			Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "style report"}},
			Stop:   llm.StopEndTurn,
		},
		llmtest.Step{
			Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "parent done"}},
			Stop:   llm.StopEndTurn,
		},
	)
	cfgPath := filepath.Join(t.TempDir(), "config.json")
	cfg := `{
		"agents":{
			"style":{
				"provider":"openai",
				"model":"gpt-5.5",
				"allowed_tools":["read_file"],
				"prompt":"STYLE AGENT PROMPT"
			}
		}
	}`
	if err := os.WriteFile(cfgPath, []byte(cfg), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	env, out, errw, _, proxy := fakeProviderEnvWithProxy(t, []string{"-config", cfgPath, "-model", "claude-opus-4-8", "-p", "hi"}, fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if !strings.Contains(out.String(), "parent done") {
		t.Fatalf("parent final text missing from stdout: %q", out.String())
	}
	if len(proxy.requests) != 3 {
		t.Fatalf("proxy requests = %d, want parent/tool, child, parent/final", len(proxy.requests))
	}
	child := proxy.requests[1]
	if child.Provider != "openai" || child.Request.Model != "gpt-5.5" {
		t.Fatalf("delegate child provider/model = %q/%q, want openai/gpt-5.5", child.Provider, child.Request.Model)
	}
	if got := toolNames(child.Request); !slices.Equal(got, []string{"read_file"}) {
		t.Fatalf("delegate child tools = %v, want [read_file]", got)
	}
	if !strings.Contains(child.Request.System, "STYLE AGENT PROMPT") {
		t.Fatalf("delegate child system missing style prompt: %q", child.Request.System)
	}
}

// /mode remains an alias for /agent and switches the advertised tool set on the next turn.
func TestRunREPLModeAliasSwitchesTools(t *testing.T) {
	fp := llmtest.New("fake", okStepWithUsage(1, 1))
	env, _, errw, _ := fakeProviderEnv(t,
		[]string{"-model", "claude-opus-4-8"},
		fp,
		"/mode plan\nhello\n/exit\n",
	)
	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if len(fp.Requests) != 1 {
		t.Fatalf("want 1 post-switch request, got %d", len(fp.Requests))
	}
	want := expectedPlanToolNames()
	if got := toolNames(fp.Requests[0]); !slices.Equal(got, want) {
		t.Errorf("post-/mode tools = %v, want plan set %v", got, want)
	}
	if !strings.Contains(errw.String(), "agent switched: plan") {
		t.Errorf("switch should be acknowledged, errw=%q", errw.String())
	}
}

// A resumed session restores its active agent (and thus its restricted tool set)
// when no -agent flag overrides it.
func TestRunResumeRestoresAgent(t *testing.T) {
	dir := t.TempDir()
	sessPath := filepath.Join(dir, "prior")
	prior := session.Session{
		Version:  session.Version,
		Provider: "anthropic",
		Model:    "claude-opus-4-8",
		System:   "you are a test",
		Agent:    "plan",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: "hi"}}},
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: "hello"}}},
		},
	}
	if err := prior.Save(sessPath); err != nil {
		t.Fatalf("save prior session: %v", err)
	}

	fp := llmtest.New("fake", okStepWithUsage(1, 1))
	env, _, errw, _ := fakeProviderEnv(t, []string{"-model", "claude-opus-4-8", "-resume", sessPath, "-p", "again"}, fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	want := expectedPlanToolNames()
	if got := toolNames(fp.Requests[0]); !slices.Equal(got, want) {
		t.Errorf("resumed plan session tools = %v, want %v", got, want)
	}
}

// TestRunREPLToolsCommandListsTools verifies that /tools prints built-in tools
// (always including read_file and delegate), identifies disabled tools (rg when
// unavailable), and does not trigger any model request.
func TestRunREPLToolsCommandListsTools(t *testing.T) {
	fp := llmtest.New("fake", okStepWithUsage(1, 1))
	env, _, errw, _ := fakeProviderEnv(t,
		[]string{"-model", "claude-opus-4-8"},
		fp,
		"/tools\n/exit\n",
	)
	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	// /tools is a meta-command — it must not trigger a model request.
	if len(fp.Requests) != 0 {
		t.Errorf("want 0 requests, got %d", len(fp.Requests))
	}
	out := errw.String()
	if !strings.Contains(out, "built-in tools:") {
		t.Errorf("/tools output missing built-in heading, got:\n%s", out)
	}
	for _, name := range tools.DefaultNames() {
		if !toolsOutputHasDescribedTool(out, name) {
			t.Errorf("/tools output missing built-in tool %q, got:\n%s", name, out)
		}
	}
	if !strings.Contains(out, "  delegate     - Run a configured delegate agent") {
		t.Errorf("/tools output missing delegate, got:\n%s", out)
	}
	if !tools.RipgrepAvailable() {
		if !strings.Contains(out, "disabled tools:") {
			t.Errorf("/tools output missing disabled section when rg unavailable, got:\n%s", out)
		}
		if !strings.Contains(out, "rg") {
			t.Errorf("/tools output missing disabled rg entry, got:\n%s", out)
		}
	}
}

func toolsOutputHasDescribedTool(output, name string) bool {
	for _, line := range strings.Split(output, "\n") {
		if strings.Contains(line, "  "+name) && strings.Contains(line, " - ") {
			return true
		}
	}
	return false
}

func expectedPlanToolNames() []string {
	names := []string{"read_file", "list_dir", "grep"}
	if tools.RipgrepAvailable() {
		names = append(names, "rg")
	}
	names = append(names, "web_fetch")
	if tools.GitAvailable() {
		names = append(names, "git_readonly")
	}
	return append(names, "write_tmp_file", "delegate")
}

func expectedDefaultToolNames() []string {
	return append(tools.DefaultNames(), "delegate")
}
