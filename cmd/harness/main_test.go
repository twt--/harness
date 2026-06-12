package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"testing"
	"time"

	"harness/internal/llm"
	"harness/internal/llm/factory"
	"harness/internal/llm/llmtest"
	"harness/internal/modelsdev"
	"harness/internal/session"
	"harness/internal/tools"
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

// captureOptions replaces env's provider factory with one that records the
// factory.Options it receives (still returning fp), so tests can assert on the
// resolved wiring through the returned pointer after run() completes.
func captureOptions(env *environment, fp *llmtest.FakeProvider) *factory.Options {
	var got factory.Options
	env.newProvider = func(opts factory.Options) (llm.Provider, error) {
		got = opts
		return fp, nil
	}
	return &got
}

func testModelsDevCatalog(t *testing.T) *modelsdev.Catalog {
	t.Helper()
	catalog, err := modelsdev.Decode(strings.NewReader(`{
  "openai": {
    "id": "openai",
    "name": "OpenAI",
    "env": ["OPENAI_API_KEY"],
    "npm": "@ai-sdk/openai",
    "models": {
      "gpt-5.5": {
        "id": "gpt-5.5",
        "name": "GPT-5.5",
        "release_date": "2026-06-01",
        "reasoning": true,
        "reasoning_options": [{"type":"effort","values":["none","low","medium","high","xhigh"]}],
        "limit": {"context": 1050000},
        "cost": {"input": 5, "output": 30, "cache_read": 0.5}
      }
    }
  },
  "openrouter": {
    "id": "openrouter",
    "name": "OpenRouter",
    "api": "https://openrouter.ai/api/v1",
    "env": ["OPENROUTER_API_KEY"],
    "npm": "@openrouter/ai-sdk-provider",
    "models": {
      "openai/gpt-5.5": {
        "id": "openai/gpt-5.5",
        "name": "GPT-5.5",
        "release_date": "2026-06-01",
        "reasoning": true,
        "reasoning_options": [{"type":"effort","values":["low","medium","high"]}],
        "limit": {"context": 1050000},
        "cost": {"input": 5, "output": 30, "cache_read": 0.5}
      }
    }
  },
  "deepseek": {
    "id": "deepseek",
    "name": "DeepSeek",
    "api": "https://api.deepseek.com",
    "env": ["DEEPSEEK_API_KEY"],
    "models": {
      "deepseek-chat": {
        "id": "deepseek-chat",
        "name": "DeepSeek Chat",
        "release_date": "2026-01-01",
        "limit": {"context": 128000},
        "cost": {"input": 0.27, "output": 1.1}
      }
    }
  }
}`))
	if err != nil {
		t.Fatalf("decode test models.dev catalog: %v", err)
	}
	return catalog
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

func TestRunREPLModelCommandSwitchesProvider(t *testing.T) {
	initial := llmtest.New("initial")
	switched := llmtest.New("switched", okStep())
	env, _, errw, _ := fakeProviderEnv(t,
		[]string{"-model", "claude-opus-4-8"},
		initial,
		"/model gpt-5.5\nhello\n/exit\n",
	)
	var opts []factory.Options
	env.newProvider = func(opt factory.Options) (llm.Provider, error) {
		opts = append(opts, opt)
		if len(opts) == 1 {
			return initial, nil
		}
		return switched, nil
	}

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if len(opts) != 2 {
		t.Fatalf("provider factory calls = %d, want 2 (%+v)", len(opts), opts)
	}
	if opts[1].Provider != "openai" || opts[1].Model != "gpt-5.5" {
		t.Fatalf("switch provider options = %+v, want openai/gpt-5.5", opts[1])
	}
	if len(initial.Requests) != 0 {
		t.Fatalf("initial provider should receive no turns after switch, got %d", len(initial.Requests))
	}
	if len(switched.Requests) != 1 || switched.Requests[0].Model != "gpt-5.5" {
		t.Fatalf("switched requests = %+v, want one gpt-5.5 request", switched.Requests)
	}
	if !strings.Contains(errw.String(), "model switched") {
		t.Errorf("switch should be acknowledged, errw=%q", errw.String())
	}
}

func TestRunREPLModelCommandAcceptsProviderQualifiedModel(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	if err := os.WriteFile(cfgPath, []byte(`{
  "provider": "openai",
  "model": "gpt-5.5",
  "provider_configs": ["openai.json", "openrouter.json"]
}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "openai.json"), []byte(`{
  "name": "openai",
  "api_type": "openai",
  "base_url": "https://api.openai.com/v1",
  "models": [{"name":"gpt-5.5","context_window":400000}]
}`), 0o644); err != nil {
		t.Fatalf("write openai provider: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "openrouter.json"), []byte(`{
  "name": "openrouter",
  "api_type": "openai",
  "base_url": "https://openrouter.ai/api/v1",
  "models": [{"name":"gpt-5.5","context_window":1050000}]
}`), 0o644); err != nil {
		t.Fatalf("write openrouter provider: %v", err)
	}

	initial := llmtest.New("initial")
	switched := llmtest.New("switched", okStep())
	env, _, errw, _ := fakeProviderEnv(t,
		[]string{"-config", cfgPath},
		initial,
		"/model openrouter:gpt-5.5\nhello\n/exit\n",
	)
	var opts []factory.Options
	env.newProvider = func(opt factory.Options) (llm.Provider, error) {
		opts = append(opts, opt)
		if len(opts) == 1 {
			return initial, nil
		}
		return switched, nil
	}

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if len(opts) != 2 {
		t.Fatalf("provider factory calls = %d, want 2 (%+v)", len(opts), opts)
	}
	got := opts[1]
	if got.Provider != "openai" || got.ProviderName != "openrouter" || got.Model != "gpt-5.5" ||
		got.BaseURL != "https://openrouter.ai/api/v1" || got.ContextWindow != 1_050_000 {
		t.Fatalf("qualified switch options = %+v", got)
	}
	if len(switched.Requests) != 1 || switched.Requests[0].Model != "gpt-5.5" {
		t.Fatalf("switched requests = %+v, want one provider-local gpt-5.5 request", switched.Requests)
	}
}

func TestRunREPLModelCommandPromptsConfiguredProviderAndModel(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	if err := os.WriteFile(cfgPath, []byte(`{
  "provider": "openai",
  "model": "gpt-5.5",
  "provider_configs": ["openai.json", "openrouter.json"]
}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "openai.json"), []byte(`{
  "name": "openai",
  "api_type": "responses",
  "base_url": "https://api.openai.com/v1",
  "models": [{"name":"gpt-5.5","context_window":400000}]
}`), 0o644); err != nil {
		t.Fatalf("write openai provider: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "openrouter.json"), []byte(`{
  "name": "openrouter",
  "api_type": "openai",
  "base_url": "https://openrouter.ai/api/v1",
  "models": [
    {"name":"anthropic/claude-sonnet-4","context_window":200000},
    {"name":"openai/gpt-5.5","context_window":1050000}
  ]
}`), 0o644); err != nil {
		t.Fatalf("write openrouter provider: %v", err)
	}

	initial := llmtest.New("initial")
	switched := llmtest.New("switched", okStep())
	env, _, errw, _ := fakeProviderEnv(t,
		[]string{"-config", cfgPath},
		initial,
		"/model\nopenrouter\nopenai/gpt-5.5\nhello\n/exit\n",
	)
	var opts []factory.Options
	env.newProvider = func(opt factory.Options) (llm.Provider, error) {
		opts = append(opts, opt)
		if len(opts) == 1 {
			return initial, nil
		}
		return switched, nil
	}

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	if len(opts) != 2 {
		t.Fatalf("provider factory calls = %d, want 2 (%+v)", len(opts), opts)
	}
	got := opts[1]
	if got.Provider != "openai" || got.ProviderName != "openrouter" || got.Model != "openai/gpt-5.5" ||
		got.BaseURL != "https://openrouter.ai/api/v1" || got.ContextWindow != 1_050_000 {
		t.Fatalf("interactive switch options = %+v", got)
	}
	if len(initial.Requests) != 0 {
		t.Fatalf("initial provider should receive no turns after switch, got %d", len(initial.Requests))
	}
	if len(switched.Requests) != 1 || switched.Requests[0].Model != "openai/gpt-5.5" {
		t.Fatalf("switched requests = %+v, want one openrouter-local request", switched.Requests)
	}
	stderr := errw.String()
	if !strings.Contains(stderr, "Providers 1-2 of 2") || !strings.Contains(stderr, "Models for openrouter") ||
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
		"-p", "-provider", "-model", "-base-url", "-system", "-system-override",
		"-no-env", "-resume", "-session", "-max-steps", "-default-context-window", "-context-window",
		"-reasoning-effort", "-v", "-tool-stream", "-q", "-quiet", "-log-level", "-no-color", "-config", "-setup", "-force", "-refresh-models", "-prompt",
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
		"sk-openrouter",
		"openai/gpt-5.5",
	}, "\n") + "\n"
	env, out, errw, getenv := fakeProviderEnv(t, []string{"--setup"}, fp, stdin)
	env.modelsDevCatalog = func(context.Context) (*modelsdev.Catalog, error) {
		return testModelsDevCatalog(t), nil
	}

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
		Provider             string   `json:"provider"`
		Model                string   `json:"model"`
		ProviderConfigs      []string `json:"provider_configs"`
		DefaultContextWindow int      `json:"default_context_window"`
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
	if mainCfg.DefaultContextWindow != llm.DefaultContextWindow {
		t.Fatalf("default context window = %d, want %d", mainCfg.DefaultContextWindow, llm.DefaultContextWindow)
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

func TestRunSetupUsesModelsDevDefaults(t *testing.T) {
	fp := llmtest.New("fake")
	stdin := strings.Join([]string{
		"openr",
		"sk-openrouter",
		"openai/gpt-5",
	}, "\n") + "\n"
	env, out, errw, getenv := fakeProviderEnv(t, []string{"--setup"}, fp, stdin)
	catalog := testModelsDevCatalog(t)
	env.modelsDevCatalog = func(context.Context) (*modelsdev.Catalog, error) {
		return catalog, nil
	}

	code := run(env)
	if code != ui.ExitOK {
		t.Fatalf("setup exit = %d, want 0; errw=%q", code, errw.String())
	}

	providerPath := filepath.Join(getenv("HOME"), ".config", "harness", "openrouter.json")
	if !strings.Contains(out.String(), "Using provider openrouter") || !strings.Contains(out.String(), "Using model openai/gpt-5.5") {
		t.Fatalf("setup output should show resolved provider/model, got %q", out.String())
	}
	var providerCfg struct {
		Name      string   `json:"name"`
		APIType   string   `json:"api_type"`
		BaseURL   string   `json:"base_url"`
		APIKey    string   `json:"api_key"`
		APIKeyEnv []string `json:"api_key_env"`
		Models    []struct {
			Name             string                `json:"name"`
			ContextWindow    int                   `json:"context_window"`
			Price            llm.Price             `json:"price"`
			Reasoning        bool                  `json:"reasoning"`
			ReasoningOptions []llm.ReasoningOption `json:"reasoning_options"`
		} `json:"models"`
	}
	data, err := os.ReadFile(providerPath)
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
	if len(providerCfg.APIKeyEnv) != 1 || providerCfg.APIKeyEnv[0] != "OPENROUTER_API_KEY" {
		t.Fatalf("api key env = %#v", providerCfg.APIKeyEnv)
	}
	if len(providerCfg.Models) != 1 {
		t.Fatalf("models = %#v", providerCfg.Models)
	}
	model := providerCfg.Models[0]
	if model.Name != "openai/gpt-5.5" || model.ContextWindow != 1_050_000 ||
		model.Price.Input != 5 || model.Price.Output != 30 || model.Price.CacheRead != 0.5 {
		t.Fatalf("generated model = %+v", model)
	}
	if !model.Reasoning || len(model.ReasoningOptions) != 1 || model.ReasoningOptions[0].Type != "effort" {
		t.Fatalf("generated reasoning metadata = reasoning:%v options:%+v", model.Reasoning, model.ReasoningOptions)
	}
}

func TestRunSetupAddsProviderToExistingDefaultConfig(t *testing.T) {
	fp := llmtest.New("fake")
	stdin := strings.Join([]string{
		"deepseek",
		"sk-deepseek",
		"deepseek-chat",
	}, "\n") + "\n"
	env, out, errw, getenv := fakeProviderEnv(t, []string{"--setup"}, fp, stdin)
	env.modelsDevCatalog = func(context.Context) (*modelsdev.Catalog, error) {
		return testModelsDevCatalog(t), nil
	}
	configDir := filepath.Join(getenv("HOME"), ".config", "harness")
	if err := os.MkdirAll(configDir, 0o700); err != nil {
		t.Fatalf("mkdir config dir: %v", err)
	}
	configPath := filepath.Join(configDir, "config.json")
	if err := os.WriteFile(configPath, []byte(`{
  "provider": "openai",
  "model": "gpt-5.5",
  "default_context_window": 123,
  "provider_configs": ["openai.json"],
  "system": "keep me"
}`), 0o600); err != nil {
		t.Fatalf("write existing config: %v", err)
	}

	code := run(env)
	if code != ui.ExitOK {
		t.Fatalf("setup with existing config exit = %d, want 0; errw=%q", code, errw.String())
	}
	providerPath := filepath.Join(configDir, "deepseek.json")
	if !strings.Contains(out.String(), "Updated "+configPath) || !strings.Contains(out.String(), "Wrote "+providerPath) {
		t.Fatalf("setup output should name updated/written files, got %q", out.String())
	}

	var mainCfg struct {
		Provider             string   `json:"provider"`
		Model                string   `json:"model"`
		DefaultContextWindow int      `json:"default_context_window"`
		ProviderConfigs      []string `json:"provider_configs"`
		System               string   `json:"system"`
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read updated config: %v", err)
	}
	if err := json.Unmarshal(data, &mainCfg); err != nil {
		t.Fatalf("decode updated config: %v", err)
	}
	if mainCfg.Provider != "openai" || mainCfg.Model != "gpt-5.5" || mainCfg.DefaultContextWindow != 123 || mainCfg.System != "keep me" {
		t.Fatalf("existing config fields were overwritten: %+v", mainCfg)
	}
	if got, want := strings.Join(mainCfg.ProviderConfigs, ","), "openai.json,deepseek.json"; got != want {
		t.Fatalf("provider configs = %q, want %q", got, want)
	}

	var providerCfg struct {
		Name   string                  `json:"name"`
		APIKey string                  `json:"api_key"`
		Models []struct{ Name string } `json:"models"`
	}
	data, err = os.ReadFile(providerPath)
	if err != nil {
		t.Fatalf("read provider config: %v", err)
	}
	if err := json.Unmarshal(data, &providerCfg); err != nil {
		t.Fatalf("decode provider config: %v", err)
	}
	if providerCfg.Name != "deepseek" || providerCfg.APIKey != "sk-deepseek" ||
		len(providerCfg.Models) != 1 || providerCfg.Models[0].Name != "deepseek-chat" {
		t.Fatalf("provider config = %+v", providerCfg)
	}
}

func TestRunSetupRefusesExistingProviderConfigWithoutForce(t *testing.T) {
	fp := llmtest.New("fake")
	env, _, errw, getenv := fakeProviderEnv(t, []string{"--setup"}, fp, "deepseek\n")
	configDir := filepath.Join(getenv("HOME"), ".config", "harness")
	if err := os.MkdirAll(configDir, 0o700); err != nil {
		t.Fatalf("mkdir config dir: %v", err)
	}
	configPath := filepath.Join(configDir, "config.json")
	providerPath := filepath.Join(configDir, "deepseek.json")
	if err := os.WriteFile(configPath, []byte(`{"provider_configs":["openai.json"]}`), 0o600); err != nil {
		t.Fatalf("write existing config: %v", err)
	}
	if err := os.WriteFile(providerPath, []byte(`{"name":"old"}`), 0o600); err != nil {
		t.Fatalf("write existing provider config: %v", err)
	}

	code := run(env)
	if code != ui.ExitUsage {
		t.Fatalf("setup with existing provider config exit = %d, want 2", code)
	}
	if !strings.Contains(errw.String(), "already exists") || !strings.Contains(errw.String(), providerPath) {
		t.Fatalf("stderr should name existing provider config, got %q", errw.String())
	}
	data, err := os.ReadFile(providerPath)
	if err != nil {
		t.Fatalf("read provider config: %v", err)
	}
	if string(data) != `{"name":"old"}` {
		t.Fatalf("provider config should not be overwritten without --force, got %q", string(data))
	}
}

func TestRunSetupForceOverwritesProviderConfigAndDefaults(t *testing.T) {
	fp := llmtest.New("fake")
	stdin := strings.Join([]string{
		"deepseek",
		"sk-new",
		"deepseek-chat",
	}, "\n") + "\n"
	env, out, errw, getenv := fakeProviderEnv(t, []string{"--setup", "--force"}, fp, stdin)
	env.modelsDevCatalog = func(context.Context) (*modelsdev.Catalog, error) {
		return testModelsDevCatalog(t), nil
	}
	configDir := filepath.Join(getenv("HOME"), ".config", "harness")
	if err := os.MkdirAll(configDir, 0o700); err != nil {
		t.Fatalf("mkdir config dir: %v", err)
	}
	configPath := filepath.Join(configDir, "config.json")
	providerPath := filepath.Join(configDir, "deepseek.json")
	if err := os.WriteFile(configPath, []byte(`{
  "provider": "openai",
  "model": "gpt-5.5",
  "provider_configs": ["openai.json", "deepseek.json"]
}`), 0o600); err != nil {
		t.Fatalf("write existing config: %v", err)
	}
	if err := os.WriteFile(providerPath, []byte(`{"name":"old","api_key":"sk-old"}`), 0o600); err != nil {
		t.Fatalf("write existing provider config: %v", err)
	}

	code := run(env)
	if code != ui.ExitOK {
		t.Fatalf("setup --force exit = %d, want 0; errw=%q", code, errw.String())
	}
	if !strings.Contains(out.String(), "Updated "+configPath) || !strings.Contains(out.String(), "Overwrote "+providerPath) {
		t.Fatalf("setup output should name overwritten provider, got %q", out.String())
	}

	var mainCfg struct {
		Provider             string   `json:"provider"`
		Model                string   `json:"model"`
		DefaultContextWindow int      `json:"default_context_window"`
		ProviderConfigs      []string `json:"provider_configs"`
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read updated config: %v", err)
	}
	if err := json.Unmarshal(data, &mainCfg); err != nil {
		t.Fatalf("decode updated config: %v", err)
	}
	if mainCfg.Provider != "deepseek" || mainCfg.Model != "deepseek-chat" || mainCfg.DefaultContextWindow != llm.DefaultContextWindow {
		t.Fatalf("force should update defaults, got %+v", mainCfg)
	}
	if got, want := strings.Join(mainCfg.ProviderConfigs, ","), "openai.json,deepseek.json"; got != want {
		t.Fatalf("provider configs = %q, want %q", got, want)
	}

	var providerCfg struct {
		Name   string `json:"name"`
		APIKey string `json:"api_key"`
	}
	data, err = os.ReadFile(providerPath)
	if err != nil {
		t.Fatalf("read provider config: %v", err)
	}
	if err := json.Unmarshal(data, &providerCfg); err != nil {
		t.Fatalf("decode provider config: %v", err)
	}
	if providerCfg.Name != "deepseek" || providerCfg.APIKey != "sk-new" {
		t.Fatalf("provider config was not overwritten: %+v", providerCfg)
	}
}

func TestRunRefreshModelsUpdatesConfiguredProviders(t *testing.T) {
	fp := llmtest.New("fake")
	env, out, errw, getenv := fakeProviderEnv(t, []string{"--refresh-models"}, fp, "")
	env.modelsDevCatalog = func(context.Context) (*modelsdev.Catalog, error) {
		return testModelsDevCatalog(t), nil
	}
	configDir := filepath.Join(getenv("HOME"), ".config", "harness")
	if err := os.MkdirAll(configDir, 0o700); err != nil {
		t.Fatalf("mkdir config dir: %v", err)
	}
	configPath := filepath.Join(configDir, "config.json")
	providerPath := filepath.Join(configDir, "openrouter.json")
	if err := os.WriteFile(configPath, []byte(`{
  "provider": "openrouter",
  "model": "openai/gpt-5.5",
  "provider_configs": ["openrouter.json"]
}`), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := os.WriteFile(providerPath, []byte(`{
  "name": "openrouter",
  "api_type": "openai",
  "base_url": "https://old.example/v1",
  "api_key": "sk-keep",
  "models": [
    {"name":"old-model","context_window":1}
  ]
}`), 0o600); err != nil {
		t.Fatalf("write provider: %v", err)
	}

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("refresh exit = %d, want 0; errw=%q", code, errw.String())
	}
	if !strings.Contains(out.String(), "Updated "+providerPath) {
		t.Fatalf("refresh output should name updated provider, got %q", out.String())
	}
	var providerCfg struct {
		Name      string   `json:"name"`
		BaseURL   string   `json:"base_url"`
		APIKey    string   `json:"api_key"`
		APIKeyEnv []string `json:"api_key_env"`
		Models    []struct {
			Name          string    `json:"name"`
			ContextWindow int       `json:"context_window"`
			Price         llm.Price `json:"price"`
		} `json:"models"`
	}
	data, err := os.ReadFile(providerPath)
	if err != nil {
		t.Fatalf("read refreshed provider: %v", err)
	}
	if err := json.Unmarshal(data, &providerCfg); err != nil {
		t.Fatalf("decode refreshed provider: %v", err)
	}
	if providerCfg.Name != "openrouter" || providerCfg.BaseURL != "https://openrouter.ai/api/v1" || providerCfg.APIKey != "sk-keep" {
		t.Fatalf("refreshed provider = %+v", providerCfg)
	}
	if len(providerCfg.APIKeyEnv) != 1 || providerCfg.APIKeyEnv[0] != "OPENROUTER_API_KEY" {
		t.Fatalf("api key env = %#v", providerCfg.APIKeyEnv)
	}
	if len(providerCfg.Models) != 1 || providerCfg.Models[0].Name != "openai/gpt-5.5" ||
		providerCfg.Models[0].ContextWindow != 1_050_000 || providerCfg.Models[0].Price.Input != 5 {
		t.Fatalf("refreshed models = %#v", providerCfg.Models)
	}
}

func TestRunRefreshModelsErrorsWhenModelsDevUnavailable(t *testing.T) {
	fp := llmtest.New("fake")
	env, _, errw, getenv := fakeProviderEnv(t, []string{"--refresh-models"}, fp, "")
	env.modelsDevCatalog = func(context.Context) (*modelsdev.Catalog, error) {
		return nil, errors.New("offline")
	}
	configDir := filepath.Join(getenv("HOME"), ".config", "harness")
	if err := os.MkdirAll(configDir, 0o700); err != nil {
		t.Fatalf("mkdir config dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(configDir, "config.json"), []byte(`{"provider_configs":["openrouter.json"]}`), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(configDir, "openrouter.json"), []byte(`{"name":"openrouter"}`), 0o600); err != nil {
		t.Fatalf("write provider: %v", err)
	}

	if code := run(env); code != ui.ExitUsage {
		t.Fatalf("refresh exit = %d, want usage error", code)
	}
	if !strings.Contains(errw.String(), "offline") {
		t.Fatalf("stderr should include fetch error, got %q", errw.String())
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
	  "default_context_window": 512000,
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

	fp := llmtest.New("fake", okStep())
	env, _, errw, _ := fakeProviderEnv(t, []string{"-config", cfgPath, "-p", "hi"}, fp, "")
	got := captureOptions(&env, fp)

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

func TestRunModelsDevOpenAIDefaultsToResponses(t *testing.T) {
	fp := llmtest.New("fake", okStep())
	env, _, errw, _ := fakeProviderEnv(t, []string{"-model", "gpt-5.5", "-p", "hi"}, fp, "")
	env.modelsDevCatalog = func(context.Context) (*modelsdev.Catalog, error) {
		return testModelsDevCatalog(t), nil
	}
	got := captureOptions(&env, fp)

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit = %d, want 0; errw=%q", code, errw.String())
	}
	if got.Provider != "responses" || got.ProviderName != "openai" {
		t.Fatalf("factory options = %+v, want openai provider name on responses api type", got)
	}
}

func TestRunModelsDevOpenAICompatibleBaseURLStaysChatCompletions(t *testing.T) {
	fp := llmtest.New("fake", okStep())
	env, _, errw, _ := fakeProviderEnv(t, []string{"-model", "gpt-5.5", "-base-url", "https://proxy.example/v1", "-p", "hi"}, fp, "")
	env.modelsDevCatalog = func(context.Context) (*modelsdev.Catalog, error) {
		return testModelsDevCatalog(t), nil
	}
	got := captureOptions(&env, fp)

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit = %d, want 0; errw=%q", code, errw.String())
	}
	if got.Provider != "openai" {
		t.Fatalf("factory provider = %q, want chat-completions api type for custom base URL", got.Provider)
	}
}

func TestRunExplicitResponsesUsesOpenAIModelsDevMetadata(t *testing.T) {
	fp := llmtest.New("fake", okStep())
	env, _, errw, _ := fakeProviderEnv(t, []string{"-provider", "responses", "-model", "gpt-5.5", "-p", "hi"}, fp, "")
	env.modelsDevCatalog = func(context.Context) (*modelsdev.Catalog, error) {
		return testModelsDevCatalog(t), nil
	}
	got := captureOptions(&env, fp)

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit = %d, want 0; errw=%q", code, errw.String())
	}
	if got.Provider != "responses" {
		t.Fatalf("factory provider = %q, want responses", got.Provider)
	}
	if got.ContextWindow != 1_050_000 {
		t.Fatalf("context window = %d, want OpenAI models.dev value", got.ContextWindow)
	}
}

func TestRunProviderConfigCanSelectResponses(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	providerPath := filepath.Join(dir, "openai.json")
	if err := os.WriteFile(cfgPath, []byte(`{
	  "provider": "openai",
	  "model": "gpt-5.5",
	  "provider_configs": ["openai.json"]
	}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := os.WriteFile(providerPath, []byte(`{
  "name": "openai",
  "api_type": "responses",
  "base_url": "https://api.openai.com/v1",
  "models": [
    {"name":"gpt-5.5","context_window":1050000,"price":{"input":5,"output":30}}
  ]
}`), 0o644); err != nil {
		t.Fatalf("write provider config: %v", err)
	}

	fp := llmtest.New("fake", okStep())
	env, _, errw, _ := fakeProviderEnv(t, []string{"-config", cfgPath, "-p", "hi"}, fp, "")
	got := captureOptions(&env, fp)

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit = %d, want 0; errw=%q", code, errw.String())
	}
	if got.Provider != "responses" || got.ProviderName != "openai" {
		t.Fatalf("factory options = %+v, want responses api type", got)
	}
}

func TestRunReasoningEffortUsesOpenRouterModeAndRequestConfig(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	providerPath := filepath.Join(dir, "openrouter.json")
	if err := os.WriteFile(cfgPath, []byte(`{
	  "provider": "openrouter",
	  "model": "openai/gpt-5.5",
	  "provider_configs": ["openrouter.json"],
	  "reasoning_effort": "medium"
	}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := os.WriteFile(providerPath, []byte(`{
  "name": "openrouter",
  "api_type": "openai",
  "base_url": "https://openrouter.ai/api/v1",
  "models": [
    {"name":"openai/gpt-5.5","context_window":1050000,"reasoning":true,"reasoning_options":[{"type":"effort","values":["low","medium","high"]}]}
  ]
}`), 0o644); err != nil {
		t.Fatalf("write provider config: %v", err)
	}

	fp := llmtest.New("fake", okStep())
	env, _, errw, _ := fakeProviderEnv(t, []string{"-config", cfgPath, "-p", "hi"}, fp, "")
	got := captureOptions(&env, fp)

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit = %d, want 0; errw=%q", code, errw.String())
	}
	if got.Provider != "openai" || got.ProviderName != "openrouter" || got.ReasoningMode != "openrouter" {
		t.Fatalf("factory options = %+v, want openrouter reasoning mode over openai dialect", got)
	}
	if len(fp.Requests) != 1 || fp.Requests[0].Reasoning.Effort != "medium" {
		t.Fatalf("request reasoning = %+v", fp.Requests)
	}
}

func TestRunReasoningEffortRejectedWhenModelsDevSaysUnsupported(t *testing.T) {
	fp := llmtest.New("fake")
	env, _, errw, _ := fakeProviderEnv(t, []string{"-model", "gpt-4o", "-reasoning-effort", "high", "-p", "hi"}, fp, "")
	catalog, err := modelsdev.Decode(strings.NewReader(`{
  "openai": {
    "id": "openai",
    "name": "OpenAI",
    "models": {
      "gpt-4o": {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "reasoning": false,
        "limit": {"context": 128000},
        "cost": {"input": 2.5, "output": 10}
      }
    }
  }
}`))
	if err != nil {
		t.Fatalf("decode catalog: %v", err)
	}
	env.modelsDevCatalog = func(context.Context) (*modelsdev.Catalog, error) {
		return catalog, nil
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

func TestRunReasoningEffortRejectedWhenValueUnsupported(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	providerPath := filepath.Join(dir, "openai.json")
	if err := os.WriteFile(cfgPath, []byte(`{
	  "provider": "openai",
	  "model": "gpt-5-pro",
	  "provider_configs": ["openai.json"],
	  "reasoning_effort": "xhigh"
	}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := os.WriteFile(providerPath, []byte(`{
  "name": "openai",
  "api_type": "openai",
  "base_url": "https://api.openai.com/v1",
  "models": [
    {"name":"gpt-5-pro","context_window":400000,"reasoning":true,"reasoning_options":[{"type":"effort","values":["high"]}]}
  ]
}`), 0o644); err != nil {
		t.Fatalf("write provider config: %v", err)
	}

	fp := llmtest.New("fake")
	env, _, errw, _ := fakeProviderEnv(t, []string{"-config", cfgPath, "-p", "hi"}, fp, "")
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

func TestRunProviderConfigUsesProviderSpecificAPIKeyEnv(t *testing.T) {
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
  "api_key_env": ["OPENROUTER_API_KEY"],
  "models": [
    {"name":"openai/gpt-5.1","context_window":1000000,"price":{"input":2,"output":8}}
  ]
}`), 0o644); err != nil {
		t.Fatalf("write provider config: %v", err)
	}

	fp := llmtest.New("fake", okStep())
	env, _, errw, _ := fakeProviderEnv(t, []string{"-config", cfgPath, "-p", "hi"}, fp, "")
	baseGetenv := env.getenv
	env.getenv = func(k string) string {
		if k == "OPENROUTER_API_KEY" {
			return "sk-openrouter-env"
		}
		if k == "OPENAI_API_KEY" {
			return "sk-openai-env"
		}
		return baseGetenv(k)
	}
	got := captureOptions(&env, fp)

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit = %d, want 0; errw=%q", code, errw.String())
	}
	if got.APIKey != "sk-openrouter-env" {
		t.Fatalf("factory API key = %q, want provider-specific env key", got.APIKey)
	}
}

func TestRunEnrichesUnknownModelFromModelsDev(t *testing.T) {
	fp := llmtest.New("fake", okStepWithUsage(1_000_000, 1_000_000))
	env, _, errw, _ := fakeProviderEnv(t, []string{"-model", "gpt-5.5", "-p", "hi"}, fp, "")
	catalog := testModelsDevCatalog(t)
	env.modelsDevCatalog = func(context.Context) (*modelsdev.Catalog, error) {
		return catalog, nil
	}
	got := captureOptions(&env, fp)

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit = %d, want 0; errw=%q", code, errw.String())
	}
	if got.ContextWindow != 1_050_000 {
		t.Fatalf("factory context window = %d, want models.dev context", got.ContextWindow)
	}
	if !strings.Contains(errw.String(), "$35.000") {
		t.Fatalf("usage line should include models.dev cost, errw=%q", errw.String())
	}
}

func TestRunDefaultContextWindowUsedForUnknownModel(t *testing.T) {
	fp := llmtest.New("fake", okStep())
	env, _, errw, _ := fakeProviderEnv(t, []string{
		"-model", "local-model",
		"-base-url", "http://localhost:11434/v1",
		"-default-context-window", "512000",
		"-p", "hi",
	}, fp, "")
	got := captureOptions(&env, fp)

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit = %d, want 0; errw=%q", code, errw.String())
	}
	if got.ContextWindow != 512_000 {
		t.Fatalf("factory context window = %d, want default fallback 512000", got.ContextWindow)
	}
}

func TestRunSkipsModelsDevForLocalBaseURL(t *testing.T) {
	fp := llmtest.New("fake", okStep())
	env, _, errw, _ := fakeProviderEnv(t, []string{
		"-model", "local-model",
		"-base-url", "http://localhost:11434/v1",
		"-p", "hi",
	}, fp, "")
	env.modelsDevCatalog = func(context.Context) (*modelsdev.Catalog, error) {
		t.Fatal("models.dev should not be queried for localhost base URLs")
		return nil, nil
	}

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit = %d, want 0; errw=%q", code, errw.String())
	}
}

func TestRunContextWindowOverrideStillWins(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	providerPath := filepath.Join(dir, "openrouter.json")
	if err := os.WriteFile(cfgPath, []byte(`{
	  "provider": "openrouter",
	  "model": "openai/gpt-5.1",
	  "default_context_window": 512000,
	  "context_window": 64000,
	  "provider_configs": ["openrouter.json"]
	}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := os.WriteFile(providerPath, []byte(`{
	  "name": "openrouter",
	  "api_type": "openai",
	  "base_url": "https://openrouter.ai/api/v1",
	  "models": [
	    {"name":"openai/gpt-5.1","context_window":1000000}
	  ]
	}`), 0o644); err != nil {
		t.Fatalf("write provider config: %v", err)
	}

	fp := llmtest.New("fake", okStep())
	env, _, errw, _ := fakeProviderEnv(t, []string{"-config", cfgPath, "-p", "hi"}, fp, "")
	got := captureOptions(&env, fp)

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit = %d, want 0; errw=%q", code, errw.String())
	}
	if got.ContextWindow != 64_000 {
		t.Fatalf("factory context window = %d, want explicit override 64000", got.ContextWindow)
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

	fp := llmtest.New("fake", okStep())
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
	// A plain (non-API, non-cancel) provider error is retryable, so it must
	// recur through the whole per-step budget (1 + 2 retries) to surface as the
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

// Default (auto) mode advertises the default tool set plus delegate and carries
// no mode section.
func TestRunDefaultModeTools(t *testing.T) {
	fp := llmtest.New("fake", okStepWithUsage(1, 1))
	env, _, errw, _ := fakeProviderEnv(t, []string{"-model", "claude-opus-4-8", "-p", "hi"}, fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	want := expectedDefaultToolNames()
	if got := toolNames(fp.Requests[0]); !slices.Equal(got, want) {
		t.Errorf("default mode tools = %v, want %v", got, want)
	}
	if strings.Contains(fp.Requests[0].System, "plan mode") || strings.Contains(fp.Requests[0].System, "independent mode") {
		t.Errorf("default mode should carry no mode section; system=%q", fp.Requests[0].System)
	}
}

func TestRunDelegateToolUsesReadOnlyChildAgent(t *testing.T) {
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
	if slices.Contains(childTools, "delegate") || slices.Contains(childTools, "write_tmp_file") || slices.Contains(childTools, "edit") {
		t.Fatalf("child request advertised non-read-only or recursive tools: %v", childTools)
	}
	if got := fp.Requests[1].Messages[0].Content[0].Text; got != "inspect only" {
		t.Fatalf("child task = %q", got)
	}
	if !strings.Contains(errw.String(), "[delegate]") {
		t.Fatalf("delegate tool result was not rendered: %q", errw.String())
	}
	if !strings.Contains(errw.String(), "60 in / 13 out") {
		t.Fatalf("turn usage should include parent and child model calls, stderr=%q", errw.String())
	}
}

func TestRunDelegateUsesSwitchedModelAndMode(t *testing.T) {
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
		"/model gpt-5.5\n/mode plan\nhi\n/exit\n",
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
	if !strings.Contains(child.System, "plan mode") {
		t.Fatalf("delegate child system should include switched mode prompt, system=%q", child.System)
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

// Plan mode advertises only its read-only tool set and includes its prompt.
func TestRunPlanModeRestrictsToolsAndAddsPrompt(t *testing.T) {
	fp := llmtest.New("fake", okStepWithUsage(1, 1))
	env, _, errw, _ := fakeProviderEnv(t, []string{"-model", "claude-opus-4-8", "-mode", "plan", "-p", "hi"}, fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("exit code = %d, want 0; errw=%q", code, errw.String())
	}
	want := expectedPlanToolNames()
	if got := toolNames(fp.Requests[0]); !slices.Equal(got, want) {
		t.Errorf("plan mode tools = %v, want %v", got, want)
	}
	if !strings.Contains(fp.Requests[0].System, "plan mode") {
		t.Errorf("plan mode system prompt should include the plan section; system=%q", fp.Requests[0].System)
	}
}

// An unknown mode is a startup usage error that lists the available modes.
func TestRunUnknownModeIsUsageError(t *testing.T) {
	fp := llmtest.New("fake")
	env, _, errw, _ := fakeProviderEnv(t, []string{"-model", "claude-opus-4-8", "-mode", "bogus", "-p", "hi"}, fp, "")

	if code := run(env); code != ui.ExitUsage {
		t.Fatalf("exit code = %d, want ExitUsage; errw=%q", code, errw.String())
	}
	got := errw.String()
	if !strings.Contains(got, "bogus") || !strings.Contains(got, "auto") || !strings.Contains(got, "plan") {
		t.Errorf("error should name the bad mode and list valid ones, errw=%q", got)
	}
	if len(fp.Requests) != 0 {
		t.Errorf("no turn should run for an unknown mode, got %d requests", len(fp.Requests))
	}
}

// A config mode entry overriding only the prompt keeps the built-in tool list.
func TestRunConfigModePromptOverrideKeepsTools(t *testing.T) {
	fp := llmtest.New("fake", okStepWithUsage(1, 1))
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	if err := os.WriteFile(cfgPath, []byte(`{"mode":"plan","modes":{"plan":{"prompt":"CUSTOM PLAN GUIDANCE"}}}`), 0644); err != nil {
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

// /mode in the REPL switches the advertised tool set on the next turn.
func TestRunREPLModeCommandSwitchesTools(t *testing.T) {
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
	if !strings.Contains(errw.String(), "mode switched: plan") {
		t.Errorf("switch should be acknowledged, errw=%q", errw.String())
	}
}

// A resumed session restores its run mode (and thus its restricted tool set)
// when no -mode flag overrides it.
func TestRunResumeRestoresMode(t *testing.T) {
	dir := t.TempDir()
	sessPath := filepath.Join(dir, "prior")
	prior := session.Session{
		Version:  session.Version,
		Provider: "anthropic",
		Model:    "claude-opus-4-8",
		System:   "you are a test",
		Mode:     "plan",
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
