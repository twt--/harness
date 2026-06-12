package config

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// noEnv is a getenv that returns "" for everything: the default environment for
// tests that exercise flag/file/default precedence without env interference.
func noEnv(string) string { return "" }

// envFrom builds a getenv closure backed by a map.
func envFrom(m map[string]string) func(string) string {
	return func(k string) string { return m[k] }
}

// writeConfig writes a config file in a temp dir and returns its path.
func writeConfig(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	return p
}

func TestModelPrecedenceFlagBeatsEnvBeatsFileBeatsDefault(t *testing.T) {
	cfgPath := writeConfig(t, `{"model":"file-model"}`)
	env := envFrom(map[string]string{"HARNESS_MODEL": "env-model"})

	// Flag wins over env and file.
	c, err := Load([]string{"-model", "flag-model"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Model != "flag-model" {
		t.Fatalf("flag precedence: got model %q, want flag-model", c.Model)
	}

	// Env wins over file when no flag.
	c, err = Load(nil, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Model != "env-model" {
		t.Fatalf("env precedence: got model %q, want env-model", c.Model)
	}

	// File wins over default when no flag and no env.
	c, err = Load(nil, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Model != "file-model" {
		t.Fatalf("file precedence: got model %q, want file-model", c.Model)
	}
}

// TestLoadSplitsProviderModel pins SplitProviderModel's contract at the Load
// call site, including the whitespace trimming the consolidated helper adopted
// from the REPL-side copy (regression: the two pre-merge copies had drifted).
func TestLoadSplitsProviderModel(t *testing.T) {
	cases := []struct {
		name         string
		model        string
		wantProvider string
		wantModel    string
	}{
		{name: "plain split", model: "anthropic:claude-opus-4-8", wantProvider: "anthropic", wantModel: "claude-opus-4-8"},
		{name: "padded value is trimmed before split", model: " anthropic:claude-opus-4-8 ", wantProvider: "anthropic", wantModel: "claude-opus-4-8"},
		{name: "colon inside model id is not a provider prefix", model: "org/model:fp16", wantProvider: "", wantModel: "org/model:fp16"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, err := Load([]string{"-model", tc.model}, noEnv, "")
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			if c.Provider != tc.wantProvider || c.Model != tc.wantModel {
				t.Fatalf("got provider=%q model=%q, want provider=%q model=%q", c.Provider, c.Model, tc.wantProvider, tc.wantModel)
			}
		})
	}
}

func TestProviderPrecedenceFlagBeatsEnvBeatsFile(t *testing.T) {
	cfgPath := writeConfig(t, `{"provider":"openai"}`)
	env := envFrom(map[string]string{"HARNESS_PROVIDER": "anthropic"})

	c, err := Load([]string{"-provider", "openai"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Provider != "openai" {
		t.Fatalf("flag precedence: got provider %q", c.Provider)
	}

	c, err = Load(nil, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Provider != "anthropic" {
		t.Fatalf("env precedence: got provider %q", c.Provider)
	}

	c, err = Load(nil, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Provider != "openai" {
		t.Fatalf("file precedence: got provider %q", c.Provider)
	}
}

func TestModelProxyURLPrecedenceFlagBeatsEnvBeatsFile(t *testing.T) {
	cfgPath := writeConfig(t, `{"model_proxy_url":"http://file.example"}`)
	env := envFrom(map[string]string{"HARNESS_MODEL_PROXY_URL": "http://env.example"})

	c, err := Load([]string{"-model-proxy-url", "http://flag.example"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.ModelProxyURL != "http://flag.example" {
		t.Fatalf("flag precedence: got model proxy URL %q", c.ModelProxyURL)
	}

	c, err = Load(nil, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.ModelProxyURL != "http://env.example" {
		t.Fatalf("env precedence: got model proxy URL %q", c.ModelProxyURL)
	}

	c, err = Load(nil, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.ModelProxyURL != "http://file.example" {
		t.Fatalf("file precedence: got model proxy URL %q", c.ModelProxyURL)
	}
}

func TestExplicitProviderIsPreserved(t *testing.T) {
	c, err := Load([]string{"-model", "claude-opus-4-8", "-provider", "openai"}, noEnv, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Provider != "openai" {
		t.Fatalf("provider %q, want openai (explicit overrides inference)", c.Provider)
	}
}

// HARNESS_* env mapping covers the user-facing flags.
func TestHarnessEnvMapping(t *testing.T) {
	env := envFrom(map[string]string{
		"HARNESS_MODEL":                  "env-model",
		"HARNESS_MODEL_PROXY_URL":        "http://proxy.example",
		"HARNESS_MAX_STEPS":              "12",
		"HARNESS_DEFAULT_CONTEXT_WINDOW": "512000",
		"HARNESS_CONTEXT_WINDOW":         "256000",
		"HARNESS_REASONING_EFFORT":       "HIGH",
		"HARNESS_SYSTEM":                 "env system note",
		"HARNESS_NO_ENV":                 "true",
		"HARNESS_NO_COLOR":               "true",
		"HARNESS_VERBOSE":                "true",
		"HARNESS_TOOL_STREAM":            "false",
		"HARNESS_PROMPT":                 "env> ",
		"LOG_LEVEL":                      "WARN",
	})
	c, err := Load(nil, env, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Model != "env-model" {
		t.Fatalf("model %q", c.Model)
	}
	if c.ModelProxyURL != "http://proxy.example" {
		t.Fatalf("model proxy URL %q", c.ModelProxyURL)
	}
	if c.MaxSteps != 12 {
		t.Fatalf("max-steps %d, want 12", c.MaxSteps)
	}
	if c.DefaultContextWindow != 512000 {
		t.Fatalf("default-context-window %d, want 512000", c.DefaultContextWindow)
	}
	if c.ContextWindow != 256000 {
		t.Fatalf("context-window %d, want 256000", c.ContextWindow)
	}
	if c.ReasoningEffort != "high" {
		t.Fatalf("reasoning effort %q, want high", c.ReasoningEffort)
	}
	if c.System != "env system note" {
		t.Fatalf("system %q", c.System)
	}
	if !c.NoEnv {
		t.Fatalf("no-env false, want true")
	}
	if !c.NoColor {
		t.Fatalf("no-color false, want true")
	}
	if !c.Verbose {
		t.Fatalf("verbose false, want true")
	}
	if c.ToolStream {
		t.Fatalf("tool-stream true, want false")
	}
	if c.LogLevel != "warn" {
		t.Fatalf("log level %q, want warn", c.LogLevel)
	}
	if c.ReplPrompt != "env> " {
		t.Fatalf("repl prompt %q, want env> ", c.ReplPrompt)
	}
}

func TestReplPromptPrecedence(t *testing.T) {
	// Default is "> ".
	c, err := Load([]string{"-model", "gpt-5.5"}, noEnv, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.ReplPrompt != "> " {
		t.Fatalf("default repl prompt %q, want %q", c.ReplPrompt, "> ")
	}

	// File overrides default.
	cfgPath := writeConfig(t, `{"prompt":"$ "}`)
	c, err = Load([]string{"-model", "gpt-5.5"}, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.ReplPrompt != "$ " {
		t.Fatalf("file repl prompt %q, want %q", c.ReplPrompt, "$ ")
	}

	// Env overrides file.
	env := envFrom(map[string]string{"HARNESS_PROMPT": "# "})
	c, err = Load([]string{"-model", "gpt-5.5"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.ReplPrompt != "# " {
		t.Fatalf("env repl prompt %q, want %q", c.ReplPrompt, "# ")
	}

	// Flag overrides all.
	c, err = Load([]string{"-model", "gpt-5.5", "-prompt", ">>> "}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.ReplPrompt != ">>> " {
		t.Fatalf("flag repl prompt %q, want %q", c.ReplPrompt, ">>> ")
	}
}

func TestToolStreamPrecedence(t *testing.T) {
	c, err := Load([]string{"-model", "gpt-5.5"}, noEnv, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !c.ToolStream {
		t.Fatalf("default tool-stream false, want true")
	}

	cfgPath := writeConfig(t, `{"tool_stream":false}`)
	c, err = Load([]string{"-model", "gpt-5.5"}, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load file: %v", err)
	}
	if c.ToolStream {
		t.Fatalf("file tool-stream true, want false")
	}

	env := envFrom(map[string]string{"HARNESS_TOOL_STREAM": "true"})
	c, err = Load([]string{"-model", "gpt-5.5"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load env: %v", err)
	}
	if !c.ToolStream {
		t.Fatalf("env tool-stream false, want true")
	}

	c, err = Load([]string{"-model", "gpt-5.5", "-tool-stream=false"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load flag: %v", err)
	}
	if c.ToolStream {
		t.Fatalf("flag tool-stream true, want false")
	}
}

// NO_COLOR (the de-facto standard env var) disables color independent of HARNESS_*.
func TestNoColorStandardEnv(t *testing.T) {
	env := envFrom(map[string]string{"NO_COLOR": "1"})
	c, err := Load([]string{"-model", "gpt-5.5"}, env, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !c.NoColor {
		t.Fatalf("NO_COLOR did not disable color")
	}
}

func TestMaxStepsDefault(t *testing.T) {
	c, err := Load([]string{"-model", "gpt-5.5"}, noEnv, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.MaxSteps != 50 {
		t.Fatalf("default max-steps %d, want 50", c.MaxSteps)
	}
}

func TestDefaultContextWindowDefault(t *testing.T) {
	c, err := Load([]string{"-model", "gpt-5.5"}, noEnv, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.DefaultContextWindow != 256_000 {
		t.Fatalf("default context window %d, want 256000", c.DefaultContextWindow)
	}
}

func TestDefaultContextWindowPrecedenceFlagBeatsEnvBeatsFile(t *testing.T) {
	cfgPath := writeConfig(t, `{"default_context_window":300000}`)
	env := envFrom(map[string]string{"HARNESS_DEFAULT_CONTEXT_WINDOW": "400000"})

	c, err := Load([]string{"-model", "gpt-5.5", "-default-context-window", "500000"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.DefaultContextWindow != 500000 {
		t.Fatalf("flag precedence: got default context window %d, want 500000", c.DefaultContextWindow)
	}

	c, err = Load([]string{"-model", "gpt-5.5"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.DefaultContextWindow != 400000 {
		t.Fatalf("env precedence: got default context window %d, want 400000", c.DefaultContextWindow)
	}

	c, err = Load([]string{"-model", "gpt-5.5"}, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.DefaultContextWindow != 300000 {
		t.Fatalf("file precedence: got default context window %d, want 300000", c.DefaultContextWindow)
	}
}

func TestReasoningEffortPrecedenceFlagBeatsEnvBeatsFile(t *testing.T) {
	cfgPath := writeConfig(t, `{"reasoning_effort":"low"}`)
	env := envFrom(map[string]string{"HARNESS_REASONING_EFFORT": "medium"})

	c, err := Load([]string{"-model", "gpt-5.5", "-reasoning-effort", "HIGH"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.ReasoningEffort != "high" {
		t.Fatalf("flag precedence: got reasoning effort %q, want high", c.ReasoningEffort)
	}

	c, err = Load([]string{"-model", "gpt-5.5"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.ReasoningEffort != "medium" {
		t.Fatalf("env precedence: got reasoning effort %q, want medium", c.ReasoningEffort)
	}

	c, err = Load([]string{"-model", "gpt-5.5"}, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.ReasoningEffort != "low" {
		t.Fatalf("file precedence: got reasoning effort %q, want low", c.ReasoningEffort)
	}
}

func TestMaxStepsFlagBeatsFile(t *testing.T) {
	cfgPath := writeConfig(t, `{"max_steps":7}`)
	c, err := Load([]string{"-max-steps", "9"}, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.MaxSteps != 9 {
		t.Fatalf("max-steps %d, want 9 (flag beats file)", c.MaxSteps)
	}

	c, err = Load(nil, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.MaxSteps != 7 {
		t.Fatalf("max-steps %d, want 7 (file beats default)", c.MaxSteps)
	}
}

func TestDelegateMaxStepsConfigOnly(t *testing.T) {
	c, err := Load(nil, noEnv, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.DelegateMaxSteps != 20 {
		t.Fatalf("default delegate max steps = %d, want 20", c.DelegateMaxSteps)
	}

	cfgPath := writeConfig(t, `{"delegate_max_steps":5}`)
	c, err = Load(nil, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.DelegateMaxSteps != 5 {
		t.Fatalf("file delegate max steps = %d, want 5", c.DelegateMaxSteps)
	}
}

func TestDelegateMaxStepsMustBePositive(t *testing.T) {
	cfgPath := writeConfig(t, `{"delegate_max_steps":0}`)
	if _, err := Load(nil, noEnv, cfgPath); err == nil {
		t.Fatal("delegate_max_steps=0 should be invalid")
	}
}

func TestBoolFlagsParsed(t *testing.T) {
	c, err := Load([]string{"-model", "gpt-5.5", "-no-env", "-no-color", "-v", "-q"}, noEnv, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !c.NoEnv || !c.NoColor || !c.Verbose || !c.Quiet {
		t.Fatalf("bool flags not all set: %+v", c)
	}
}

func TestQuietLongFlagParsed(t *testing.T) {
	c, err := Load([]string{"--quiet"}, noEnv, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !c.Quiet {
		t.Fatalf("Quiet = false, want true")
	}
}

func TestLogLevelPrecedenceFlagBeatsEnvBeatsFile(t *testing.T) {
	cfgPath := writeConfig(t, `{"log_level":"debug"}`)
	env := envFrom(map[string]string{"LOG_LEVEL": "error"})

	c, err := Load([]string{"--log-level", "warn"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.LogLevel != "warn" {
		t.Fatalf("flag precedence: log level %q, want warn", c.LogLevel)
	}

	c, err = Load(nil, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.LogLevel != "error" {
		t.Fatalf("env precedence: log level %q, want error", c.LogLevel)
	}

	c, err = Load(nil, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.LogLevel != "debug" {
		t.Fatalf("file precedence: log level %q, want debug", c.LogLevel)
	}
}

func TestInvalidLogLevelIsUsageError(t *testing.T) {
	if _, err := Load([]string{"--log-level", "verbose"}, noEnv, ""); err == nil {
		t.Fatal("expected invalid log level to fail")
	}
}

func TestOneShotAndSessionFlags(t *testing.T) {
	c, err := Load([]string{
		"-model", "gpt-5.5",
		"-p", "do the thing",
		"-resume", "/tmp/in.json",
		"-session", "/tmp/out.json",
		"-system-override", "be terse",
	}, noEnv, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Prompt != "do the thing" || !c.PromptSet {
		t.Fatalf("prompt %q set=%v", c.Prompt, c.PromptSet)
	}
	if c.Resume != "/tmp/in.json" {
		t.Fatalf("resume %q", c.Resume)
	}
	if c.Session != "/tmp/out.json" {
		t.Fatalf("session %q", c.Session)
	}
	if c.SystemOverride != "be terse" {
		t.Fatalf("system-override %q", c.SystemOverride)
	}
}

func TestBadFlagIsUsageError(t *testing.T) {
	_, err := Load([]string{"-nonexistent-flag"}, noEnv, "")
	if err == nil {
		t.Fatalf("expected usage error for unknown flag")
	}
}

func TestBadMaxStepsValueIsUsageError(t *testing.T) {
	_, err := Load([]string{"-max-steps", "notanumber"}, noEnv, "")
	if err == nil {
		t.Fatalf("expected usage error for non-integer -max-steps")
	}
}

// helpFlags are every flag the design §10 table lists. The -h usage screen must
// name every one of them so the help is an accurate reference.
var helpFlags = []string{
	"-p", "-provider", "-model", "-model-proxy-url", "-system", "-system-override",
	"-no-env", "-resume", "-session", "-max-steps", "-default-context-window", "-context-window",
	"-reasoning-effort", "-v", "-tool-stream", "-q", "-quiet", "-log-level", "-no-color", "-config", "-prompt",
}

// -h and --help are help requests, not usage errors: Load reports ErrHelp so the
// caller can print a proper usage screen and exit 0 (design §10).
func TestHelpFlagReturnsErrHelp(t *testing.T) {
	for _, arg := range []string{"-h", "--help", "-help"} {
		_, err := Load([]string{arg}, noEnv, "")
		if !errors.Is(err, ErrHelp) {
			t.Fatalf("Load(%q) err = %v, want ErrHelp", arg, err)
		}
	}
}

func TestProviderQualifiedModelSetsProviderAndStripsModel(t *testing.T) {
	c, err := Load([]string{"-model", "openrouter:openai/gpt-5.5"}, noEnv, "")
	if err != nil {
		t.Fatalf("Load provider-qualified model: %v", err)
	}
	if c.Provider != "openrouter" || c.Model != "openai/gpt-5.5" {
		t.Fatalf("provider/model = %q/%q, want openrouter/openai/gpt-5.5", c.Provider, c.Model)
	}
}

func TestModelColonWithoutProviderQualifierStaysModel(t *testing.T) {
	c, err := Load([]string{"-model", "qwen/qwen3-coder:free"}, noEnv, "")
	if err != nil {
		t.Fatalf("Load colon model: %v", err)
	}
	if c.Provider != "" || c.Model != "qwen/qwen3-coder:free" {
		t.Fatalf("provider/model = %q/%q, want no provider and unchanged model", c.Provider, c.Model)
	}
}

// Usage writes a screen that names every design §10 flag with its default, so the
// help output is a complete and accurate reference.
func TestUsageListsEveryFlag(t *testing.T) {
	var b bytes.Buffer
	Usage(&b)
	out := b.String()
	for _, f := range helpFlags {
		if !strings.Contains(out, f) {
			t.Errorf("usage text missing flag %q:\n%s", f, out)
		}
	}
	// -max-steps default (50) must be visible so the reference is accurate.
	if !strings.Contains(out, "50") {
		t.Errorf("usage text should show the -max-steps default 50:\n%s", out)
	}
	if !strings.Contains(out, "256000") {
		t.Errorf("usage text should show the -default-context-window default 256000:\n%s", out)
	}
}

// A malformed config file is a usage/config error, not a silent ignore.
func TestMalformedConfigFileIsError(t *testing.T) {
	cfgPath := writeConfig(t, `{not valid json`)
	_, err := Load(nil, noEnv, cfgPath)
	if err == nil {
		t.Fatalf("expected error for malformed config file")
	}
}

// A missing config file at the explicit path is an error (the user asked for it);
// a missing file at the implicit default path is silently tolerated.
func TestMissingExplicitConfigFileIsError(t *testing.T) {
	_, err := Load(nil, noEnv, filepath.Join(t.TempDir(), "does-not-exist.json"))
	if err == nil {
		t.Fatalf("expected error for missing explicit config file")
	}
}

func TestModePrecedenceFlagBeatsEnvBeatsFile(t *testing.T) {
	cfgPath := writeConfig(t, `{"mode":"plan"}`)
	env := envFrom(map[string]string{"HARNESS_MODE": "independent"})

	c, err := Load([]string{"-model", "gpt-5.5", "-mode", "AUTO"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Mode != "auto" {
		t.Fatalf("flag precedence (lowercased): got mode %q, want auto", c.Mode)
	}

	c, err = Load([]string{"-model", "gpt-5.5"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Mode != "independent" {
		t.Fatalf("env precedence: got mode %q, want independent", c.Mode)
	}

	c, err = Load([]string{"-model", "gpt-5.5"}, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Mode != "plan" {
		t.Fatalf("file precedence: got mode %q, want plan", c.Mode)
	}
}

// An unspecified mode stays empty so main can distinguish "not specified"
// (session resume may supply the mode) from an explicit choice.
func TestModeUnspecifiedIsEmpty(t *testing.T) {
	c, err := Load([]string{"-model", "gpt-5.5"}, noEnv, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Mode != "" {
		t.Fatalf("mode %q, want empty when unspecified", c.Mode)
	}
}

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

func TestModesObjectDecodes(t *testing.T) {
	cfgPath := writeConfig(t, `{
		"modes": {
			"review": {"allowed_tools": ["read_file", "grep"], "prompt": "review the diff"},
			"plan": {"prompt": "custom plan prompt"}
		}
	}`)
	c, err := Load([]string{"-model", "gpt-5.5"}, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	review, ok := c.Modes["review"]
	if !ok {
		t.Fatal("modes.review not decoded")
	}
	if len(review.AllowedTools) != 2 || review.AllowedTools[0] != "read_file" || review.AllowedTools[1] != "grep" {
		t.Errorf("review.AllowedTools = %v", review.AllowedTools)
	}
	if review.Prompt != "review the diff" {
		t.Errorf("review.Prompt = %q", review.Prompt)
	}
	if c.Modes["plan"].Prompt != "custom plan prompt" {
		t.Errorf("plan.Prompt = %q", c.Modes["plan"].Prompt)
	}
	if len(c.Modes["plan"].AllowedTools) != 0 {
		t.Errorf("plan.AllowedTools should be empty (inherit), got %v", c.Modes["plan"].AllowedTools)
	}
}

func TestMCPDefaults(t *testing.T) {
	c, err := Load([]string{"-model", "gpt-5.5"}, noEnv, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.MCP.Enable {
		t.Errorf("MCP.Enable default = true, want false")
	}
	if c.MCP.Gateway != "" {
		t.Errorf("MCP.Gateway default = %q, want empty (resolved at use)", c.MCP.Gateway)
	}
}

func TestMCPFromFile(t *testing.T) {
	cfgPath := writeConfig(t, `{"mcp":{"enable":true,"gateway":"http://127.0.0.1:8766"}}`)
	c, err := Load([]string{"-model", "gpt-5.5"}, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !c.MCP.Enable {
		t.Errorf("MCP.Enable = false, want true")
	}
	if c.MCP.Gateway != "http://127.0.0.1:8766" {
		t.Errorf("MCP.Gateway = %q, want http://127.0.0.1:8766", c.MCP.Gateway)
	}
}

func TestMCPEnvOverridesFile(t *testing.T) {
	cfgPath := writeConfig(t, `{"mcp":{"enable":false,"gateway":"http://file.example/mcp"}}`)
	env := envFrom(map[string]string{
		"HARNESS_MCP_ENABLE":  "true",
		"HARNESS_MCP_GATEWAY": "http://env.example/mcp",
	})
	c, err := Load([]string{"-model", "gpt-5.5"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !c.MCP.Enable {
		t.Errorf("MCP.Enable = false, want true (env overrides file)")
	}
	if c.MCP.Gateway != "http://env.example/mcp" {
		t.Errorf("MCP.Gateway = %q, want http://env.example/mcp (env overrides file)", c.MCP.Gateway)
	}
}

func TestMCPEnableBoolParsing(t *testing.T) {
	// A bogus env value falls through to the file value (resolveBool ignores
	// unparseable env), and an empty/unset env leaves the file/default in place.
	cfgPath := writeConfig(t, `{"mcp":{"enable":true}}`)
	env := envFrom(map[string]string{"HARNESS_MCP_ENABLE": "not-a-bool"})
	c, err := Load([]string{"-model", "gpt-5.5"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !c.MCP.Enable {
		t.Errorf("MCP.Enable = false, want true (unparseable env falls back to file)")
	}

	// "0" parses as false and overrides the file's true.
	env = envFrom(map[string]string{"HARNESS_MCP_ENABLE": "0"})
	c, err = Load([]string{"-model", "gpt-5.5"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.MCP.Enable {
		t.Errorf("MCP.Enable = true, want false (HARNESS_MCP_ENABLE=0)")
	}
}

// TestMCPHeadersFromFile decodes the "headers" map under the "mcp" object.
func TestMCPHeadersFromFile(t *testing.T) {
	cfgPath := writeConfig(t, `{"mcp":{"enable":true,"gateway":"https://gw.example/mcp","headers":{"Authorization":"Bearer tok","X-Env":"prod"}}}`)
	c, err := Load([]string{"-model", "gpt-5.5"}, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := c.MCP.Headers["Authorization"]; got != "Bearer tok" {
		t.Errorf("Headers[Authorization] = %q, want %q", got, "Bearer tok")
	}
	if got := c.MCP.Headers["X-Env"]; got != "prod" {
		t.Errorf("Headers[X-Env] = %q, want %q", got, "prod")
	}
	if c.MCP.Gateway != "https://gw.example/mcp" {
		t.Errorf("Gateway = %q, want the http URL", c.MCP.Gateway)
	}
}

func TestMCPHeadersExpandEnvRefs(t *testing.T) {
	cfgPath := writeConfig(t, `{"mcp":{"headers":{"Authorization":"Bearer ${TOKEN}","X-Default":"${MISSING:-fallback}","X-Literal":"price$5 $$ ${1BAD}"}}}`)
	env := envFrom(map[string]string{"TOKEN": "secret"})
	c, err := Load([]string{"-model", "gpt-5.5"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := c.MCP.Headers["Authorization"]; got != "Bearer secret" {
		t.Fatalf("Authorization = %q, want Bearer secret", got)
	}
	if got := c.MCP.Headers["X-Default"]; got != "fallback" {
		t.Fatalf("X-Default = %q, want fallback", got)
	}
	if got := c.MCP.Headers["X-Literal"]; got != "price$5 $$ ${1BAD}" {
		t.Fatalf("X-Literal = %q, want literal dollar forms preserved", got)
	}
}

func TestMCPHeadersUnsetEnvRefErrors(t *testing.T) {
	cfgPath := writeConfig(t, `{"mcp":{"headers":{"Authorization":"Bearer ${TOKEN}"}}}`)
	if _, err := Load([]string{"-model", "gpt-5.5"}, noEnv, cfgPath); err == nil {
		t.Fatal("unset mcp header variable should error")
	} else if !strings.Contains(err.Error(), "mcp.headers.Authorization") || !strings.Contains(err.Error(), "TOKEN") {
		t.Fatalf("error should name header and variable, got %v", err)
	}
}

// TestMCPHeadersAbsentIsNil confirms an mcp block without "headers" leaves
// Headers nil (not an empty map), and that there is NO env var for headers: an
// env that looks header-ish cannot leak into the resolved map.
func TestMCPHeadersAbsentIsNil(t *testing.T) {
	cfgPath := writeConfig(t, `{"mcp":{"enable":true,"gateway":"https://gw.example/mcp"}}`)
	// Throw a plausible-but-irrelevant env at Load; headers are config-file-only.
	env := envFrom(map[string]string{
		"HARNESS_MCP_HEADERS":       `{"Authorization":"leak"}`,
		"HARNESS_MCP_AUTHORIZATION": "leak",
	})
	c, err := Load([]string{"-model", "gpt-5.5"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.MCP.Headers != nil {
		t.Errorf("Headers = %v, want nil (absent in file, no env layer)", c.MCP.Headers)
	}
}

// TestMCPHeadersNoEnvLeakageWithFileHeaders confirms env cannot override or
// augment file headers: the file is the only source.
func TestMCPHeadersNoEnvLeakageWithFileHeaders(t *testing.T) {
	cfgPath := writeConfig(t, `{"mcp":{"headers":{"Authorization":"Bearer file"}}}`)
	env := envFrom(map[string]string{
		"HARNESS_MCP_HEADERS":       `{"Authorization":"Bearer env","X-Extra":"env"}`,
		"HARNESS_MCP_AUTHORIZATION": "Bearer env",
	})
	c, err := Load([]string{"-model", "gpt-5.5"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := c.MCP.Headers["Authorization"]; got != "Bearer file" {
		t.Errorf("Headers[Authorization] = %q, want %q (env must not leak)", got, "Bearer file")
	}
	if _, ok := c.MCP.Headers["X-Extra"]; ok {
		t.Errorf("Headers gained X-Extra from env; headers are config-file-only")
	}
	if n := len(c.MCP.Headers); n != 1 {
		t.Errorf("Headers has %d entries, want 1 (file only)", n)
	}
}
