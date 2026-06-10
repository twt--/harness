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

func TestBaseURLPrecedenceFlagBeatsEnvBeatsFile(t *testing.T) {
	cfgPath := writeConfig(t, `{"base_url":"http://file.example"}`)
	env := envFrom(map[string]string{"HARNESS_BASE_URL": "http://env.example"})

	c, err := Load([]string{"-base-url", "http://flag.example"}, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.BaseURL != "http://flag.example" {
		t.Fatalf("flag precedence: got base-url %q", c.BaseURL)
	}

	c, err = Load(nil, env, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.BaseURL != "http://env.example" {
		t.Fatalf("env precedence: got base-url %q", c.BaseURL)
	}

	c, err = Load(nil, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.BaseURL != "http://file.example" {
		t.Fatalf("file precedence: got base-url %q", c.BaseURL)
	}
}

func TestProviderConfigsReadFromConfigFile(t *testing.T) {
	cfgPath := writeConfig(t, `{"provider_configs":["openai.json","anthropic.json"]}`)
	c, err := Load(nil, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := strings.Join(c.ProviderConfigs, ","); got != "openai.json,anthropic.json" {
		t.Fatalf("provider configs %q, want openai.json,anthropic.json", got)
	}
}

// The provider-specific base-url env vars seed the base URL for the selected
// provider. A custom base URL also lets the empty-API-key case stand.
func TestProviderSpecificBaseURLEnv(t *testing.T) {
	env := envFrom(map[string]string{"OPENAI_BASE_URL": "http://localhost:11434/v1"})
	c, err := Load([]string{"-model", "llama3"}, env, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Provider != "openai" {
		t.Fatalf("inferred provider %q, want openai", c.Provider)
	}
	if c.BaseURL != "http://localhost:11434/v1" {
		t.Fatalf("base-url %q, want OPENAI_BASE_URL value", c.BaseURL)
	}
}

func TestAnthropicBaseURLEnv(t *testing.T) {
	env := envFrom(map[string]string{"ANTHROPIC_BASE_URL": "http://local-anthropic"})
	c, err := Load([]string{"-model", "claude-opus-4-8"}, env, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Provider != "anthropic" {
		t.Fatalf("inferred provider %q, want anthropic", c.Provider)
	}
	if c.BaseURL != "http://local-anthropic" {
		t.Fatalf("base-url %q, want ANTHROPIC_BASE_URL value", c.BaseURL)
	}
}

// An explicit -base-url flag overrides the provider-specific base-url env var.
func TestBaseURLFlagBeatsProviderEnv(t *testing.T) {
	env := envFrom(map[string]string{"OPENAI_BASE_URL": "http://env.example"})
	c, err := Load([]string{"-model", "gpt-5.5", "-base-url", "http://flag.example"}, env, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.BaseURL != "http://flag.example" {
		t.Fatalf("base-url %q, want flag value", c.BaseURL)
	}
}

func TestAPIKeyReadFromEnvOnlyAnthropic(t *testing.T) {
	env := envFrom(map[string]string{"ANTHROPIC_API_KEY": "sk-ant-secret"})
	c, err := Load([]string{"-model", "claude-opus-4-8"}, env, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.APIKey != "sk-ant-secret" {
		t.Fatalf("api key %q, want ANTHROPIC_API_KEY value", c.APIKey)
	}
}

func TestAPIKeyReadFromEnvOnlyOpenAI(t *testing.T) {
	env := envFrom(map[string]string{"OPENAI_API_KEY": "sk-openai-secret"})
	c, err := Load([]string{"-model", "gpt-5.5"}, env, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.APIKey != "sk-openai-secret" {
		t.Fatalf("api key %q, want OPENAI_API_KEY value", c.APIKey)
	}
}

// API keys are never read from the config file. A key-like field there must be
// ignored entirely.
func TestAPIKeyNotReadFromConfigFile(t *testing.T) {
	cfgPath := writeConfig(t, `{"model":"gpt-5.5","api_key":"sk-leaked","openai_api_key":"sk-leaked2"}`)
	c, err := Load(nil, noEnv, cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.APIKey != "" {
		t.Fatalf("api key %q, want empty (keys are env-only)", c.APIKey)
	}
}

// The provider chooses which API-key env var is read: anthropic -> ANTHROPIC_API_KEY.
func TestAPIKeySelectedByProvider(t *testing.T) {
	env := envFrom(map[string]string{
		"OPENAI_API_KEY":    "sk-openai",
		"ANTHROPIC_API_KEY": "sk-ant",
	})
	c, err := Load([]string{"-model", "claude-opus-4-8"}, env, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.APIKey != "sk-ant" {
		t.Fatalf("api key %q, want sk-ant for anthropic provider", c.APIKey)
	}

	c, err = Load([]string{"-model", "gpt-5.5"}, env, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.APIKey != "sk-openai" {
		t.Fatalf("api key %q, want sk-openai for openai provider", c.APIKey)
	}
}

func TestExplicitProviderOverridesInference(t *testing.T) {
	// A claude* model would infer anthropic, but explicit -provider wins.
	c, err := Load([]string{"-model", "claude-opus-4-8", "-provider", "openai"}, noEnv, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Provider != "openai" {
		t.Fatalf("provider %q, want openai (explicit overrides inference)", c.Provider)
	}
}

func TestProviderInferenceFromModel(t *testing.T) {
	for _, tc := range []struct{ model, want string }{
		{"claude-opus-4-8", "anthropic"},
		{"claude-haiku-4-5", "anthropic"},
		{"gpt-5.5", "openai"},
		{"llama3.2", "openai"},
		{"some-local-model", "openai"},
	} {
		c, err := Load([]string{"-model", tc.model}, noEnv, "")
		if err != nil {
			t.Fatalf("Load %q: %v", tc.model, err)
		}
		if c.Provider != tc.want {
			t.Fatalf("model %q inferred provider %q, want %q", tc.model, c.Provider, tc.want)
		}
	}
}

// HARNESS_* env mapping covers the non-key, non-base-url flags too.
func TestHarnessEnvMapping(t *testing.T) {
	env := envFrom(map[string]string{
		"HARNESS_MODEL":          "env-model",
		"HARNESS_MAX_STEPS":      "12",
		"HARNESS_CONTEXT_WINDOW": "256000",
		"HARNESS_SYSTEM":         "env system note",
		"HARNESS_NO_ENV":         "true",
		"HARNESS_NO_COLOR":       "true",
		"HARNESS_VERBOSE":        "true",
	})
	c, err := Load(nil, env, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Model != "env-model" {
		t.Fatalf("model %q", c.Model)
	}
	if c.MaxSteps != 12 {
		t.Fatalf("max-steps %d, want 12", c.MaxSteps)
	}
	if c.ContextWindow != 256000 {
		t.Fatalf("context-window %d, want 256000", c.ContextWindow)
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

func TestBoolFlagsParsed(t *testing.T) {
	c, err := Load([]string{"-model", "gpt-5.5", "-no-env", "-no-color", "-v"}, noEnv, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !c.NoEnv || !c.NoColor || !c.Verbose {
		t.Fatalf("bool flags not all set: %+v", c)
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
	"-p", "-provider", "-model", "-base-url", "-system", "-system-override",
	"-no-env", "-resume", "-session", "-max-steps", "-context-window",
	"-v", "-no-color", "-config", "-setup",
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

func TestSetupFlagReturnsSetupWithoutReadingConfig(t *testing.T) {
	c, err := Load([]string{"--setup"}, noEnv, filepath.Join(t.TempDir(), "missing.json"))
	if err != nil {
		t.Fatalf("Load --setup: %v", err)
	}
	if !c.Setup {
		t.Fatalf("Setup = false, want true")
	}
	if c.Model != "" || c.Provider != "" {
		t.Fatalf("setup config should not resolve model/provider, got %+v", c)
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
