// Package config resolves user-facing settings from flags, environment
// variables, an optional config file, and built-in defaults, with precedence
// flags > env > file > defaults (design §7). It deliberately depends on neither
// internal/llm nor the provider factory: it is the flag/env/file machinery
// layer, and main (Phase 10) translates a resolved Config into factory.Options
// and agent/ui options.
//
// API keys are read from the environment only — never from flags or the config
// file — because both leak into shell history and committed dotfiles (design §2,
// §7).
package config

import (
	"encoding/json"
	"flag"
	"io"
	"os"
	"strconv"
	"strings"
)

// Config is the fully resolved, provider-neutral configuration.
type Config struct {
	// Provider selection.
	Provider string // "openai" | "anthropic"; resolved after inference
	Model    string
	BaseURL  string
	APIKey   string // env-only; selected by the resolved provider

	// System prompt composition (design §8.5).
	System         string // -system: appended to the builtin instructions
	SystemOverride string // -system-override: replaces the builtin instructions
	NoEnv          bool   // -no-env: drop the env-context block

	// Session.
	Resume  string // -resume: load this transcript and continue
	Session string // -session: explicit save path

	// Loop / model limits.
	MaxSteps      int // -max-steps, default 50
	ContextWindow int // -context-window, 0 = registry/default

	// One-shot mode (design §10).
	Prompt    string // -p value
	PromptSet bool   // -p was supplied (distinguishes "" from absent)

	// UI.
	Verbose bool // -v
	NoColor bool // -no-color or NO_COLOR
}

const defaultMaxSteps = 50

// fileConfig mirrors the subset of Config that the config file may set
// (design §7). API keys are intentionally absent: keys are env-only, so any
// key-like field in the file is ignored.
type fileConfig struct {
	Provider      string `json:"provider"`
	Model         string `json:"model"`
	BaseURL       string `json:"base_url"`
	System        string `json:"system"`
	NoEnv         *bool  `json:"no_env"`
	MaxSteps      *int   `json:"max_steps"`
	ContextWindow *int   `json:"context_window"`
	Verbose       *bool  `json:"verbose"`
	NoColor       *bool  `json:"no_color"`
}

// Load resolves a Config from the given args (argv after the program name), a
// getenv accessor, and a config-file path. getenv and configPath are injected so
// the loader has no hidden dependency on os.Args/os.Getenv and is testable. An
// empty configPath means "no config file": the caller (main) is responsible for
// existence-checking the implicit default ~/.config/harness/config.json and
// passing "" when it is absent, so that a non-empty path is always required to
// exist (a typo'd -config must not be silently ignored).
func Load(args []string, getenv func(string) string, configPath string) (Config, error) {
	fc, err := readConfigFile(configPath)
	if err != nil {
		return Config{}, err
	}

	fs := flag.NewFlagSet("harness", flag.ContinueOnError)
	fs.SetOutput(io.Discard) // errors are returned, not printed by the loader

	var (
		fProvider       = fs.String("provider", "", "openai | anthropic (default: inferred from -model)")
		fModel          = fs.String("model", "", "model id")
		fBaseURL        = fs.String("base-url", "", "provider base URL")
		fSystem         = fs.String("system", "", "append to system prompt (text or @file)")
		fSystemOverride = fs.String("system-override", "", "replace builtin instructions (text or @file)")
		fNoEnv          = fs.Bool("no-env", false, "omit the environment context block")
		fResume         = fs.String("resume", "", "load a session transcript and continue")
		fSession        = fs.String("session", "", "explicit session save path")
		fMaxSteps       = fs.Int("max-steps", defaultMaxSteps, "model round-trips per user turn")
		fContextWindow  = fs.Int("context-window", 0, "context window override")
		fPrompt         = fs.String("p", "", "one-shot prompt; \"-\" or piped stdin reads stdin")
		fVerbose        = fs.Bool("v", false, "show tool result snippets")
		fNoColor        = fs.Bool("no-color", false, "disable color output")
		// -config is consumed by the caller before Load; accepted here so it is
		// not rejected as an unknown flag.
		_ = fs.String("config", "", "alternate config path")
	)

	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}

	// set records which flags were explicitly provided, so a flag only overrides
	// env/file when actually present (flag defaults must not beat lower sources).
	set := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })

	var c Config

	// Each resolution goes default -> file -> env -> flag, last writer wins.

	c.Model = resolveString(set["model"], *fModel,
		getenv("HARNESS_MODEL"), fc.Model, "")
	c.Provider = resolveString(set["provider"], *fProvider,
		getenv("HARNESS_PROVIDER"), fc.Provider, "")
	c.System = resolveString(set["system"], *fSystem,
		getenv("HARNESS_SYSTEM"), fc.System, "")

	if set["system-override"] {
		c.SystemOverride = *fSystemOverride
	} else {
		c.SystemOverride = getenv("HARNESS_SYSTEM_OVERRIDE")
	}
	if set["resume"] {
		c.Resume = *fResume
	} else {
		c.Resume = getenv("HARNESS_RESUME")
	}
	if set["session"] {
		c.Session = *fSession
	} else {
		c.Session = getenv("HARNESS_SESSION")
	}

	c.MaxSteps = resolveInt(set["max-steps"], *fMaxSteps,
		getenv("HARNESS_MAX_STEPS"), fc.MaxSteps, defaultMaxSteps)
	c.ContextWindow = resolveInt(set["context-window"], *fContextWindow,
		getenv("HARNESS_CONTEXT_WINDOW"), fc.ContextWindow, 0)

	c.NoEnv = resolveBool(set["no-env"], *fNoEnv,
		getenv("HARNESS_NO_ENV"), fc.NoEnv, false)
	c.Verbose = resolveBool(set["v"], *fVerbose,
		getenv("HARNESS_VERBOSE"), fc.Verbose, false)
	c.NoColor = resolveBool(set["no-color"], *fNoColor,
		getenv("HARNESS_NO_COLOR"), fc.NoColor, false)
	// NO_COLOR (the de-facto standard) disables color regardless of HARNESS_*.
	if getenv("NO_COLOR") != "" {
		c.NoColor = true
	}

	if set["p"] {
		c.Prompt = *fPrompt
		c.PromptSet = true
	}

	// Provider inference (design §7): -model is primary; claude* -> anthropic,
	// else openai. An explicit provider (flag/env/file, resolved above) wins.
	if c.Provider == "" {
		c.Provider = inferProvider(c.Model)
	}

	// Base URL: provider-specific env var seeds the default, then the generic
	// HARNESS_BASE_URL / file / flag layer on top.
	base := providerBaseURLEnv(c.Provider, getenv)
	c.BaseURL = resolveString(set["base-url"], *fBaseURL,
		getenv("HARNESS_BASE_URL"), fc.BaseURL, base)

	// API key: env-only, selected by the resolved provider.
	c.APIKey = providerAPIKeyEnv(c.Provider, getenv)

	return c, nil
}

// readConfigFile reads and decodes the config file at path. An empty path means
// "no config"; a missing or malformed file at a non-empty path is an error (the
// path was requested, so silently ignoring it would hide a typo).
func readConfigFile(path string) (fileConfig, error) {
	if path == "" {
		return fileConfig{}, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return fileConfig{}, err
	}
	var fc fileConfig
	if err := json.Unmarshal(data, &fc); err != nil {
		return fileConfig{}, err
	}
	return fc, nil
}

// resolveString returns the highest-precedence non-empty value among an
// explicitly-set flag, an env value, a file value, and a default.
func resolveString(flagSet bool, flagVal, envVal, fileVal, def string) string {
	if flagSet && flagVal != "" {
		return flagVal
	}
	if envVal != "" {
		return envVal
	}
	if fileVal != "" {
		return fileVal
	}
	return def
}

// resolveInt mirrors resolveString for integers. fileVal of nil means unset.
func resolveInt(flagSet bool, flagVal int, envVal string, fileVal *int, def int) int {
	if flagSet {
		return flagVal
	}
	if envVal != "" {
		if n, err := strconv.Atoi(envVal); err == nil {
			return n
		}
	}
	if fileVal != nil {
		return *fileVal
	}
	return def
}

// resolveBool mirrors resolveString for booleans. fileVal of nil means unset.
func resolveBool(flagSet bool, flagVal bool, envVal string, fileVal *bool, def bool) bool {
	if flagSet {
		return flagVal
	}
	if envVal != "" {
		if b, err := strconv.ParseBool(envVal); err == nil {
			return b
		}
	}
	if fileVal != nil {
		return *fileVal
	}
	return def
}

// inferProvider applies the §7 rule: model names starting with "claude" are
// anthropic; everything else is OpenAI-compatible.
func inferProvider(model string) string {
	if strings.HasPrefix(model, "claude") {
		return "anthropic"
	}
	return "openai"
}

// providerBaseURLEnv returns the provider-specific base-url env var value, or "".
func providerBaseURLEnv(provider string, getenv func(string) string) string {
	switch provider {
	case "anthropic":
		return getenv("ANTHROPIC_BASE_URL")
	default:
		return getenv("OPENAI_BASE_URL")
	}
}

// providerAPIKeyEnv returns the provider-specific API-key env var value, or "".
func providerAPIKeyEnv(provider string, getenv func(string) string) string {
	switch provider {
	case "anthropic":
		return getenv("ANTHROPIC_API_KEY")
	default:
		return getenv("OPENAI_API_KEY")
	}
}
