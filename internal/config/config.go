// Package config resolves user-facing settings from flags, environment
// variables, an optional config file, and built-in defaults, with precedence
// flags > env > file > defaults (design §7). It deliberately depends on neither
// internal/llm nor the provider factory: it is the flag/env/file machinery
// layer, and main (Phase 10) translates a resolved Config into factory.Options
// and agent/ui options.
//
// API keys are never read from flags or the main config file. Provider config
// files may carry keys for users who choose that tradeoff; environment variables
// still take precedence where the main package resolves the selected provider.
package config

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// ErrHelp is returned by Load when -h/--help is requested. It is not a usage
// error: the caller prints the usage screen (via Usage) and exits 0, because
// help is a request, not a misuse (design §10).
var ErrHelp = flag.ErrHelp

// Config is the fully resolved, provider-neutral configuration.
type Config struct {
	Setup bool // -setup: create a basic config in the default config directory

	// Provider selection.
	Provider string // provider config name or api type; resolved after inference
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

	// Provider configs: filenames resolved relative to the config file's directory.
	ProviderConfigs []string
}

const defaultMaxSteps = 50

// fileConfig mirrors the subset of Config that the main config file may set.
// API keys are intentionally absent here; provider config files carry provider
// connection settings and optional keys.
type fileConfig struct {
	Provider        string   `json:"provider"`
	Model           string   `json:"model"`
	BaseURL         string   `json:"base_url"`
	System          string   `json:"system"`
	NoEnv           *bool    `json:"no_env"`
	MaxSteps        *int     `json:"max_steps"`
	ContextWindow   *int     `json:"context_window"`
	Verbose         *bool    `json:"verbose"`
	NoColor         *bool    `json:"no_color"`
	ProviderConfigs []string `json:"provider_configs"`
}

// Load resolves a Config from the given args (argv after the program name), a
// getenv accessor, and a config-file path. getenv and configPath are injected so
// the loader has no hidden dependency on os.Args/os.Getenv and is testable. An
// empty configPath means "no config file": the caller (main) is responsible for
// existence-checking the implicit default ~/.config/harness/config.json and
// passing "" when it is absent, so that a non-empty path is always required to
// exist (a typo'd -config must not be silently ignored).
func Load(args []string, getenv func(string) string, configPath string) (Config, error) {
	fs, f := newFlagSet()
	fs.SetOutput(io.Discard) // errors are returned, not printed by the loader

	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}

	// -setup is intentionally independent of any existing config file: it is the
	// path for creating the default config, so a malformed or missing explicit
	// config should not block setup from running.
	if *f.setup {
		return Config{Setup: true}, nil
	}

	fc, err := readConfigFile(configPath)
	if err != nil {
		return Config{}, err
	}

	fProvider, fModel, fBaseURL := f.provider, f.model, f.baseURL
	fSystem, fSystemOverride, fNoEnv := f.system, f.systemOverride, f.noEnv
	fResume, fSession := f.resume, f.session
	fMaxSteps, fContextWindow := f.maxSteps, f.contextWindow
	fPrompt, fVerbose, fNoColor := f.prompt, f.verbose, f.noColor

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
	c.ProviderConfigs = append([]string(nil), fc.ProviderConfigs...)
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

// flags holds the pointers returned by the FlagSet so the same flag definitions
// back both Load (parsing) and Usage (the -h screen) — one source of truth, so
// the help can never drift from what is actually parsed (design §10).
type flags struct {
	provider, model, baseURL *string
	system, systemOverride   *string
	noEnv                    *bool
	resume, session          *string
	maxSteps, contextWindow  *int
	prompt                   *string
	verbose, noColor         *bool
	config                   *string
	setup                    *bool
}

// newFlagSet defines every design §10 flag on a fresh FlagSet, used by both Load
// and Usage so the help screen lists exactly the flags that are parsed.
func newFlagSet() (*flag.FlagSet, flags) {
	fs := flag.NewFlagSet("harness", flag.ContinueOnError)
	var f flags
	f.prompt = fs.String("p", "", "one-shot prompt; \"-\" or piped stdin reads the prompt from stdin")
	f.provider = fs.String("provider", "", "provider config name or api type (default: inferred from -model)")
	f.model = fs.String("model", "", "model id (required)")
	f.baseURL = fs.String("base-url", "", "provider base URL (e.g. http://localhost:11434/v1 for Ollama)")
	f.system = fs.String("system", "", "append to system prompt (text or @file)")
	f.systemOverride = fs.String("system-override", "", "replace builtin instructions (text or @file)")
	f.noEnv = fs.Bool("no-env", false, "omit the environment context block")
	f.resume = fs.String("resume", "", "load a session transcript and continue")
	f.session = fs.String("session", "", "explicit session save path")
	f.maxSteps = fs.Int("max-steps", defaultMaxSteps, "model round-trips per user turn")
	f.contextWindow = fs.Int("context-window", 0, "context window override (tokens)")
	f.verbose = fs.Bool("v", false, "show tool result snippets")
	f.noColor = fs.Bool("no-color", false, "disable color output")
	// -config is consumed by the caller before Load (it picks the file Load
	// reads); accepted here so it is not rejected as an unknown flag.
	f.config = fs.String("config", "", "alternate config path")
	f.setup = fs.Bool("setup", false, "create a basic config in the default config directory")
	return fs, f
}

// Usage writes the -h/--help screen: a one-line summary followed by every design
// §10 flag with its description and default. API keys are intentionally absent
// from flags; use environment variables or provider config files.
func Usage(w io.Writer) {
	fmt.Fprintln(w, "harness — a minimal agentic coding harness.")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Usage:")
	fmt.Fprintln(w, "  harness [flags]            interactive REPL")
	fmt.Fprintln(w, "  harness -p \"prompt\" [flags]  one-shot: prints the assistant's answer to stdout")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "API keys come from environment variables or provider config files; env wins.")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Flags:")
	fs, _ := newFlagSet()
	fs.SetOutput(w)
	fs.PrintDefaults()
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
