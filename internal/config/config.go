// Package config resolves user-facing settings from flags, environment
// variables, an optional config file, and built-in defaults, with precedence
// flags > env > file > defaults. It deliberately depends on neither internal/llm
// nor the model proxy client: it is the flag/env/file machinery layer, and main
// translates a resolved Config into agent/ui options.
package config

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"maps"
	"os"
	"strconv"
	"strings"

	"harness/internal/logging"
)

// ErrHelp is returned by Load when -h/--help is requested. It is not a usage
// error: the caller prints the usage screen (via Usage) and exits 0, because
// help is a request, not a misuse (design §10).
var ErrHelp = flag.ErrHelp

// Config is the fully resolved, provider-neutral configuration.
type Config struct {
	// Provider selection.
	Provider      string // model proxy provider id
	Model         string
	ModelProxyURL string

	// System prompt composition (design §8.5).
	System         string // -system: appended to the builtin instructions
	SystemOverride string // -system-override: replaces the builtin instructions
	NoEnv          bool   // -no-env: drop the env-context block

	// Session.
	Resume  string // -resume: load this transcript and continue
	Session string // -session: explicit save path

	// Loop / model limits.
	MaxSteps                  int // -max-steps, default 50
	DefaultContextWindow      int // -default-context-window, fallback for unknown/unconfigured models
	ContextWindow             int // -context-window, 0 = registry/default
	ReasoningEffort           string
	OnMaxSteps                string // -on-max-steps: "stop" (default) or "continue"
	AgentsMDWarnBytes         int    // config-only warning threshold in bytes; default 8192, explicit 0 disables
	ToolResultMaxBytes        int    // config-only; 0 = tool default
	ToolResultMaxLines        int    // config-only; 0 = tool default
	ReadFileDefaultLimit      int    // config-only; 0 = tool default
	CompactKeepTurns          int    // config-only; 0 = agent default
	CompactSummaryMaxTokens   int    // config-only; 0 = agent default
	CompactToolResultMaxBytes int    // config-only; 0 = agent default, negative disables
	DelegateMaxSteps          int    // config-only; default 20, per delegate call cap

	// Run mode. Empty means "not specified" so main can let a resumed
	// session supply the mode before falling back to the default.
	Mode  string
	Modes map[string]FileModeConfig // raw "modes" config entries; main converts to mode.FileMode

	// One-shot mode (design §10).
	Prompt    string // -p value
	PromptSet bool   // -p was supplied (distinguishes "" from absent)

	// UI.
	Verbose    bool   // -v
	ToolStream bool   // -tool-stream: show live tool-call progress
	Quiet      bool   // -q / --quiet: suppress slog-backed diagnostics
	LogLevel   string // --log-level / LOG_LEVEL: debug, info, warn, error
	NoColor    bool   // -no-color or NO_COLOR
	ReplPrompt string // -prompt: REPL input prompt (default "> ")

	// MCP proxy integration (opt-in). Proxy is the HTTP proxy URL; an empty
	// Proxy means "use the shared default", which main resolves at connect
	// time so internal/config stays free of proxy packages.
	MCP MCPConfig
}

// MCPConfig is the resolved harness-side MCP block. All downstream server
// configuration lives with the proxy; the harness only needs to know whether
// MCP is enabled and which proxy to dial.
type MCPConfig struct {
	Enable bool
	Proxy  string // http(s) proxy URL; "" means resolve the shared default at use

	// Headers are static request headers (e.g. Authorization) sent on every
	// request to the proxy. It is config-file-only (file key "headers" under
	// "mcp"), with NO env var: this matches the config-file-only precedent for
	// structured settings (a map cannot be expressed cleanly through a single env
	// var), so a header set belongs in the config file alongside the proxy URL
	// it authenticates to.
	Headers map[string]string
}

const (
	defaultMaxSteps         = 50
	defaultContextWindow    = 256_000
	defaultDelegateMaxSteps = 20
)

// FileModeConfig is one entry of the config file's "modes" object. It mirrors
// mode.FileMode; config deliberately does not import internal/mode (which
// would pull in the tools/llm layers), so main performs the conversion.
type FileModeConfig struct {
	AllowedTools []string `json:"allowed_tools"`
	Prompt       string   `json:"prompt"`
}

// fileConfig mirrors the subset of Config that the main config file may set.
// Provider connection settings and secrets belong to harness-model-proxy, not
// the harness process.
type fileConfig struct {
	Provider                  string                    `json:"provider"`
	Model                     string                    `json:"model"`
	ModelProxyURL             string                    `json:"model_proxy_url"`
	System                    string                    `json:"system"`
	NoEnv                     *bool                     `json:"no_env"`
	MaxSteps                  *int                      `json:"max_steps"`
	DefaultContextWindow      *int                      `json:"default_context_window"`
	ContextWindow             *int                      `json:"context_window"`
	ReasoningEffort           string                    `json:"reasoning_effort"`
	OnMaxSteps                string                    `json:"on_max_steps"`
	AgentsMDWarnBytes         *int                      `json:"agents_md_warn_bytes"`
	ToolResultMaxBytes        *int                      `json:"tool_result_max_bytes"`
	ToolResultMaxLines        *int                      `json:"tool_result_max_lines"`
	ReadFileDefaultLimit      *int                      `json:"read_file_default_limit"`
	CompactKeepTurns          *int                      `json:"compact_keep_turns"`
	CompactSummaryMaxTokens   *int                      `json:"compact_summary_max_tokens"`
	CompactToolResultMaxBytes *int                      `json:"compact_tool_result_max_bytes"`
	DelegateMaxSteps          *int                      `json:"delegate_max_steps"`
	Verbose                   *bool                     `json:"verbose"`
	ToolStream                *bool                     `json:"tool_stream"`
	LogLevel                  string                    `json:"log_level"`
	NoColor                   *bool                     `json:"no_color"`
	Prompt                    string                    `json:"prompt"`
	Mode                      string                    `json:"mode"`
	Modes                     map[string]FileModeConfig `json:"modes"`

	MCP *fileMCPConfig `json:"mcp"`
}

// fileMCPConfig mirrors the config file's "mcp" object. Pointer/string fields
// follow the existing unset-detection convention: a nil block means "no mcp
// config", letting env and defaults supply every field.
type fileMCPConfig struct {
	Enable  *bool             `json:"enable"`
	Proxy   string            `json:"proxy"`
	Headers map[string]string `json:"headers"`
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

	fc, err := readConfigFile(configPath)
	if err != nil {
		return Config{}, err
	}

	fProvider, fModel, fModelProxyURL := f.provider, f.model, f.modelProxyURL
	fSystem, fSystemOverride, fNoEnv := f.system, f.systemOverride, f.noEnv
	fResume, fSession := f.resume, f.session
	fMaxSteps, fDefaultContextWindow, fContextWindow := f.maxSteps, f.defaultContextWindow, f.contextWindow
	fReasoningEffort := f.reasoningEffort
	fPrompt, fReplPrompt, fVerbose, fToolStream, fNoColor := f.prompt, f.replPrompt, f.verbose, f.toolStream, f.noColor

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
	if provider, model, ok := SplitProviderModel(c.Model); ok {
		if c.Provider == "" {
			c.Provider = provider
		}
		c.Model = model
	}
	c.ModelProxyURL = resolveString(set["model-proxy-url"], *fModelProxyURL,
		getenv("HARNESS_MODEL_PROXY_URL"), fc.ModelProxyURL, "")
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
	c.DefaultContextWindow = resolveInt(set["default-context-window"], *fDefaultContextWindow,
		getenv("HARNESS_DEFAULT_CONTEXT_WINDOW"), fc.DefaultContextWindow, defaultContextWindow)
	c.ContextWindow = resolveInt(set["context-window"], *fContextWindow,
		getenv("HARNESS_CONTEXT_WINDOW"), fc.ContextWindow, 0)
	c.ReasoningEffort = strings.ToLower(strings.TrimSpace(resolveString(set["reasoning-effort"], *fReasoningEffort,
		getenv("HARNESS_REASONING_EFFORT"), fc.ReasoningEffort, "")))
	c.OnMaxSteps = strings.ToLower(strings.TrimSpace(resolveString(set["on-max-steps"], *f.onMaxSteps,
		getenv("HARNESS_ON_MAX_STEPS"), fc.OnMaxSteps, "stop")))
	if c.OnMaxSteps != "stop" && c.OnMaxSteps != "continue" {
		return Config{}, fmt.Errorf("invalid -on-max-steps %q (valid: stop, continue)", c.OnMaxSteps)
	}
	c.AgentsMDWarnBytes = intValue(fc.AgentsMDWarnBytes, 8192)
	c.ToolResultMaxBytes = intValue(fc.ToolResultMaxBytes, 0)
	c.ToolResultMaxLines = intValue(fc.ToolResultMaxLines, 0)
	c.ReadFileDefaultLimit = intValue(fc.ReadFileDefaultLimit, 0)
	c.CompactKeepTurns = intValue(fc.CompactKeepTurns, 0)
	c.CompactSummaryMaxTokens = intValue(fc.CompactSummaryMaxTokens, 0)
	c.CompactToolResultMaxBytes = intValue(fc.CompactToolResultMaxBytes, 0)
	c.DelegateMaxSteps = intValue(fc.DelegateMaxSteps, defaultDelegateMaxSteps)
	if c.DelegateMaxSteps <= 0 {
		return Config{}, fmt.Errorf("delegate_max_steps must be positive")
	}
	c.Mode = strings.ToLower(strings.TrimSpace(resolveString(set["mode"], *f.mode,
		getenv("HARNESS_MODE"), fc.Mode, "")))
	c.Modes = fc.Modes

	c.NoEnv = resolveBool(set["no-env"], *fNoEnv,
		getenv("HARNESS_NO_ENV"), fc.NoEnv, false)
	c.Verbose = resolveBool(set["v"], *fVerbose,
		getenv("HARNESS_VERBOSE"), fc.Verbose, false)
	c.ToolStream = resolveBool(set["tool-stream"], *fToolStream,
		getenv("HARNESS_TOOL_STREAM"), fc.ToolStream, true)
	c.Quiet = *f.quietShort || *f.quiet
	logLevel := resolveString(set["log-level"], *f.logLevel,
		getenv("LOG_LEVEL"), fc.LogLevel, logging.LevelInfo)
	canonicalLogLevel, err := logging.CanonicalLevel(logLevel)
	if err != nil {
		return Config{}, err
	}
	c.LogLevel = canonicalLogLevel
	c.NoColor = resolveBool(set["no-color"], *fNoColor,
		getenv("HARNESS_NO_COLOR"), fc.NoColor, false)
	c.ReplPrompt = resolveString(set["prompt"], *fReplPrompt,
		getenv("HARNESS_PROMPT"), fc.Prompt, "> ")

	// MCP block (env > file > default; no flags). Proxy is left empty when
	// unset so main can resolve the shared default HTTP URL at connect time.
	var mcpEnableFile *bool
	var mcpProxyFile string
	if fc.MCP != nil {
		mcpEnableFile = fc.MCP.Enable
		mcpProxyFile = fc.MCP.Proxy
		// Headers are config-file-only (no env layer); copy so a later mutation
		// of fc cannot reach the resolved Config. Values support ${VAR} and
		// ${VAR:-default} interpolation. Absent → nil.
		if len(fc.MCP.Headers) > 0 {
			headers, err := expandMCPHeaders(fc.MCP.Headers, getenv)
			if err != nil {
				return Config{}, err
			}
			c.MCP.Headers = headers
		}
	}
	c.MCP.Enable = resolveBool(false, false,
		getenv("HARNESS_MCP_ENABLE"), mcpEnableFile, false)
	c.MCP.Proxy = resolveString(false, "",
		getenv("HARNESS_MCP_PROXY"), mcpProxyFile, "")

	// NO_COLOR (the de-facto standard) disables color regardless of HARNESS_*.
	if getenv("NO_COLOR") != "" {
		c.NoColor = true
	}

	if set["p"] {
		c.Prompt = *fPrompt
		c.PromptSet = true
	}

	return c, nil
}

func expandMCPHeaders(headers map[string]string, getenv func(string) string) (map[string]string, error) {
	out := maps.Clone(headers)
	for k, v := range out {
		expanded, err := expandMCPHeaderValue(v, getenv)
		if err != nil {
			return nil, fmt.Errorf("mcp.headers.%s: %w", k, err)
		}
		out[k] = expanded
	}
	return out, nil
}

func expandMCPHeaderValue(s string, getenv func(string) string) (string, error) {
	if !strings.ContainsRune(s, '$') {
		return s, nil
	}
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); {
		c := s[i]
		if c != '$' {
			b.WriteByte(c)
			i++
			continue
		}
		ref, ok := parseMCPHeaderVarRef(s, i)
		if !ok {
			b.WriteByte('$')
			i++
			continue
		}
		if val := getenv(ref.name); val != "" {
			b.WriteString(val)
		} else if ref.hasDefault {
			b.WriteString(ref.def)
		} else {
			return "", fmt.Errorf("references unset variable ${%s}", ref.name)
		}
		i = ref.end
	}
	return b.String(), nil
}

type mcpHeaderVarRef struct {
	name       string
	def        string
	hasDefault bool
	end        int
}

func parseMCPHeaderVarRef(s string, i int) (mcpHeaderVarRef, bool) {
	if i+1 >= len(s) || s[i+1] != '{' {
		return mcpHeaderVarRef{}, false
	}
	j := i + 2
	start := j
	for j < len(s) && s[j] != '}' {
		j++
	}
	if j >= len(s) {
		return mcpHeaderVarRef{}, false
	}
	body := s[start:j]
	name, def, hasDefault := strings.Cut(body, ":-")
	if !isMCPHeaderVarName(name) {
		return mcpHeaderVarRef{}, false
	}
	return mcpHeaderVarRef{name: name, def: def, hasDefault: hasDefault, end: j + 1}, true
}

func isMCPHeaderVarName(name string) bool {
	if name == "" {
		return false
	}
	for i := 0; i < len(name); i++ {
		c := name[i]
		switch {
		case c == '_':
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z':
		case c >= '0' && c <= '9':
			if i == 0 {
				return false
			}
		default:
			return false
		}
	}
	return true
}

// flags holds the pointers returned by the FlagSet so the same flag definitions
// back both Load (parsing) and Usage (the -h screen) — one source of truth, so
// the help can never drift from what is actually parsed (design §10).
type flags struct {
	provider, model, modelProxyURL *string
	system, systemOverride         *string
	noEnv                          *bool
	resume, session                *string
	maxSteps                       *int
	defaultContextWindow           *int
	contextWindow                  *int
	reasoningEffort                *string
	onMaxSteps                     *string
	mode                           *string
	prompt                         *string
	replPrompt                     *string
	logLevel                       *string
	verbose, toolStream            *bool
	noColor                        *bool
	quietShort, quiet              *bool
	config                         *string
}

// newFlagSet defines every design §10 flag on a fresh FlagSet, used by both Load
// and Usage so the help screen lists exactly the flags that are parsed.
func newFlagSet() (*flag.FlagSet, flags) {
	fs := flag.NewFlagSet("harness", flag.ContinueOnError)
	var f flags
	f.prompt = fs.String("p", "", "one-shot prompt; \"-\" or piped stdin reads the prompt from stdin")
	f.provider = fs.String("provider", "", "model proxy provider id")
	f.model = fs.String("model", "", "model id")
	f.modelProxyURL = fs.String("model-proxy-url", "", "harness-model-proxy URL")
	f.system = fs.String("system", "", "append to system prompt (text or @file)")
	f.systemOverride = fs.String("system-override", "", "replace builtin instructions (text or @file)")
	f.noEnv = fs.Bool("no-env", false, "omit the environment context block")
	f.resume = fs.String("resume", "", "load a session transcript and continue")
	f.session = fs.String("session", "", "explicit session save path")
	f.maxSteps = fs.Int("max-steps", defaultMaxSteps, "model round-trips per user turn")
	f.defaultContextWindow = fs.Int("default-context-window", defaultContextWindow, "default context window for unknown/unconfigured models (tokens)")
	f.contextWindow = fs.Int("context-window", 0, "context window override (tokens)")
	f.reasoningEffort = fs.String("reasoning-effort", "", "reasoning/thinking effort (provider/model dependent)")
	f.onMaxSteps = fs.String("on-max-steps", "", "when the step budget is hit: stop (default) or continue (up to 3 fresh budgets)")
	f.mode = fs.String("mode", "", "run mode: auto, plan, independent, or a config-defined mode (default auto)")
	f.verbose = fs.Bool("v", false, "show tool result snippets")
	f.toolStream = fs.Bool("tool-stream", true, "show live tool-call progress")
	f.quietShort = fs.Bool("q", false, "suppress informational diagnostics")
	f.quiet = fs.Bool("quiet", false, "suppress informational diagnostics")
	f.logLevel = fs.String("log-level", logging.LevelInfo, "diagnostic log level: debug, info, warn, error (also LOG_LEVEL)")
	f.noColor = fs.Bool("no-color", false, "disable color output")
	f.replPrompt = fs.String("prompt", "> ", "REPL input prompt")
	// -config is consumed by the caller before Load (it picks the file Load
	// reads); accepted here so it is not rejected as an unknown flag.
	f.config = fs.String("config", "", "alternate config path")
	return fs, f
}

// Usage writes the -h/--help screen: a one-line summary followed by every design
// §10 flag with its description and default. Provider secrets are configured on
// harness-model-proxy, not exposed through harness flags.
func Usage(w io.Writer) {
	fmt.Fprintln(w, "harness — a minimal agentic coding harness.")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Usage:")
	fmt.Fprintln(w, "  harness [flags]            interactive REPL")
	fmt.Fprintln(w, "  harness -p \"prompt\" [flags]  one-shot: prints the assistant's answer to stdout")
	fmt.Fprintln(w, "  harness session replay <session-dir>")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Model provider access goes through harness-model-proxy.")
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

func intValue(v *int, def int) int {
	if v == nil {
		return def
	}
	return *v
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

// SplitProviderModel splits a "provider:model" string into its parts. The
// provider half must look like a provider name ([a-zA-Z0-9._-]); anything else
// (e.g. a model id with a colon in it) is returned as not-ok. Shared with the
// REPL /model switch in cmd/harness.
func SplitProviderModel(model string) (provider, bareModel string, ok bool) {
	provider, bareModel, ok = strings.Cut(strings.TrimSpace(model), ":")
	if !ok || provider == "" || bareModel == "" {
		return "", "", false
	}
	for _, r := range provider {
		switch {
		case r >= 'a' && r <= 'z':
		case r >= 'A' && r <= 'Z':
		case r >= '0' && r <= '9':
		case r == '-' || r == '_' || r == '.':
		default:
			return "", "", false
		}
	}
	return strings.ToLower(provider), bareModel, true
}
