// Package mcpgateway implements the MCP gateway daemon: it loads a Claude
// Code-compatible config, supervises downstream MCP servers (stdio children and
// streamable-HTTP endpoints), aggregates their tools under a stable namespace,
// and serves the merged tool surface to harness over a unix socket as a single
// MCP server. The binary CLI wrapper (cmd/harness-mcp-gateway) is a thin shell
// around Daemon.Run; all gateway logic lives here so it stays testable without
// a process.
package mcpgateway

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"

	"harness/internal/mcp"
)

// FileConfig is the on-disk config shape. It is Claude Code-compatible
// (camelCase, "mcpServers") with a gateway-level extension block. Decode is
// tolerant (plain json.Unmarshal, unknown keys ignored) per repo convention.
type FileConfig struct {
	MCPServers map[string]ServerConfig `json:"mcpServers"`
	Gateway    GatewaySettings         `json:"gateway"`
}

// ServerConfig is one downstream server entry. Type "" or "stdio" selects a
// stdio child (Command/Args/Env); "http" selects a streamable-HTTP endpoint
// (URL/Headers). The two variants are mutually exclusive; validation enforces it.
type ServerConfig struct {
	Type    string            `json:"type"`
	Command string            `json:"command"`
	Args    []string          `json:"args"`
	Env     map[string]string `json:"env"`
	URL     string            `json:"url"`
	Headers map[string]string `json:"headers"`
}

// GatewaySettings carries gateway-level overrides. Empty fields fall back to
// defaults (socket → mcp.DefaultSocketPath, logLevel → info). An empty Listen
// disables the optional HTTP listener; only the unix socket is served.
type GatewaySettings struct {
	Socket   string `json:"socket"`
	Listen   string `json:"listen"`
	LogFile  string `json:"logFile"`
	LogLevel string `json:"logLevel"`
}

// Transport selects a resolved server's downstream transport.
type Transport int

const (
	// TransportStdio drives a child process over its stdin/stdout pipes.
	TransportStdio Transport = iota
	// TransportHTTP drives a streamable-HTTP endpoint.
	TransportHTTP
)

// ResolvedServer is a validated, ${VAR}-expanded server ready to supervise.
type ResolvedServer struct {
	Name      string
	Transport Transport
	// stdio
	Command string
	Args    []string
	Env     map[string]string
	// http
	URL     string
	Headers map[string]string
}

// Config is the resolved, validated gateway configuration. Servers is sorted by
// name for stable ordering. Warnings collects non-fatal load problems (unset
// expansion vars, skipped invalid servers); the caller logs them — library code
// never prints.
type Config struct {
	Servers  []ResolvedServer
	Socket   string
	Listen   string
	LogFile  string
	LogLevel string
	Warnings []string
}

// serverNameRE constrains a server name. It is the qualified-name charset, kept
// tight because the name becomes the middle segment of mcp__<server>__<tool>.
var serverNameRE = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// LoadConfig reads, expands, and validates the config at path. An empty path or
// a missing file at the DEFAULT location resolves to a valid empty config (zero
// servers). An explicitly-given path that is missing, or any present-but-
// malformed file, is an error. Invalid individual servers are skipped with a
// Warning, never fatal.
func LoadConfig(path string) (Config, error) {
	if path == "" {
		// No config requested: valid empty config with default socket.
		return resolve(FileConfig{}), nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			// An explicit path that does not exist is a hard error: a typo must not
			// silently degrade to "no servers". (A missing DEFAULT path is handled by
			// the caller passing "" instead.)
			return Config{}, fmt.Errorf("mcpgateway: config %s not found: %w", path, err)
		}
		return Config{}, fmt.Errorf("mcpgateway: read config %s: %w", path, err)
	}

	var fc FileConfig
	if err := json.Unmarshal(data, &fc); err != nil {
		return Config{}, fmt.Errorf("mcpgateway: parse config %s: %w", path, err)
	}
	return resolve(fc), nil
}

// resolve expands ${VAR} references, validates each server, and produces a
// sorted Config. Expansion happens before validation so a validated field is
// always the post-expansion value.
func resolve(fc FileConfig) Config {
	cfg := Config{
		LogFile:  fc.Gateway.LogFile,
		LogLevel: fc.Gateway.LogLevel,
	}

	// Expand variables across the whole config first, accumulating one warning
	// per distinct unset var.
	var exp expander
	expand := exp.expand

	// Sort names so validation/warning order is deterministic.
	names := make([]string, 0, len(fc.MCPServers))
	for name := range fc.MCPServers {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		raw := fc.MCPServers[name]
		rs, warn := resolveServer(name, raw, expand)
		if warn != "" {
			cfg.Warnings = append(cfg.Warnings, warn)
			continue
		}
		cfg.Servers = append(cfg.Servers, rs)
	}

	// Servers are already appended in name order (names is sorted), but sort
	// defensively so the contract holds regardless of append order.
	slices.SortFunc(cfg.Servers, func(a, b ResolvedServer) int {
		switch {
		case a.Name < b.Name:
			return -1
		case a.Name > b.Name:
			return 1
		default:
			return 0
		}
	})

	// Emit one warning per distinct unset var, in sorted order for determinism.
	unsetNames := exp.unsetNames()
	for _, v := range unsetNames {
		cfg.Warnings = append(cfg.Warnings, fmt.Sprintf("config references unset variable ${%s}; expanded to empty string", v))
	}

	cfg.Socket = fc.Gateway.Socket
	if cfg.Socket == "" {
		cfg.Socket = mcp.DefaultSocketPath(os.Getenv)
	}
	// Listen has no default: an empty value means "no HTTP listener". A non-empty
	// address is validated lazily by net.Listen at bind time, so config load does
	// not reject it here (a bad address surfaces as a Run error).
	cfg.Listen = fc.Gateway.Listen
	return cfg
}

// resolveServer expands and validates one entry. It returns either a resolved
// server (warn == "") or a non-empty warning describing why it was skipped.
func resolveServer(name string, sc ServerConfig, expand func(string) string) (ResolvedServer, string) {
	if !serverNameRE.MatchString(name) {
		return ResolvedServer{}, fmt.Sprintf("server %q skipped: name must match [a-zA-Z0-9_-]{1,64}", name)
	}

	command := expand(sc.Command)
	rawURL := expand(sc.URL)
	args := make([]string, len(sc.Args))
	for i, a := range sc.Args {
		args[i] = expand(a)
	}
	env := expandMap(sc.Env, expand)
	headers := expandMap(sc.Headers, expand)

	switch sc.Type {
	case "", "stdio":
		if command == "" {
			return ResolvedServer{}, fmt.Sprintf("server %q skipped: stdio server requires a command", name)
		}
		if rawURL != "" {
			return ResolvedServer{}, fmt.Sprintf("server %q skipped: stdio server must not set url", name)
		}
		return ResolvedServer{
			Name:      name,
			Transport: TransportStdio,
			Command:   command,
			Args:      args,
			Env:       env,
		}, ""
	case "http":
		if rawURL == "" {
			return ResolvedServer{}, fmt.Sprintf("server %q skipped: http server requires a url", name)
		}
		if command != "" {
			return ResolvedServer{}, fmt.Sprintf("server %q skipped: http server must not set command", name)
		}
		u, err := url.Parse(rawURL)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") {
			return ResolvedServer{}, fmt.Sprintf("server %q skipped: url must use http or https scheme", name)
		}
		return ResolvedServer{
			Name:      name,
			Transport: TransportHTTP,
			URL:       rawURL,
			Headers:   headers,
		}, ""
	default:
		return ResolvedServer{}, fmt.Sprintf("server %q skipped: unknown type %q (want \"\", \"stdio\", or \"http\")", name, sc.Type)
	}
}

// expandMap returns a copy of m with every value expanded; nil maps stay nil.
func expandMap(m map[string]string, expand func(string) string) map[string]string {
	if m == nil {
		return nil
	}
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = expand(v)
	}
	return out
}

// expander substitutes ${NAME} references in config strings, tracking which
// referenced vars were unset in the process environment so the loader can warn
// once per distinct name. An unset var expands to the empty string.
//
// Only the strict ${NAME} form (NAME matching [A-Za-z_][A-Za-z0-9_]*) is
// recognized, matching Claude Code's ${VAR} semantics. Every other '$' is
// passed through literally: a bare "$5", "price$5", "$$", or an unterminated
// "${" are all left exactly as written. This avoids os.Expand's behavior of
// eating "$$" and "$<digit>" tokens, which would corrupt secrets that contain
// '$'.
type expander struct {
	seen  map[string]bool
	names []string
}

func (e *expander) expand(s string) string {
	if !strings.ContainsRune(s, '$') {
		return s
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
		// Try to parse a ${NAME} reference starting at i. Anything that is not a
		// well-formed reference is emitted as a literal '$'.
		name, end, ok := parseVarRef(s, i)
		if !ok {
			b.WriteByte('$')
			i++
			continue
		}
		b.WriteString(e.lookup(name))
		i = end
	}
	return b.String()
}

// parseVarRef parses a ${NAME} reference at s[i] (s[i] is '$'). On success it
// returns the variable name, the index just past the closing brace, and true.
// A malformed reference (no brace, empty/invalid name, unterminated) returns
// ok=false so the caller emits a literal '$'.
func parseVarRef(s string, i int) (name string, end int, ok bool) {
	if i+1 >= len(s) || s[i+1] != '{' {
		return "", 0, false
	}
	j := i + 2
	start := j
	for j < len(s) && s[j] != '}' {
		j++
	}
	if j >= len(s) {
		return "", 0, false // unterminated "${..."
	}
	name = s[start:j]
	if !isVarName(name) {
		return "", 0, false
	}
	return name, j + 1, true
}

// isVarName reports whether name matches [A-Za-z_][A-Za-z0-9_]*.
func isVarName(name string) bool {
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

// lookup resolves a variable, recording an unset reference once per distinct
// name.
func (e *expander) lookup(name string) string {
	if val, ok := os.LookupEnv(name); ok {
		return val
	}
	if e.seen == nil {
		e.seen = make(map[string]bool)
	}
	if !e.seen[name] {
		e.seen[name] = true
		e.names = append(e.names, name)
	}
	return ""
}

// unsetNames returns the distinct unset variable names referenced during
// expansion, sorted for deterministic warning order.
func (e *expander) unsetNames() []string {
	out := slices.Clone(e.names)
	sort.Strings(out)
	return out
}

// ChildEnv builds the environment for a stdio child: the full parent
// environment with extra entries appended so they win on conflict (exec uses
// the last assignment of a key). A nil/empty extra map yields the parent
// environment unchanged.
func ChildEnv(extra map[string]string) []string {
	env := os.Environ()
	if len(extra) == 0 {
		return env
	}
	// Append in sorted key order for deterministic output (tests, logs).
	keys := make([]string, 0, len(extra))
	for k := range extra {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		env = append(env, k+"="+extra[k])
	}
	return env
}

// DefaultConfigPath resolves the default config file path:
// $XDG_CONFIG_HOME/harness-mcp-gateway/config.json, else
// ~/.config/harness-mcp-gateway/config.json. getenv injects the environment so
// the resolution is testable.
func DefaultConfigPath(getenv func(string) string) string {
	if xdg := getenv("XDG_CONFIG_HOME"); xdg != "" {
		return filepath.Join(xdg, "harness-mcp-gateway", "config.json")
	}
	home := getenv("HOME")
	return filepath.Join(home, ".config", "harness-mcp-gateway", "config.json")
}
