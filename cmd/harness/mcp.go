package main

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"sort"
	"strings"
	"time"

	"harness/internal/config"
	"harness/internal/logging"
	"harness/internal/mcp"
	"harness/internal/mcpproxy"
	"harness/internal/mcptools"
	"harness/internal/mode"
	"harness/internal/tools"
)

// MCP startup timing budget.
const (
	mcpRegisterTimeout = 5 * time.Second
)

// setupMCP connects to the already-running HTTP proxy, registers the
// discovered tools into catalog, and returns the live conn plus its initial
// registration summary and a cleanup func. It NEVER fails harness startup: if
// the proxy is unreachable or registration fails it logs a single warning via
// logger and returns ok=false with a nil conn and a no-op cleanup, so the caller
// can defer cleanup unconditionally. The harness does not start the proxy;
// that is the operator's job (run harness-mcp-proxy separately).
//
// The returned conn (when ok) backs tool dispatch; cleanup closes that conn (the
// daemon itself keeps running and serving other sessions).
func setupMCP(ctx context.Context, mcpCfg config.MCPConfig, catalog *tools.Registry, logger *slog.Logger) (conn *mcptools.Conn, summary mcptools.Summary, cleanup func(), ok bool) {
	noop := func() {}
	proxy := resolveMCPProxy(mcpCfg.Proxy)
	if !isHTTPProxy(proxy) {
		logger.Warn(fmt.Sprintf("mcp: cannot connect to proxy at %s: proxy must be an http(s) URL; MCP tools unavailable", proxy), logging.Category("mcp"))
		return nil, mcptools.Summary{}, noop, false
	}

	c := mcptools.NewConn(mcptools.Options{
		Endpoint: proxy,
		Headers:  mcpCfg.Headers,
		Info:     mcp.Implementation{Name: "harness", Version: "dev"},
		Logger:   logger,
	})
	regCtx, cancel := context.WithTimeout(ctx, mcpRegisterTimeout)
	defer cancel()
	sum, err := mcptools.Register(regCtx, catalog, c)
	if err != nil {
		logger.Warn(fmt.Sprintf("mcp: cannot connect to proxy at %s: %v; MCP tools unavailable", proxy, err), logging.Category("mcp"))
		_ = c.Close()
		return nil, mcptools.Summary{}, noop, false
	}

	logger.Info(mcpConnectedLine(sum), logging.Category("mcp"))
	for _, name := range sum.Skipped {
		logger.Warn(fmt.Sprintf("mcp: skipping tool %q: name must match [a-zA-Z0-9_-]{1,64}", name), logging.Category("mcp"))
	}
	return c, sum, func() { _ = c.Close() }, true
}

// augmentModesWithMCP appends the discovered MCP tool names to every mode whose
// allowed-tool set is the inherited default (auto, independent, and any config
// mode without an explicit allowed_tools whitelist). Modes with an explicit
// whitelist are left untouched, so they opt out of MCP tools by construction —
// matching the design's mode/allowed_tools contract. It is a no-op when there
// are no MCP names.
//
// Classification is slices.Equal against mode.DefaultTools(): a config mode that
// explicitly lists exactly the default tools in default order is indistinguishable
// from an inheriting one and is treated as default-inheriting, so it gains MCP
// tools. This edge is benign and accepted (such a mode wanted the full default
// set anyway).
func augmentModesWithMCP(modes map[string]mode.Mode, mcpNames []string) {
	if len(mcpNames) == 0 {
		return
	}
	def := mode.DefaultTools()
	for name, m := range modes {
		if slices.Equal(m.AllowedTools, def) {
			m.AllowedTools = append(slices.Clone(m.AllowedTools), mcpNames...)
			modes[name] = m
		}
	}
}

// resolveMCPProxy turns the configured proxy value into a dialable HTTP URL.
// An empty value resolves to the shared default proxy URL.
func resolveMCPProxy(proxy string) string {
	if proxy == "" {
		return mcpproxy.DefaultURL()
	}
	return proxy
}

// isHTTPProxy reports whether proxy is an http(s) URL (case-insensitive
// scheme).
func isHTTPProxy(proxy string) bool {
	lower := strings.ToLower(proxy)
	return strings.HasPrefix(lower, "http://") || strings.HasPrefix(lower, "https://")
}

// mcpModeBases is the per-mode base allowed-tool list for every default-
// inheriting mode (the modes that expose MCP tools). A mode absent from the map
// is an explicit whitelist that opts out of MCP tools. Built once at startup
// from the resolved modes, it lets a refresh re-derive each such mode's full
// allowed list (base ∪ live MCP names) without re-classifying.
type mcpModeBases map[string][]string

// defaultInheritingModeBases returns, for each mode whose allowed-tool set is
// the inherited default, its base list (a clone of the default set). It must be
// called on the modes BEFORE augmentModesWithMCP mutates them.
func defaultInheritingModeBases(modes map[string]mode.Mode) mcpModeBases {
	def := mode.DefaultTools()
	bases := make(mcpModeBases)
	for name, m := range modes {
		if slices.Equal(m.AllowedTools, def) {
			bases[name] = slices.Clone(def)
		}
	}
	return bases
}

// newMCPRefresher returns the prompt-boundary refresh hook for ui.App. It owns
// the conn, the tool catalog, the resolved modes, and the previous
// registration's tool names so it can compute which tools vanished. On a
// list_changed it re-lists, removes departed tools from the catalog, re-derives
// every MCP-exposing mode's allowed list (so a later /mode switch stays valid),
// and returns the current mode's subset. It returns a nil registry ("no
// change") fast when nothing changed, and on a re-discovery error keeps the
// current tools. Not safe for concurrent use: the REPL calls it only at the
// idle prompt boundary, between turns.
func newMCPRefresher(conn *mcptools.Conn, catalog *tools.Registry, modes map[string]mode.Mode, bases mcpModeBases, prev mcptools.Summary, logger *slog.Logger) func(modeName string) (*tools.Registry, string) {
	prevNames := prev.Names
	return func(modeName string) (*tools.Registry, string) {
		if !conn.Dirty() {
			return nil, ""
		}

		if _, ok := modes[modeName]; !ok {
			return nil, ""
		}

		// Worst case, a proxy that hangs mid-re-list stalls this prompt for up to
		// mcpRegisterTimeout (~5s) before the warn-and-keep path fires, since the
		// re-list runs synchronously at the prompt boundary. Accepted: it only
		// happens on a misbehaving proxy after an explicit list_changed, the
		// bound is finite, and the alternative (background re-list racing the
		// turn's Specs()/Dispatch reads) is the unsafe mid-turn swap we avoid.
		ctx, cancel := context.WithTimeout(context.Background(), mcpRegisterTimeout)
		defer cancel()
		sum, err := mcptools.Register(ctx, catalog, conn)
		if err != nil {
			logger.Warn(fmt.Sprintf("mcp: tool list refresh failed: %v; keeping current tools", err), logging.Category("mcp"))
			return nil, ""
		}
		conn.ClearDirty()

		// Drop tools that were registered before but are gone now. Register
		// replaces survivors in place; only departures need explicit removal.
		removed := removedNames(prevNames, sum.Names)
		for _, name := range removed {
			catalog.Remove(name)
		}
		prevNames = sum.Names

		// Re-derive every MCP-exposing mode's allowed list against the live tool
		// set, so /mode switches after a tool vanishes never reference a name the
		// catalog no longer has.
		for name, base := range bases {
			m := modes[name]
			m.AllowedTools = append(slices.Clone(base), sum.Names...)
			modes[name] = m
		}

		// An explicit-whitelist mode (one not in bases) exposes no MCP tools, so a
		// refresh leaves its subset unchanged — unless it explicitly whitelisted a
		// tool that was just removed. In the unchanged case, skip the swap and the
		// "tool list updated" notice, which would otherwise mislead (the mode's
		// tools did not change). The catalog/mode re-derivation above still ran so
		// a later /mode switch to an MCP-exposing mode is correct.
		allowed := modes[modeName].AllowedTools
		if _, exposesMCP := bases[modeName]; !exposesMCP {
			if !anyRemovedInMode(allowed, removed) {
				return nil, ""
			}
			// The whitelist named a removed tool: drop the gone names so Subset
			// does not error on a name the catalog no longer has.
			allowed = withoutNames(allowed, removed)
			m := modes[modeName]
			m.AllowedTools = allowed
			modes[modeName] = m
		}

		sel, err := catalog.Subset(allowed)
		if err != nil {
			logger.Warn(fmt.Sprintf("mcp: tool list refresh subset failed: %v; keeping current tools", err), logging.Category("mcp"))
			return nil, ""
		}
		return sel, fmt.Sprintf("[mcp: tool list updated; %d tools]", sum.Total)
	}
}

// anyRemovedInMode reports whether allowed references any of the removed tool
// names, i.e. whether the refresh shrank a mode's effective tool set.
func anyRemovedInMode(allowed, removed []string) bool {
	for _, name := range removed {
		if slices.Contains(allowed, name) {
			return true
		}
	}
	return false
}

// withoutNames returns allowed with every entry in drop removed, preserving
// order. It is used to drop just-removed MCP tool names from a whitelist mode's
// allowed list so Subset does not error on a name the catalog no longer has.
func withoutNames(allowed, drop []string) []string {
	out := make([]string, 0, len(allowed))
	for _, name := range allowed {
		if !slices.Contains(drop, name) {
			out = append(out, name)
		}
	}
	return out
}

// removedNames returns the entries of prev that are absent from next, preserving
// prev's order.
func removedNames(prev, next []string) []string {
	keep := make(map[string]bool, len(next))
	for _, n := range next {
		keep[n] = true
	}
	var gone []string
	for _, n := range prev {
		if !keep[n] {
			gone = append(gone, n)
		}
	}
	return gone
}

// mcpConnectedLine renders the one-line success notice, e.g.
// "mcp: connected (2 servers, 5 tools): a=3 b=2" with servers sorted by name.
func mcpConnectedLine(sum mcptools.Summary) string {
	names := make([]string, 0, len(sum.Servers))
	for name := range sum.Servers {
		names = append(names, name)
	}
	sort.Strings(names)
	parts := make([]string, len(names))
	for i, name := range names {
		parts[i] = fmt.Sprintf("%s=%d", name, sum.Servers[name])
	}
	line := fmt.Sprintf("mcp: connected (%d servers, %d tools)", len(names), sum.Total)
	if len(parts) > 0 {
		line += ": " + strings.Join(parts, " ")
	}
	return line
}
