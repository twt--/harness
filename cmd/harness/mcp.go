package main

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"os"
	"slices"
	"sort"
	"strings"
	"time"

	"harness/internal/config"
	"harness/internal/logging"
	"harness/internal/mcp"
	"harness/internal/mcptools"
	"harness/internal/mode"
	"harness/internal/tools"
)

// MCP startup timing budget. The probe is a single short unix-socket dial so a
// dead socket fails fast; registration gets a larger budget.
const (
	mcpProbeTimeout    = 250 * time.Millisecond
	mcpRegisterTimeout = 5 * time.Second
)

// mcpDeps bundles the dial seam setupMCP depends on, so the liveness probe is
// injectable in tests. A zero mcpDeps is unusable; use defaultMCPDeps for
// production wiring.
type mcpDeps struct {
	dial func(network, addr string, timeout time.Duration) (net.Conn, error)
}

// defaultMCPDeps returns the production seams: a real unix dialer.
func defaultMCPDeps() mcpDeps {
	return mcpDeps{
		dial: net.DialTimeout,
	}
}

// setupMCP probes the already-running gateway, registers the discovered tools
// into catalog, and returns the live conn plus its initial registration summary
// and a cleanup func. It NEVER fails harness startup: if the gateway is
// unreachable or registration fails it logs a single warning via logger and
// returns ok=false with a nil conn and a no-op cleanup, so the caller can defer
// cleanup unconditionally. The harness does not start the gateway; that is the
// operator's job (run harness-mcp-gateway separately).
//
// The returned conn (when ok) backs the prompt-boundary refresh hook; cleanup
// closes that conn (the daemon itself keeps running and serving other sessions).
func setupMCP(ctx context.Context, mcpCfg config.MCPConfig, catalog *tools.Registry, logger *slog.Logger, deps mcpDeps) (conn *mcptools.Conn, summary mcptools.Summary, cleanup func(), ok bool) {
	noop := func() {}
	gateway := resolveMCPGateway(mcpCfg.Gateway)

	// Probe only the unix family with a short dial. An http(s) gateway is not
	// probed: an HTTP HEAD/GET would hit 405 (no GET stream in v1) and tell us
	// little, so we attempt NewConn+Register directly — the mcpRegisterTimeout
	// below bounds that attempt — and surface the same single warning on failure.
	if !isHTTPGateway(gateway) {
		if err := probeMCP(deps, gateway); err != nil {
			logger.Warn(fmt.Sprintf("mcp: cannot connect to gateway at %s: %v; MCP tools unavailable", gateway, err), logging.Category("mcp"))
			return nil, mcptools.Summary{}, noop, false
		}
	}

	c := mcptools.NewConn(mcptools.Options{
		Socket:  gateway,
		Headers: mcpCfg.Headers,
		Info:    mcp.Implementation{Name: "harness", Version: "dev"},
		Logger:  logger,
	})
	regCtx, cancel := context.WithTimeout(ctx, mcpRegisterTimeout)
	defer cancel()
	sum, err := mcptools.Register(regCtx, catalog, c)
	if err != nil {
		// For an unprobed http gateway, a register failure is the first sign the
		// gateway is unreachable: emit the same single "cannot connect" warning
		// shape so the operator sees the URL and the error. For the unix family
		// (already probed) this path is the rarer post-probe discovery failure.
		if isHTTPGateway(gateway) {
			logger.Warn(fmt.Sprintf("mcp: cannot connect to gateway at %s: %v; MCP tools unavailable", gateway, err), logging.Category("mcp"))
		} else {
			logger.Warn(fmt.Sprintf("mcp: tool discovery failed: %v; continuing without MCP tools", err), logging.Category("mcp"))
		}
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

// resolveMCPGateway turns the configured gateway value into a dialable address.
// An http(s) URL passes through verbatim (the http transport family). An empty
// value resolves to the shared default unix socket. Otherwise it is a unix path,
// with an optional unix:// URL prefix stripped (users sometimes sketch the
// socket as a URL).
func resolveMCPGateway(gateway string) string {
	if isHTTPGateway(gateway) {
		return gateway
	}
	gateway = strings.TrimPrefix(gateway, "unix://")
	if gateway == "" {
		return mcp.DefaultSocketPath(os.Getenv)
	}
	return gateway
}

// isHTTPGateway reports whether gateway is an http(s) URL (case-insensitive
// scheme). It must agree with mcptools' own URL detection so the probe decision
// here matches the transport family the Conn picks.
func isHTTPGateway(gateway string) bool {
	lower := strings.ToLower(gateway)
	return strings.HasPrefix(lower, "http://") || strings.HasPrefix(lower, "https://")
}

// probeMCP reports whether something is accepting on the socket: a short dial
// that is immediately closed on success. It returns the dial error on failure
// so the caller can surface why the gateway was unreachable.
func probeMCP(deps mcpDeps, socket string) error {
	conn, err := deps.dial("unix", socket, mcpProbeTimeout)
	if err != nil {
		return err
	}
	_ = conn.Close()
	return nil
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
		conn.ClearDirty()

		if _, ok := modes[modeName]; !ok {
			return nil, ""
		}

		// Worst case, a gateway that hangs mid-re-list stalls this prompt for up to
		// mcpRegisterTimeout (~5s) before the warn-and-keep path fires, since the
		// re-list runs synchronously at the prompt boundary. Accepted: it only
		// happens on a misbehaving gateway after an explicit list_changed, the
		// bound is finite, and the alternative (background re-list racing the
		// turn's Specs()/Dispatch reads) is the unsafe mid-turn swap we avoid.
		ctx, cancel := context.WithTimeout(context.Background(), mcpRegisterTimeout)
		defer cancel()
		sum, err := mcptools.Register(ctx, catalog, conn)
		if err != nil {
			logger.Warn(fmt.Sprintf("mcp: tool list refresh failed: %v; keeping current tools", err), logging.Category("mcp"))
			return nil, ""
		}

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
