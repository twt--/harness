package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"harness/internal/config"
	"harness/internal/llm"
	"harness/internal/llm/llmtest"
	"harness/internal/logging"
	"harness/internal/mcp"
	"harness/internal/mcpgateway"
	"harness/internal/mcptools"
	"harness/internal/mode"
	"harness/internal/tools"
	"harness/internal/ui"
)

// echoProvider is an mcp.ToolProvider that advertises a mutable tool list and
// echoes a tool call's arguments back as a text block.
type echoProvider struct {
	mu    sync.Mutex
	tools []mcp.Tool
}

func (p *echoProvider) ListTools(ctx context.Context, cursor string) (mcp.ListToolsResult, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return mcp.ListToolsResult{Tools: append([]mcp.Tool(nil), p.tools...)}, nil
}

func (p *echoProvider) CallTool(ctx context.Context, name string, args json.RawMessage) (*mcp.CallToolResult, error) {
	return &mcp.CallToolResult{
		Content: []mcp.ContentBlock{{Type: "text", Text: "echo: " + string(args)}},
	}, nil
}

func (p *echoProvider) setTools(t []mcp.Tool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.tools = t
}

type flakyListProvider struct {
	mu       sync.Mutex
	tools    []mcp.Tool
	failNext bool
}

func (p *flakyListProvider) ListTools(ctx context.Context, cursor string) (mcp.ListToolsResult, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.failNext {
		p.failNext = false
		return mcp.ListToolsResult{}, errors.New("temporary list failure")
	}
	return mcp.ListToolsResult{Tools: append([]mcp.Tool(nil), p.tools...)}, nil
}

func (p *flakyListProvider) CallTool(ctx context.Context, name string, args json.RawMessage) (*mcp.CallToolResult, error) {
	return &mcp.CallToolResult{Content: []mcp.ContentBlock{{Type: "text", Text: string(args)}}}, nil
}

func (p *flakyListProvider) setTools(tools []mcp.Tool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.tools = tools
}

func (p *flakyListProvider) failOnce() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.failNext = true
}

func echoTool() mcp.Tool {
	return mcp.Tool{
		Name:        "mcp__test__echo",
		Description: "echoes its arguments",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}}}`),
	}
}

func mcpTool(name string) mcp.Tool {
	return mcp.Tool{Name: name, InputSchema: json.RawMessage(`{"type":"object"}`)}
}

// fakeGateway is a stream MCP server backed by provider, already accepting
// local test connections and serving one mcp.Serve session each. It captures the
// most recent session so a test can fire tools/list_changed.
type fakeGateway struct {
	path     string
	provider mcp.ToolProvider
	ln       net.Listener

	mu      sync.Mutex
	session *mcp.ServerSession
}

// start begins accepting connections, serving one mcp.Serve session each.
func (g *fakeGateway) start() {
	ln, err := net.Listen("unix", g.path)
	if err != nil {
		panic("fakeGateway listen: " + err.Error())
	}
	g.ln = ln
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func() {
				_ = mcp.Serve(context.Background(), conn, mcp.ServerOptions{
					Info:        mcp.Implementation{Name: "fake-gateway", Version: "test"},
					Provider:    g.provider,
					ListChanged: true,
					OnSession: func(s *mcp.ServerSession) {
						g.mu.Lock()
						g.session = s
						g.mu.Unlock()
					},
				})
			}()
		}
	}()
}

func (g *fakeGateway) close() {
	if g.ln != nil {
		_ = g.ln.Close()
	}
}

func (g *fakeGateway) dial(ctx context.Context) (io.ReadWriteCloser, error) {
	var d net.Dialer
	return d.DialContext(ctx, "unix", g.path)
}

// startFakeGateway returns a gateway already accepting connections on a unix
// socket under a short temp dir (sun_path length — t.TempDir nests too deep).
func startFakeGateway(t *testing.T, provider mcp.ToolProvider) *fakeGateway {
	t.Helper()
	dir, err := os.MkdirTemp("", "hmg-harness")
	if err != nil {
		t.Fatalf("mkdtemp: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	g := &fakeGateway{path: filepath.Join(dir, "gw.sock"), provider: provider}
	g.start()
	t.Cleanup(g.close)
	return g
}

// notifyListChanged fires tools/list_changed on the current session and waits
// for conn to observe it (the notification crosses the socket on the client's
// read goroutine), so the dirty flag is deterministically set before return.
func (g *fakeGateway) notifyListChanged(t *testing.T, conn *mcptools.Conn) {
	t.Helper()
	g.mu.Lock()
	s := g.session
	g.mu.Unlock()
	if s == nil {
		t.Fatal("no live gateway session to notify")
	}
	if err := s.NotifyToolsListChanged(); err != nil {
		t.Fatalf("NotifyToolsListChanged: %v", err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if conn.Dirty() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("conn never observed list_changed")
}

func mcpToolCallStep(id, name, args string) llmtest.Step {
	return llmtest.Step{
		Events: []llm.StreamEvent{{
			Kind:      llm.EventToolCallDone,
			Index:     0,
			ToolID:    id,
			ToolName:  name,
			ToolInput: json.RawMessage(args),
		}},
		Stop: llm.StopToolUse,
	}
}

// TestSetupMCPRegistersToolsAndOneShotCalls drives a full one-shot run with MCP
// enabled against a real HTTP MCP gateway handler: the scripted model calls
// mcp__test__echo, and the echoed result must flow back into the next request's
// transcript.
func TestSetupMCPRegistersToolsAndOneShotCalls(t *testing.T) {
	url, _ := startHTTPGateway(t, &echoProvider{tools: []mcp.Tool{echoTool()}})

	fp := llmtest.New("fake",
		mcpToolCallStep("call_1", "mcp__test__echo", `{"text":"hi"}`),
		llmtest.Step{
			Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "done echoing"}},
			Stop:   llm.StopEndTurn,
		},
	)

	env, out, errw, _ := fakeProviderEnv(t, []string{"-model", "claude-opus-4-8", "-p", "echo it"}, fp, "")
	env.getenv = withMCPEnv(env.getenv, url)

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("run exit = %d, want 0; errw=%q", code, errw.String())
	}
	if !strings.Contains(out.String(), "done echoing") {
		t.Errorf("assistant text missing from stdout: %q", out.String())
	}
	if len(fp.Requests) < 2 {
		t.Fatalf("want >=2 requests (tool round-trip), got %d", len(fp.Requests))
	}
	if !transcriptHasToolResult(fp.Requests[1].Messages, "echo:") {
		t.Errorf("second request missing echo tool result")
	}
	if !strings.Contains(errw.String(), "mcp: connected") {
		t.Errorf("expected mcp connected notice on stderr, got %q", errw.String())
	}
}

// TestSetupMCPRejectsNonHTTPGatewayAndContinues enables MCP with a non-HTTP
// gateway value. Startup must proceed, emit a single [warn] [mcp] line naming
// the bad gateway, register zero mcp__ tools, and return a no-op cleanup.
func TestSetupMCPRejectsNonHTTPGatewayAndContinues(t *testing.T) {
	catalog := tools.Catalog()
	before := len(catalog.Names())

	var errw strings.Builder
	logger, err := logging.NewLogger(&errw, logging.LevelInfo, false)
	if err != nil {
		t.Fatal(err)
	}
	conn, _, cleanup, ok := setupMCP(context.Background(), config.MCPConfig{Enable: true, Gateway: "/no/such.sock"}, catalog, logger)
	defer cleanup()

	if ok || conn != nil {
		t.Fatalf("setupMCP should fail soft: ok=%v conn=%v", ok, conn != nil)
	}
	if got := len(catalog.Names()); got != before {
		t.Errorf("catalog grew from %d to %d; no MCP tools should register", before, got)
	}
	if !strings.Contains(errw.String(), "[warn]") || !strings.Contains(errw.String(), "[mcp]") {
		t.Errorf("expected [warn] [mcp] line, got %q", errw.String())
	}
	if !strings.Contains(errw.String(), "/no/such.sock") || !strings.Contains(errw.String(), "http(s) URL") {
		t.Errorf("warning should name the invalid gateway, got %q", errw.String())
	}
	if strings.Count(errw.String(), "[warn]") != 1 {
		t.Errorf("expected exactly one warning, got %q", errw.String())
	}
	for _, n := range catalog.Names() {
		if strings.HasPrefix(n, "mcp__") {
			t.Errorf("unexpected mcp tool registered: %q", n)
		}
	}
}

// httpAuthMiddleware wraps an mcp HTTP handler, counting requests and recording
// the Authorization header seen on each, so a test can assert the configured
// header reached the gateway on every request.
type httpAuthMiddleware struct {
	next     http.Handler
	mu       sync.Mutex
	auths    []string
	requests int
}

func (m *httpAuthMiddleware) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	m.mu.Lock()
	m.requests++
	m.auths = append(m.auths, r.Header.Get("Authorization"))
	m.mu.Unlock()
	m.next.ServeHTTP(w, r)
}

func (m *httpAuthMiddleware) allHadAuth(want string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.requests == 0 {
		return false
	}
	for _, a := range m.auths {
		if a != want {
			return false
		}
	}
	return true
}

// startHTTPGateway starts an httptest server running a real mcp.NewHTTPHandler
// (the streamable-HTTP server) over provider, wrapped in an auth-counting
// middleware. It returns the bound URL and the middleware for assertions.
func startHTTPGateway(t *testing.T, provider mcp.ToolProvider) (string, *httpAuthMiddleware) {
	t.Helper()
	handler := mcp.NewHTTPHandler(mcp.HTTPHandlerOptions{
		Info:     mcp.Implementation{Name: "test-http-gateway", Version: "test"},
		Provider: provider,
	})
	mw := &httpAuthMiddleware{next: handler}
	srv := httptest.NewServer(mw)
	t.Cleanup(srv.Close)
	return srv.URL, mw
}

// writeHarnessConfig writes a harness config.json carrying an mcp block (gateway
// URL + headers, which are config-file-only) and returns its path. Headers have
// no env var, so a header-bearing integration test must drive them through a
// file.
func writeHarnessConfig(t *testing.T, gatewayURL, authHeader string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "config.json")
	body := fmt.Sprintf(`{"mcp":{"enable":true,"gateway":%q,"headers":{"Authorization":%q}}}`, gatewayURL, authHeader)
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	return p
}

// TestSetupMCPHTTPGatewayRoundTrip drives a full one-shot run with MCP enabled
// against a REAL streamable-HTTP gateway: the gateway URL and an Authorization
// header come from a config file (headers are config-file-only). The scripted
// model calls mcp__test__echo, the echoed result must flow back into the next
// request's transcript, and the configured header must have reached the gateway
// on every request.
func TestSetupMCPHTTPGatewayRoundTrip(t *testing.T) {
	url, mw := startHTTPGateway(t, &echoProvider{tools: []mcp.Tool{echoTool()}})
	cfgPath := writeHarnessConfig(t, url, "Bearer secret-tok")

	fp := llmtest.New("fake",
		mcpToolCallStep("call_1", "mcp__test__echo", `{"text":"hi"}`),
		llmtest.Step{
			Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "done echoing"}},
			Stop:   llm.StopEndTurn,
		},
	)

	args := []string{"-config", cfgPath, "-model", "claude-opus-4-8", "-p", "echo it"}
	env, out, errw, _ := fakeProviderEnv(t, args, fp, "")

	if code := run(env); code != ui.ExitOK {
		t.Fatalf("run exit = %d, want 0; errw=%q", code, errw.String())
	}
	if !strings.Contains(out.String(), "done echoing") {
		t.Errorf("assistant text missing from stdout: %q", out.String())
	}
	if len(fp.Requests) < 2 {
		t.Fatalf("want >=2 requests (tool round-trip), got %d", len(fp.Requests))
	}
	if !transcriptHasToolResult(fp.Requests[1].Messages, "echo:") {
		t.Errorf("second request missing echo tool result")
	}
	if !strings.Contains(errw.String(), "mcp: connected") {
		t.Errorf("expected mcp connected notice on stderr, got %q", errw.String())
	}
	if !mw.allHadAuth("Bearer secret-tok") {
		t.Errorf("Authorization header did not reach the gateway on every request")
	}
}

// TestSetupMCPHTTPUnreachableWarnsAndContinues points the gateway URL at a
// closed port: setup must fail soft on the Register attempt, emit a single
// MCP-unavailable warning naming the URL, and let the run continue.
func TestSetupMCPHTTPUnreachableWarnsAndContinues(t *testing.T) {
	// Bind then immediately close a listener to obtain a definitely-dead URL.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	deadURL := "http://" + ln.Addr().String()
	_ = ln.Close()

	catalog := tools.Catalog()
	before := len(catalog.Names())

	var errw strings.Builder
	logger, err := logging.NewLogger(&errw, logging.LevelInfo, false)
	if err != nil {
		t.Fatal(err)
	}
	conn, _, cleanup, ok := setupMCP(context.Background(), config.MCPConfig{Enable: true, Gateway: deadURL}, catalog, logger)
	defer cleanup()

	if ok || conn != nil {
		t.Fatalf("setupMCP should fail soft: ok=%v conn=%v", ok, conn != nil)
	}
	if got := len(catalog.Names()); got != before {
		t.Errorf("catalog grew from %d to %d; no MCP tools should register", before, got)
	}
	if !strings.Contains(errw.String(), "cannot connect to gateway at "+deadURL) {
		t.Errorf("warning should name the dead URL, got %q", errw.String())
	}
	if strings.Count(errw.String(), "[warn]") != 1 {
		t.Errorf("expected exactly one warning, got %q", errw.String())
	}
}

// TestResolveMCPGateway verifies an empty value resolves to the shared default
// HTTP URL and an http(s) URL passes through verbatim.
func TestResolveMCPGateway(t *testing.T) {
	if got := resolveMCPGateway(""); got != mcpgateway.DefaultURL() {
		t.Errorf("resolveMCPGateway(\"\") = %q, want %q", got, mcpgateway.DefaultURL())
	}
	for _, url := range []string{"http://127.0.0.1:8080/mcp", "https://gw.example/mcp", "HTTP://up.example"} {
		if got := resolveMCPGateway(url); got != url {
			t.Errorf("resolveMCPGateway(%q) = %q, want verbatim pass-through", url, got)
		}
	}
}

// TestMCPConnectedLine renders the success notice with sorted servers.
func TestMCPConnectedLine(t *testing.T) {
	sum := mcptools.Summary{Servers: map[string]int{"b": 2, "a": 3}, Total: 5}
	if got, want := mcpConnectedLine(sum), "mcp: connected (2 servers, 5 tools): a=3 b=2"; got != want {
		t.Errorf("mcpConnectedLine = %q, want %q", got, want)
	}
	if got := mcpConnectedLine(mcptools.Summary{Servers: map[string]int{}}); got != "mcp: connected (0 servers, 0 tools)" {
		t.Errorf("empty summary line = %q", got)
	}
}

// TestMCPRefresherAddsAndRemovesTools drives newMCPRefresher across a
// list_changed: the gateway swaps one tool for another, and the returned subset
// must reflect the addition and removal with the correct notice.
func TestMCPRefresherAddsAndRemovesTools(t *testing.T) {
	provider := &echoProvider{tools: []mcp.Tool{mcpTool("mcp__test__alpha"), mcpTool("mcp__test__beta")}}
	g := startFakeGateway(t, provider)

	catalog := tools.Catalog()
	conn := mcptools.NewConn(mcptools.Options{Dial: g.dial, Info: mcp.Implementation{Name: "harness", Version: "test"}})
	defer conn.Close()

	initial, err := mcptools.Register(context.Background(), catalog, conn)
	if err != nil {
		t.Fatalf("initial register: %v", err)
	}
	if !slices.Contains(catalog.Names(), "mcp__test__alpha") || !slices.Contains(catalog.Names(), "mcp__test__beta") {
		t.Fatalf("initial tools missing: %v", catalog.Names())
	}

	// A default-inheriting mode "auto": base is a built-in, so the refresher
	// re-unions the live MCP names (alpha + gamma after the swap).
	modes := map[string]mode.Mode{
		"auto": {Name: "auto", AllowedTools: []string{"read_file", "mcp__test__alpha", "mcp__test__beta"}},
	}
	bases := mcpModeBases{"auto": {"read_file"}}
	refresh := newMCPRefresher(conn, catalog, modes, bases, initial, slog.New(slog.DiscardHandler))

	// No change yet: not dirty.
	if sel, notice := refresh("auto"); sel != nil || notice != "" {
		t.Fatalf("refresh before dirty should be a no-op, got sel=%v notice=%q", sel != nil, notice)
	}

	// Swap beta for gamma and fire list_changed.
	provider.setTools([]mcp.Tool{mcpTool("mcp__test__alpha"), mcpTool("mcp__test__gamma")})
	g.notifyListChanged(t, conn)

	sel, notice := refresh("auto")
	if sel == nil {
		t.Fatalf("refresh after dirty returned nil registry")
	}
	names := sel.Names()
	if !slices.Contains(names, "mcp__test__gamma") {
		t.Errorf("added tool gamma missing from subset: %v", names)
	}
	if slices.Contains(names, "mcp__test__beta") {
		t.Errorf("removed tool beta still present in subset: %v", names)
	}
	if slices.Contains(catalog.Names(), "mcp__test__beta") {
		t.Errorf("removed tool beta still present in catalog: %v", catalog.Names())
	}
	if !strings.Contains(notice, "tool list updated") {
		t.Errorf("notice = %q, want refresh notice", notice)
	}
	// The mode's allowed list was re-derived so a later /mode Subset stays valid:
	// beta gone, gamma present.
	if slices.Contains(modes["auto"].AllowedTools, "mcp__test__beta") {
		t.Errorf("mode allowed list still references removed beta: %v", modes["auto"].AllowedTools)
	}
	if !slices.Contains(modes["auto"].AllowedTools, "mcp__test__gamma") {
		t.Errorf("mode allowed list missing added gamma: %v", modes["auto"].AllowedTools)
	}
}

// TestMCPRefresherSkipsUnaffectedWhitelistMode confirms that when the current
// mode is an explicit whitelist that exposes no MCP tools, a list_changed does
// not produce a (misleading) swap or notice — yet the catalog and MCP-exposing
// modes are still re-derived so a later /mode switch stays valid.
func TestMCPRefresherSkipsUnaffectedWhitelistMode(t *testing.T) {
	provider := &echoProvider{tools: []mcp.Tool{mcpTool("mcp__test__alpha"), mcpTool("mcp__test__beta")}}
	g := startFakeGateway(t, provider)

	catalog := tools.Catalog()
	conn := mcptools.NewConn(mcptools.Options{Dial: g.dial, Info: mcp.Implementation{Name: "harness", Version: "test"}})
	defer conn.Close()
	initial, err := mcptools.Register(context.Background(), catalog, conn)
	if err != nil {
		t.Fatalf("initial register: %v", err)
	}

	// "locked" is the current mode: an explicit whitelist of built-ins only, not
	// in bases. "auto" is a default-inheriting mode in bases.
	modes := map[string]mode.Mode{
		"locked": {Name: "locked", AllowedTools: []string{"read_file", "grep"}},
		"auto":   {Name: "auto", AllowedTools: []string{"read_file", "mcp__test__alpha", "mcp__test__beta"}},
	}
	bases := mcpModeBases{"auto": {"read_file"}}
	refresh := newMCPRefresher(conn, catalog, modes, bases, initial, slog.New(slog.DiscardHandler))

	// Swap beta for gamma and fire list_changed.
	provider.setTools([]mcp.Tool{mcpTool("mcp__test__alpha"), mcpTool("mcp__test__gamma")})
	g.notifyListChanged(t, conn)

	// Current mode is the unaffected whitelist: no swap, no notice.
	if sel, notice := refresh("locked"); sel != nil || notice != "" {
		t.Fatalf("whitelist mode refresh should be a silent no-op, got sel=%v notice=%q", sel != nil, notice)
	}
	// Side effects must still have happened: catalog dropped beta, auto re-derived.
	if slices.Contains(catalog.Names(), "mcp__test__beta") {
		t.Errorf("removed tool beta still in catalog: %v", catalog.Names())
	}
	if !slices.Contains(modes["auto"].AllowedTools, "mcp__test__gamma") || slices.Contains(modes["auto"].AllowedTools, "mcp__test__beta") {
		t.Errorf("auto mode not re-derived: %v", modes["auto"].AllowedTools)
	}
}

// TestMCPRefresherSwapsWhitelistModeLosingTool confirms that a whitelist mode
// that explicitly named a now-removed MCP tool DOES get a swap + notice (its
// effective tool set shrank).
func TestMCPRefresherSwapsWhitelistModeLosingTool(t *testing.T) {
	provider := &echoProvider{tools: []mcp.Tool{mcpTool("mcp__test__alpha"), mcpTool("mcp__test__beta")}}
	g := startFakeGateway(t, provider)

	catalog := tools.Catalog()
	conn := mcptools.NewConn(mcptools.Options{Dial: g.dial, Info: mcp.Implementation{Name: "harness", Version: "test"}})
	defer conn.Close()
	initial, err := mcptools.Register(context.Background(), catalog, conn)
	if err != nil {
		t.Fatalf("initial register: %v", err)
	}

	// "locked" explicitly whitelists mcp__test__beta, which is about to vanish.
	modes := map[string]mode.Mode{
		"locked": {Name: "locked", AllowedTools: []string{"read_file", "mcp__test__beta"}},
	}
	refresh := newMCPRefresher(conn, catalog, modes, mcpModeBases{}, initial, slog.New(slog.DiscardHandler))

	provider.setTools([]mcp.Tool{mcpTool("mcp__test__alpha")}) // beta removed
	g.notifyListChanged(t, conn)

	sel, notice := refresh("locked")
	if sel == nil {
		t.Fatalf("whitelist mode losing a tool should swap, got nil registry")
	}
	if slices.Contains(sel.Names(), "mcp__test__beta") {
		t.Errorf("removed beta still in subset: %v", sel.Names())
	}
	if slices.Contains(modes["locked"].AllowedTools, "mcp__test__beta") {
		t.Errorf("removed beta still persisted in whitelist mode: %v", modes["locked"].AllowedTools)
	}
	if !strings.Contains(notice, "tool list updated") {
		t.Errorf("notice = %q, want refresh notice", notice)
	}
}

func TestMCPRefresherFailedRefreshKeepsDirtyForRetry(t *testing.T) {
	provider := &flakyListProvider{tools: []mcp.Tool{mcpTool("mcp__test__alpha")}}
	g := startFakeGateway(t, provider)

	catalog := tools.Catalog()
	conn := mcptools.NewConn(mcptools.Options{Dial: g.dial, Info: mcp.Implementation{Name: "harness", Version: "test"}})
	defer conn.Close()
	initial, err := mcptools.Register(context.Background(), catalog, conn)
	if err != nil {
		t.Fatalf("initial register: %v", err)
	}

	modes := map[string]mode.Mode{
		"auto": {Name: "auto", AllowedTools: []string{"read_file", "mcp__test__alpha"}},
	}
	bases := mcpModeBases{"auto": {"read_file"}}
	refresh := newMCPRefresher(conn, catalog, modes, bases, initial, slog.New(slog.DiscardHandler))

	provider.setTools([]mcp.Tool{mcpTool("mcp__test__beta")})
	provider.failOnce()
	g.notifyListChanged(t, conn)

	if sel, notice := refresh("auto"); sel != nil || notice != "" {
		t.Fatalf("failed refresh should keep existing registry, got sel=%v notice=%q", sel != nil, notice)
	}
	if !conn.Dirty() {
		t.Fatalf("dirty flag should remain set after failed refresh")
	}

	sel, notice := refresh("auto")
	if sel == nil {
		t.Fatalf("second refresh should retry and succeed")
	}
	if !slices.Contains(sel.Names(), "mcp__test__beta") {
		t.Fatalf("retried refresh missing beta: %v", sel.Names())
	}
	if !strings.Contains(notice, "tool list updated") {
		t.Fatalf("notice = %q, want update notice", notice)
	}
	if conn.Dirty() {
		t.Fatalf("dirty flag should clear after successful refresh")
	}
}

// TestMCPRefresherNotDirtyFastPath confirms a clean conn returns nil without
// re-listing.
func TestMCPRefresherNotDirtyFastPath(t *testing.T) {
	g := startFakeGateway(t, &echoProvider{tools: []mcp.Tool{echoTool()}})
	catalog := tools.Catalog()
	conn := mcptools.NewConn(mcptools.Options{Dial: g.dial, Info: mcp.Implementation{Name: "harness", Version: "test"}})
	defer conn.Close()
	sum, err := mcptools.Register(context.Background(), catalog, conn)
	if err != nil {
		t.Fatal(err)
	}
	modes := map[string]mode.Mode{"auto": {Name: "auto", AllowedTools: catalog.Names()}}
	refresh := newMCPRefresher(conn, catalog, modes, mcpModeBases{}, sum, slog.New(slog.DiscardHandler))
	if sel, notice := refresh("auto"); sel != nil || notice != "" {
		t.Fatalf("clean conn should yield no change, got sel=%v notice=%q", sel != nil, notice)
	}
}

// TestAugmentModesWithMCP confirms default-inheriting modes gain the MCP tool
// names while explicit-whitelist modes are left untouched.
func TestAugmentModesWithMCP(t *testing.T) {
	def := mode.DefaultTools()
	modes := map[string]mode.Mode{
		"auto":   {Name: "auto", AllowedTools: slices.Clone(def)},
		"locked": {Name: "locked", AllowedTools: []string{"read_file", "grep"}},
	}
	augmentModesWithMCP(modes, []string{"mcp__test__echo"})

	if !slices.Contains(modes["auto"].AllowedTools, "mcp__test__echo") {
		t.Errorf("auto mode should gain mcp tool, got %v", modes["auto"].AllowedTools)
	}
	if slices.Contains(modes["locked"].AllowedTools, "mcp__test__echo") {
		t.Errorf("whitelist mode should NOT gain mcp tool, got %v", modes["locked"].AllowedTools)
	}

	// No MCP names is a no-op (the MCP-disabled default).
	before := slices.Clone(modes["auto"].AllowedTools)
	augmentModesWithMCP(modes, nil)
	if !slices.Equal(modes["auto"].AllowedTools, before) {
		t.Errorf("nil names should be a no-op")
	}
}

// --- helpers ---

func withMCPEnv(base func(string) string, gateway string) func(string) string {
	return func(k string) string {
		switch k {
		case "HARNESS_MCP_ENABLE":
			return "true"
		case "HARNESS_MCP_GATEWAY":
			return gateway
		default:
			return base(k)
		}
	}
}

// transcriptHasToolResult reports whether any message carries a tool_result
// block whose text contains sub.
func transcriptHasToolResult(msgs []llm.Message, sub string) bool {
	for _, m := range msgs {
		for _, b := range m.Content {
			if b.Kind == llm.BlockToolResult && strings.Contains(b.ResultText, sub) {
				return true
			}
		}
	}
	return false
}
