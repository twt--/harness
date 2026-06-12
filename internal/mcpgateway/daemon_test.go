package mcpgateway

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"harness/internal/mcp"
)

// shortSocketDir returns a short-lived temp directory whose path is well under
// the unix sun_path limit (~104 bytes on macOS). t.TempDir() on macOS nests
// under a long /var/folders/.../<TestName>/NNN path that can overflow sun_path,
// so socket tests use this instead and clean it up themselves.
func shortSocketDir(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "hmg")
	if err != nil {
		t.Fatalf("mkdtemp: %v", err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	return dir
}

// daemonConfig builds a Config wired to spawn the TestHelperProcess fake server,
// with a socket under a short temp dir.
func daemonConfig(t *testing.T, env map[string]string) (Config, func() *exec.Cmd) {
	t.Helper()
	dir := shortSocketDir(t)
	socket := filepath.Join(dir, "g.sock")
	cfg := Config{
		Socket:   socket,
		LogLevel: "debug",
		Servers: []ResolvedServer{
			{Name: "h", Transport: TransportStdio, Command: "helper"},
		},
	}
	return cfg, helperSpawn(t, env)
}

// startDaemon runs a Daemon in a goroutine and returns a channel that receives
// Run's error. The supervisor spawn func is injected so no real binary is needed.
func startDaemon(t *testing.T, ctx context.Context, cfg Config, spawn func() *exec.Cmd) <-chan error {
	t.Helper()
	d := NewDaemon(cfg, slog.New(slog.DiscardHandler))
	d.spawn = spawn
	d.sleep = func(context.Context, time.Duration) {}
	errCh := make(chan error, 1)
	go func() { errCh <- d.Run(ctx) }()
	return errCh
}

// dialGateway connects an mcp.Client to the gateway socket, initializing it.
func dialGateway(t *testing.T, socket string) *mcp.Client {
	t.Helper()
	var conn net.Conn
	var err error
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		conn, err = net.Dial("unix", socket)
		if err == nil {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if err != nil {
		t.Fatalf("dial gateway: %v", err)
	}
	client := mcp.NewClient(conn, mcp.ClientOptions{Info: mcp.Implementation{Name: "test", Version: "1"}})
	if _, err := client.Initialize(context.Background()); err != nil {
		t.Fatalf("initialize: %v", err)
	}
	return client
}

// freePort binds an ephemeral TCP port, closes it, and returns the address so a
// daemon can re-bind it. There is a small race between close and re-bind, but it
// is the standard stdlib idiom for "give me a free port" in tests.
func freePort(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen ephemeral: %v", err)
	}
	addr := l.Addr().String()
	l.Close()
	return addr
}

// dialHTTPGateway builds an mcp.Client over the daemon's HTTP listener and
// initializes it, polling until the listener accepts.
func dialHTTPGateway(t *testing.T, addr string) *mcp.Client {
	t.Helper()
	endpoint := "http://" + addr
	tr := mcp.NewHTTPTransport(mcp.HTTPOptions{Endpoint: endpoint})
	client := mcp.NewClientTransport(tr, mcp.ClientOptions{Info: mcp.Implementation{Name: "test-http", Version: "1"}})
	deadline := time.Now().Add(5 * time.Second)
	var err error
	for time.Now().Before(deadline) {
		if _, err = client.Initialize(context.Background()); err == nil {
			return client
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("initialize HTTP gateway: %v", err)
	return nil
}

func TestDaemonServesSocketAndHTTP(t *testing.T) {
	dir := shortSocketDir(t)
	socket := filepath.Join(dir, "g.sock")
	addr := freePort(t)
	cfg := Config{
		Socket:  socket,
		Listen:  addr,
		Servers: []ResolvedServer{{Name: "h", Transport: TransportStdio, Command: "helper"}},
	}
	spawn := helperSpawn(t, map[string]string{"HELPER_TOOLS": "echo"})

	ctx, cancel := context.WithCancel(context.Background())
	errCh := startDaemon(t, ctx, cfg, spawn)

	// Both transports must surface the same aggregated tool.
	socketClient := dialGateway(t, socket)
	httpClient := dialHTTPGateway(t, addr)

	for _, c := range []*mcp.Client{socketClient, httpClient} {
		var tools []mcp.Tool
		waitFor(t, 5*time.Second, func() bool {
			var err error
			tools, err = c.ListTools(context.Background())
			return err == nil && len(tools) == 1
		})
		if tools[0].Name != "mcp__h__echo" {
			t.Fatalf("unexpected tool over transport: %+v", tools)
		}
		res, err := c.CallTool(context.Background(), "mcp__h__echo", json.RawMessage(`{"hi":1}`))
		if err != nil || res.IsError || res.Content[0].Text != `{"hi":1}` {
			t.Fatalf("call round-trip wrong: err=%v res=%+v", err, res)
		}
	}

	socketClient.Close()
	httpClient.Close()
	cancel()
	if err := <-errCh; err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	// The HTTP listener is closed on shutdown: a fresh dial must fail.
	if conn, err := net.DialTimeout("tcp", addr, 250*time.Millisecond); err == nil {
		conn.Close()
		t.Fatalf("HTTP listener still accepting after shutdown")
	}
}

func TestDaemonHTTPBindFailureFailsRun(t *testing.T) {
	dir := shortSocketDir(t)
	socket := filepath.Join(dir, "g.sock")
	// An unbindable address (port 0 with a bogus host form) — use a port already
	// held to force EADDRINUSE.
	held, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("hold port: %v", err)
	}
	defer held.Close()
	addr := held.Addr().String()

	cfg := Config{Socket: socket, Listen: addr}
	d := NewDaemon(cfg, slog.New(slog.DiscardHandler))
	d.spawn = helperSpawn(t, nil)
	d.sleep = func(context.Context, time.Duration) {}

	runErr := d.Run(context.Background())
	if runErr == nil {
		t.Fatalf("Run should fail when the HTTP listener cannot bind")
	}
	if !strings.Contains(runErr.Error(), addr) {
		t.Errorf("error should name the listen addr; got %v", runErr)
	}
	// The socket must be cleaned up on the failed-bind teardown path.
	if _, err := os.Stat(socket); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("socket not removed after failed HTTP bind: %v", err)
	}
}

func TestDaemonEndToEnd(t *testing.T) {
	cfg, spawn := daemonConfig(t, map[string]string{"HELPER_TOOLS": "echo"})
	ctx, cancel := context.WithCancel(context.Background())
	errCh := startDaemon(t, ctx, cfg, spawn)

	client := dialGateway(t, cfg.Socket)

	// ListTools should eventually include the namespaced echo tool.
	var tools []mcp.Tool
	waitFor(t, 5*time.Second, func() bool {
		var err error
		tools, err = client.ListTools(context.Background())
		return err == nil && len(tools) == 1
	})
	if tools[0].Name != "mcp__h__echo" {
		t.Fatalf("unexpected tool: %+v", tools)
	}

	res, err := client.CallTool(context.Background(), "mcp__h__echo", json.RawMessage(`{"hi":1}`))
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if res.IsError || res.Content[0].Text != `{"hi":1}` {
		t.Fatalf("call round-trip wrong: %+v", res)
	}

	client.Close()
	cancel()
	if err := <-errCh; err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	// Socket removed on shutdown.
	if _, err := os.Stat(cfg.Socket); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("socket not removed: %v", err)
	}
}

func TestDaemonAlreadyRunning(t *testing.T) {
	cfg, spawn := daemonConfig(t, map[string]string{"HELPER_TOOLS": "echo"})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := startDaemon(t, ctx, cfg, spawn)

	// Wait for the first daemon to bind.
	dialGateway(t, cfg.Socket).Close()

	// A second daemon on the same socket must return ErrAlreadyRunning.
	d2 := NewDaemon(cfg, slog.New(slog.DiscardHandler))
	d2.spawn = spawn
	d2.sleep = func(context.Context, time.Duration) {}
	err := d2.Run(context.Background())
	if !errors.Is(err, ErrAlreadyRunning) {
		t.Fatalf("second daemon: want ErrAlreadyRunning, got %v", err)
	}

	cancel()
	<-errCh
}

func TestDaemonStaleSocketUnlinked(t *testing.T) {
	dir := shortSocketDir(t)
	socket := filepath.Join(dir, "g.sock")

	// Create a stale unix socket file with NO listener: bind then close the
	// listener but leave the file (simulating a crashed prior daemon).
	l, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatalf("pre-bind: %v", err)
	}
	// Closing the listener normally unlinks; recreate a plain file to be a true
	// stale path that net.Listen rejects with EADDRINUSE.
	l.Close()
	if err := os.WriteFile(socket, []byte{}, 0o600); err != nil {
		t.Fatalf("write stale file: %v", err)
	}

	cfg := Config{Socket: socket, LogLevel: "debug"}
	ctx, cancel := context.WithCancel(context.Background())
	errCh := startDaemon(t, ctx, cfg, helperSpawn(t, nil))

	// The daemon should unlink the stale path and bind successfully.
	client := dialGateway(t, socket)
	client.Close()

	cancel()
	if err := <-errCh; err != nil {
		t.Fatalf("Run: %v", err)
	}
}

func TestDaemonConcurrentSessionsShareChild(t *testing.T) {
	dir := shortSocketDir(t)
	socket := filepath.Join(dir, "g.sock")
	counter := filepath.Join(dir, "spawns.txt")
	cfg := Config{
		Socket:  socket,
		Servers: []ResolvedServer{{Name: "h", Transport: TransportStdio, Command: "helper"}},
	}
	spawn := helperSpawn(t, map[string]string{"HELPER_TOOLS": "echo", "HELPER_SPAWN_COUNTER": counter})

	ctx, cancel := context.WithCancel(context.Background())
	errCh := startDaemon(t, ctx, cfg, spawn)

	c1 := dialGateway(t, socket)
	c2 := dialGateway(t, socket)

	var wg sync.WaitGroup
	for _, c := range []*mcp.Client{c1, c2} {
		wg.Add(1)
		go func(client *mcp.Client) {
			defer wg.Done()
			waitFor(t, 5*time.Second, func() bool {
				tools, err := client.ListTools(context.Background())
				return err == nil && len(tools) == 1
			})
			res, err := client.CallTool(context.Background(), "mcp__h__echo", json.RawMessage(`{"a":1}`))
			if err != nil || res.IsError {
				t.Errorf("call failed: err=%v res=%+v", err, res)
			}
		}(c)
	}
	wg.Wait()

	// The downstream child must have been spawned exactly once (shared).
	if got := readCounter(t, counter); got != 1 {
		t.Fatalf("downstream spawned %d times, want 1 (shared)", got)
	}

	c1.Close()
	c2.Close()
	cancel()
	<-errCh
}

func TestDaemonCreatesMissingSocketDir(t *testing.T) {
	dir := shortSocketDir(t)
	socketDir := filepath.Join(dir, "sub")
	socket := filepath.Join(socketDir, "g.sock")
	cfg := Config{Socket: socket}

	ctx, cancel := context.WithCancel(context.Background())
	errCh := startDaemon(t, ctx, cfg, helperSpawn(t, nil))
	dialGateway(t, socket).Close()

	if _, err := os.Stat(socketDir); err != nil {
		t.Fatalf("stat socket dir: %v", err)
	}

	cancel()
	<-errCh
}

func TestEnsureSocketDirCreatesMissing(t *testing.T) {
	dir := filepath.Join(shortSocketDir(t), "new")
	if err := ensureSocketDir(dir); err != nil {
		t.Fatalf("ensureSocketDir: %v", err)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("stat: %v", err)
	}
}

func TestEnsureSocketDirUsesPreexistingDirAsIs(t *testing.T) {
	dir := filepath.Join(shortSocketDir(t), "loose")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := ensureSocketDir(dir); err != nil {
		t.Fatalf("ensureSocketDir: %v", err)
	}
	info, err := os.Stat(dir)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	// A pre-existing parent is used as-is: its mode is not tightened, so the
	// socket location is fully user-configurable.
	if perm := info.Mode().Perm(); perm != 0o755 {
		t.Fatalf("pre-existing dir mode changed: perm = %o, want 0755", perm)
	}
}

func TestEnsureSocketDirErrorsOnRegularFileParent(t *testing.T) {
	// MkdirAll fails naturally when a regular file occupies the parent path; the
	// error is surfaced rather than swallowed. This is a plain error-path test,
	// not a location-restriction defense.
	path := filepath.Join(shortSocketDir(t), "afile")
	if err := os.WriteFile(path, []byte{}, 0o600); err != nil {
		t.Fatalf("write file: %v", err)
	}
	err := ensureSocketDir(path)
	if err == nil {
		t.Fatalf("ensureSocketDir should error when a regular file occupies the parent path")
	}
	if !strings.Contains(err.Error(), "not a directory") {
		t.Fatalf("unexpected error: %v", err)
	}
}
