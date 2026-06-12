package mcpgateway

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"harness/internal/logging"
	"harness/internal/mcp"
)

// ErrAlreadyRunning is returned by Run when another live gateway already owns the
// socket. The CLI wrapper (Task 5) treats it as a quiet success (exit 0).
var ErrAlreadyRunning = errors.New("mcpgateway: gateway already running")

const (
	// dialProbeTimeout bounds the liveness probe against an existing socket.
	dialProbeTimeout = 250 * time.Millisecond
	// shutdownWait bounds the concurrent supervisor teardown on shutdown.
	shutdownWait = 10 * time.Second
	// httpShutdownWait bounds the graceful http.Server.Shutdown before it is
	// forced closed, so a stuck HTTP request cannot wedge daemon teardown.
	httpShutdownWait = 5 * time.Second
)

// Daemon ties config, supervisors, the registry, and the unix listener together.
// Run binds the socket, starts all supervisors eagerly, accepts connections
// (one mcp.Serve session each), and shuts down cleanly on ctx cancel.
type Daemon struct {
	cfg    Config
	logger *slog.Logger

	// spawn/sleep are injected into every supervisor; nil → production defaults.
	// Tests set them to avoid real subprocesses and backoff waits.
	spawn func() *exec.Cmd
	sleep func(context.Context, time.Duration)

	supervisors []*Supervisor
	registry    *Registry

	mu       sync.Mutex
	sessions map[*mcp.ServerSession]struct{}

	// httpServer is the optional streamable-HTTP server, non-nil only when
	// cfg.Listen is set. It is shut down alongside the socket on teardown.
	httpServer *http.Server
}

// NewDaemon builds a daemon for cfg. Run must be called to start it.
func NewDaemon(cfg Config, logger *slog.Logger) *Daemon {
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	return &Daemon{
		cfg:      cfg,
		logger:   logger,
		sessions: map[*mcp.ServerSession]struct{}{},
	}
}

// Run binds the socket and serves until ctx is cancelled. It returns nil on a
// clean shutdown, ErrAlreadyRunning if a live gateway owns the socket, or a bind
// error otherwise. Task 5 wires signals into ctx.
func (d *Daemon) Run(ctx context.Context) error {
	listener, err := d.listen()
	if err != nil {
		return err
	}

	// Build supervisors, then the registry (which injects each supervisor's
	// onToolsChanged callback), and only then start the run loops. Starting before
	// the registry assigns the callback would race the run-loop goroutine's read
	// of onToolsChanged against the registry's write.
	d.buildSupervisors()
	d.registry = NewRegistry(d.supervisors, d.logger)
	d.startSupervisors(ctx)

	// Bind the optional HTTP listener now that the registry (the Provider) exists.
	// A bind failure is fatal: the operator explicitly asked for it, mirroring the
	// socket-bind contract. Tear down what is already started before returning,
	// including the socket listener (the accept loop that would otherwise close it
	// never starts on this path).
	httpListener, err := d.listenHTTP()
	if err != nil {
		listener.Close()
		d.shutdown()
		return err
	}
	if httpListener != nil {
		d.serveHTTP(httpListener)
	}

	// Close the listener when ctx is cancelled so Accept unblocks and the loop
	// exits cleanly.
	go func() {
		<-ctx.Done()
		listener.Close()
	}()

	d.acceptLoop(ctx, listener)

	// Shutdown: remove the socket, tear down supervisors, sessions, and the
	// HTTP server.
	d.shutdown()
	return nil
}

// listenHTTP binds the TCP listener for the streamable-HTTP server when
// cfg.Listen is set, returning a nil listener (no error) when it is empty. A
// bind error is surfaced so Run fails, consistent with the socket-bind contract.
func (d *Daemon) listenHTTP() (net.Listener, error) {
	if d.cfg.Listen == "" {
		return nil, nil
	}
	l, err := net.Listen("tcp", d.cfg.Listen)
	if err != nil {
		return nil, fmt.Errorf("mcpgateway: listen %s: %w", d.cfg.Listen, err)
	}
	return l, nil
}

// serveHTTP starts the streamable-HTTP server over l in a background goroutine.
// HTTP sessions are stateless re-listers: they do NOT subscribe to list_changed
// fan-out (there is no server-push channel over this transport), so the handler
// is wired with the registry as its Provider and nothing else.
func (d *Daemon) serveHTTP(l net.Listener) {
	handler := mcp.NewHTTPHandler(mcp.HTTPHandlerOptions{
		Info:     gatewayServerInfo(),
		Provider: d.registry,
		Logger:   d.logger,
	})
	// ReadHeaderTimeout/IdleTimeout harden against slowloris-style stalls. Cheap
	// even behind the assumed local/front-proxy boundary.
	d.httpServer = &http.Server{
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
	}
	d.logger.Info("serving MCP over HTTP", logging.Category(categoryGate), "addr", l.Addr().String())
	go func() {
		if err := d.httpServer.Serve(l); err != nil && !errors.Is(err, http.ErrServerClosed) {
			d.logger.Warn("http server stopped", logging.Category(categoryGate), "err", err)
		}
	}()
}

// listen binds the unix socket, creating the parent dir when missing, and
// handles a stale or live pre-existing socket.
func (d *Daemon) listen() (net.Listener, error) {
	dir := filepath.Dir(d.cfg.Socket)
	if err := ensureSocketDir(dir); err != nil {
		return nil, err
	}

	listener, err := net.Listen("unix", d.cfg.Socket)
	if err == nil {
		return listener, nil
	}

	// Bind failed. If the path is in use, probe whether a live gateway owns it.
	if !isAddrInUse(err) {
		return nil, fmt.Errorf("mcpgateway: listen %s: %w", d.cfg.Socket, err)
	}
	if d.socketAlive() {
		return nil, ErrAlreadyRunning
	}
	// Stale socket from a crashed prior daemon: unlink and retry once.
	if rerr := os.Remove(d.cfg.Socket); rerr != nil && !errors.Is(rerr, os.ErrNotExist) {
		return nil, fmt.Errorf("mcpgateway: remove stale socket: %w", rerr)
	}
	listener, err = net.Listen("unix", d.cfg.Socket)
	if err != nil {
		return nil, fmt.Errorf("mcpgateway: listen %s after unlink: %w", d.cfg.Socket, err)
	}
	return listener, nil
}

// socketAlive reports whether something is accepting on the configured socket.
func (d *Daemon) socketAlive() bool {
	conn, err := net.DialTimeout("unix", d.cfg.Socket, dialProbeTimeout)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

// buildSupervisors constructs (but does not start) a supervisor for each server,
// wiring the injected spawn/sleep hooks. The registry assigns each one's
// onToolsChanged before startSupervisors launches the run loops.
func (d *Daemon) buildSupervisors() {
	d.supervisors = make([]*Supervisor, 0, len(d.cfg.Servers))
	for _, rs := range d.cfg.Servers {
		sup := NewSupervisor(rs, d.logger)
		if d.spawn != nil {
			sup.spawn = d.spawn
		}
		if d.sleep != nil {
			sup.sleep = d.sleep
		}
		d.supervisors = append(d.supervisors, sup)
	}
}

// startSupervisors eagerly starts each supervisor's run loop. It must be called
// after the registry has injected onToolsChanged.
func (d *Daemon) startSupervisors(ctx context.Context) {
	for _, sup := range d.supervisors {
		sup.Start(ctx)
	}
}

// acceptLoop accepts connections until the listener is closed, serving each in
// its own goroutine. It blocks until the listener stops accepting.
func (d *Daemon) acceptLoop(ctx context.Context, listener net.Listener) {
	var wg sync.WaitGroup
	for {
		conn, err := listener.Accept()
		if err != nil {
			// A closed listener (ctx cancel) ends the loop cleanly.
			if ctx.Err() != nil || errors.Is(err, net.ErrClosed) {
				break
			}
			d.logger.Warn("accept error", logging.Category(categoryGate), "err", err)
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			d.serveConn(ctx, conn)
		}()
	}
	wg.Wait()
}

// serveConn runs one MCP server session over conn, subscribing it for
// list_changed fan-out and unsubscribing on return.
func (d *Daemon) serveConn(ctx context.Context, conn net.Conn) {
	err := mcp.Serve(ctx, conn, mcp.ServerOptions{
		Info:        gatewayServerInfo(),
		Provider:    d.registry,
		ListChanged: true,
		Logger:      d.logger,
		OnSession: func(s *mcp.ServerSession) {
			d.registry.Subscribe(s)
			d.mu.Lock()
			d.sessions[s] = struct{}{}
			d.mu.Unlock()
		},
	})
	if err != nil {
		d.logger.Debug("session ended with error", logging.Category(categoryGate), "err", err)
	}
	// Best-effort unsubscribe: the session handle is only available via OnSession,
	// so the registry drops dead sessions on the next broadcast as well. Here we
	// proactively clear it from our own tracking set.
	d.dropSessions()
}

// dropSessions removes any sessions whose Done channel has closed from both the
// daemon's tracking set and the registry. It is called after each session ends.
func (d *Daemon) dropSessions() {
	d.mu.Lock()
	var dead []*mcp.ServerSession
	for s := range d.sessions {
		select {
		case <-s.Done():
			dead = append(dead, s)
		default:
		}
	}
	for _, s := range dead {
		delete(d.sessions, s)
	}
	d.mu.Unlock()
	for _, s := range dead {
		d.registry.Unsubscribe(s)
	}
}

// shutdown removes the socket, stops the HTTP server, tears down all
// supervisors concurrently (bounded), and closes any lingering sessions.
func (d *Daemon) shutdown() {
	_ = os.Remove(d.cfg.Socket)

	// Stop the HTTP server: graceful Shutdown bounded by httpShutdownWait, then a
	// hard Close so a wedged request cannot block teardown. Closing the server
	// closes its listener.
	if d.httpServer != nil {
		hctx, hcancel := context.WithTimeout(context.Background(), httpShutdownWait)
		if err := d.httpServer.Shutdown(hctx); err != nil {
			_ = d.httpServer.Close()
		}
		hcancel()
	}

	// Close lingering sessions.
	d.mu.Lock()
	sessions := make([]*mcp.ServerSession, 0, len(d.sessions))
	for s := range d.sessions {
		sessions = append(sessions, s)
	}
	d.sessions = map[*mcp.ServerSession]struct{}{}
	d.mu.Unlock()
	for _, s := range sessions {
		_ = s.Close()
	}

	// Tear down supervisors concurrently, bounded by a single shutdownWait
	// deadline. Each Shutdown honors the ctx, collapsing its reap waits to an
	// immediate SIGTERM/SIGKILL escalation once the deadline passes, so the ctx
	// timeout alone bounds the whole teardown.
	ctx, cancel := context.WithTimeout(context.Background(), shutdownWait)
	defer cancel()
	var wg sync.WaitGroup
	for _, sup := range d.supervisors {
		wg.Add(1)
		go func(s *Supervisor) {
			defer wg.Done()
			s.Shutdown(ctx)
		}(sup)
	}
	wg.Wait()
	if ctx.Err() != nil {
		d.logger.Warn("supervisor shutdown reached its deadline", logging.Category(categoryGate))
	}
}

// ensureSocketDir creates the socket's parent dir when it is missing. The
// gateway never touches permissions: a created dir gets the umask default and a
// pre-existing parent is used as-is, whatever its owner or mode, so the socket
// location is fully user-configurable. When a regular file occupies the parent
// path, MkdirAll fails naturally and that error is surfaced.
func ensureSocketDir(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("mcpgateway: create socket dir: %w", err)
	}
	return nil
}

// isAddrInUse reports whether err is an "address already in use" bind failure,
// classified by errno (not string matching).
func isAddrInUse(err error) bool {
	return errors.Is(err, syscall.EADDRINUSE)
}

// gatewayServerInfo is the Implementation the gateway presents to harness as an
// MCP server.
func gatewayServerInfo() mcp.Implementation {
	return mcp.Implementation{Name: "harness-mcp-gateway", Version: gatewayVersion}
}
