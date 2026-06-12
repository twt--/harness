package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"harness/internal/logging"
	"harness/internal/mcpgateway"
)

// serveCategory labels serve-level log records (config warnings, lifecycle).
const (
	serveCategory  = "mcp_gateway"
	configCategory = "mcp_config"
)

// gatewayLogName is the basename of the fallback log file, written next to the
// socket so a detached spawn (no TTY, no configured log file) still records.
const gatewayLogName = "gateway.log"

// runServe parses serve flags, loads config, resolves the log sink, wires
// signals into a cancellable context, and runs the daemon. ErrAlreadyRunning is
// a quiet success (exit 0) so a concurrent operator start — two `serve`
// invocations racing for the same socket — resolves to one daemon without a
// spurious failure.
func runServe(env environment, args []string) int {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(io.Discard) // errors are returned, printed once below (cmd/harness convention)
	// -config defaults to "" so we can distinguish "unset" (a missing default
	// path is non-fatal) from an explicit value (a typo is a hard error).
	configPath := fs.String("config", "", "config file path")
	socket := fs.String("socket", "", "unix socket path (overrides config and default)")
	listen := fs.String("listen", "", "HTTP listen address (overrides config; empty = no HTTP listener)")
	logPath := fs.String("log", "", "log file path (overrides config logFile)")
	logLevel := fs.String("log-level", "", "log level: debug|info|warn|error (overrides config)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			usage(env.stdout, env.getenv)
			return exitOK
		}
		fmt.Fprintf(env.stderr, "harness-mcp-gateway: %v\n", err)
		return exitUsage
	}

	cfg, err := mcpgateway.LoadConfig(resolveConfigPath(*configPath, flagWasSet(fs, "config"), env.getenv))
	if err != nil {
		fmt.Fprintf(env.stderr, "harness-mcp-gateway: %v\n", err)
		return exitRuntime
	}

	// Flags override config for socket and listen; defaults already filled by
	// LoadConfig (Listen has no default, so an unset flag leaves config's value).
	if *socket != "" {
		cfg.Socket = *socket
	}
	if flagWasSet(fs, "listen") {
		cfg.Listen = *listen
	}

	// Resolve the effective log level (flag > config > info), validating early so
	// a bad level surfaces as a usage error before we open any sink.
	level := cfg.LogLevel
	if *logLevel != "" {
		level = *logLevel
	}
	if _, err := logging.ParseLevel(level); err != nil {
		fmt.Fprintf(env.stderr, "harness-mcp-gateway: %v\n", err)
		return exitUsage
	}

	// Resolve and open the log sink (flag > config > stderr-if-TTY > file).
	sink, closeSink, err := openLogSink(logSinkParams{
		flagPath:    *logPath,
		configPath:  cfg.LogFile,
		socket:      cfg.Socket,
		stderr:      env.stderr,
		stderrIsTTY: env.stderrIsTTY,
	})
	if err != nil {
		fmt.Fprintf(env.stderr, "harness-mcp-gateway: %v\n", err)
		return exitRuntime
	}
	defer closeSink()

	logger, err := logging.NewLogger(sink, level, false)
	if err != nil {
		fmt.Fprintf(env.stderr, "harness-mcp-gateway: %v\n", err)
		return exitUsage
	}

	// Surface config load warnings (unset ${VAR}, skipped invalid servers) now
	// that the logger exists; library code never prints these itself.
	for _, w := range cfg.Warnings {
		logger.Warn(w, logging.Category(configCategory))
	}

	// Wire SIGINT/SIGTERM into ctx cancellation. The signal channel is injected
	// so tests can drive a clean shutdown without sending real process signals.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if env.sigCh != nil {
		go func() {
			select {
			case <-env.sigCh:
				cancel()
			case <-ctx.Done():
			}
		}()
	}

	d := mcpgateway.NewDaemon(cfg, logger)
	err = d.Run(ctx)
	if errors.Is(err, mcpgateway.ErrAlreadyRunning) {
		// Another live gateway already owns the socket: a quiet success so a
		// concurrent operator start (two `serve` invocations racing for the same
		// socket) does not surface as an error.
		logger.Info("gateway already running; exiting", logging.Category(serveCategory), "socket", cfg.Socket)
		return exitOK
	}
	if err != nil {
		logger.Error("gateway exited", logging.Category(serveCategory), "err", err)
		fmt.Fprintf(env.stderr, "harness-mcp-gateway: %v\n", err)
		return exitRuntime
	}
	return exitOK
}

// logSinkParams carries the inputs to log-sink resolution so the precedence
// rules are unit-testable without opening real files or process state.
type logSinkParams struct {
	flagPath    string
	configPath  string
	socket      string
	stderr      io.Writer
	stderrIsTTY bool
}

// openLogSink resolves and opens the log sink in precedence order:
//
//	-log flag > config logFile > stderr (when stderr is a TTY) > default file
//	<socket-dir>/gateway.log
//
// The default-file fallback guarantees a detached start (no TTY, no configured
// file) never loses logs. File sinks open append-only; the
// returned close func is a no-op for the stderr sink (we must not close the
// process's stderr). Parent directories for an explicit/config path are not
// created — only the socket dir's gateway.log path is created on demand, since
// the daemon owns that directory.
func openLogSink(p logSinkParams) (sink io.Writer, closeFn func(), err error) {
	switch {
	case p.flagPath != "":
		return openLogFile(p.flagPath)
	case p.configPath != "":
		return openLogFile(p.configPath)
	case p.stderrIsTTY:
		return p.stderr, func() {}, nil
	default:
		// Detached spawn: log next to the socket so output is never lost.
		dir := filepath.Dir(p.socket)
		if dir == "" || dir == "." {
			// No socket dir to anchor to: fall back to stderr rather than writing
			// a gateway.log into the current working directory.
			return p.stderr, func() {}, nil
		}
		return openLogFile(filepath.Join(dir, gatewayLogName))
	}
}

// openLogFile opens path append-only, creating it if absent. For the
// gateway.log fallback the socket dir may not exist yet when we open, so create
// it best-effort first.
func openLogFile(path string) (io.Writer, func(), error) {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		// Best-effort: a creation failure is reported by the OpenFile below with a
		// clearer path-specific error.
		_ = os.MkdirAll(dir, 0o755)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return nil, nil, fmt.Errorf("open log file %s: %w", path, err)
	}
	return f, func() { _ = f.Close() }, nil
}
