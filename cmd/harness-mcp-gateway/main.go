// Command harness-mcp-gateway is a thin CLI over internal/mcpgateway. It has
// three subcommands: serve (run the gateway daemon, started manually by an
// operator), tools (a debug client that lists the aggregated tool surface), and
// version. All gateway logic lives in internal/mcpgateway and internal/mcp; this
// binary only parses flags, resolves the log sink, wires signals, and dispatches.
//
// It mirrors cmd/harness conventions: flag.ContinueOnError with discarded flag
// output (errors are returned, not printed by the flag package), errors printed
// once at the entry point as "harness-mcp-gateway: %v" to stderr, exit codes
// 0 ok / 1 runtime / 2 usage, and an injectable environment so the run-
// equivalent code is testable in-process.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"

	"harness/internal/mcp"
	"harness/internal/mcpgateway"
)

// Exit codes mirror cmd/harness (design §1): 0 ok, 1 runtime, 2 usage.
const (
	exitOK      = 0
	exitRuntime = 1
	exitUsage   = 2
)

func main() {
	// SIGINT/SIGTERM are forwarded into the serve loop via this channel so a
	// signal cancels the daemon's context (mirrors cmd/harness's injected sigCh).
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	// SIGHUP is ignored process-wide so a detached gateway survives its
	// controlling terminal closing. This is process-lifetime signal policy, set
	// once here alongside the other signal setup rather than inside runServe, so
	// the run-equivalent code stays free of un-restored process-global mutation
	// (in-process tests call runServe directly).
	signal.Ignore(syscall.SIGHUP)

	os.Exit(run(environment{
		args:   os.Args[1:],
		stdout: os.Stdout,
		stderr: os.Stderr,
		getenv: os.Getenv,
		sigCh:  sigCh,
		// stderrIsTTY gates the "log to stderr" default: an interactive serve
		// logs to the terminal, a detached one falls back to a file so logs are
		// never lost.
		stderrIsTTY: isTTY(os.Stderr),
	}))
}

// environment carries everything run depends on, so the dispatch is testable
// with injected writers, env, signal channel, and TTY flag. A nil sigCh
// disables signal-driven cancellation (tests drive ctx directly or inject their
// own channel).
type environment struct {
	args        []string
	stdout      io.Writer
	stderr      io.Writer
	getenv      func(string) string
	sigCh       chan os.Signal
	stderrIsTTY bool
}

// run dispatches on the first non-flag argument (the subcommand) and returns the
// process exit code. Unknown/missing subcommands and -h/--help are handled here
// so every path prints usage to the right stream with the right exit code.
func run(env environment) int {
	args := env.args
	if len(args) == 0 {
		usage(env.stderr, env.getenv)
		return exitUsage
	}

	switch args[0] {
	case "-h", "--help", "help":
		usage(env.stdout, env.getenv)
		return exitOK
	case "serve":
		return runServe(env, args[1:])
	case "tools":
		return runTools(env, args[1:])
	case "version":
		fmt.Fprintf(env.stdout, "harness-mcp-gateway (MCP protocol %s)\n", mcp.ProtocolVersion)
		return exitOK
	default:
		fmt.Fprintf(env.stderr, "harness-mcp-gateway: unknown subcommand %q\n", args[0])
		usage(env.stderr, env.getenv)
		return exitUsage
	}
}

// usage prints the command summary to w. It lists the three subcommands, the
// serve flags, and the live default config/socket paths so users can find them.
func usage(w io.Writer, getenv func(string) string) {
	fmt.Fprint(w, `harness-mcp-gateway - MCP gateway daemon and debug client

Usage:
  harness-mcp-gateway serve   [-config path] [-socket path] [-listen addr] [-log path] [-log-level level]
  harness-mcp-gateway tools   [-config path] [-socket path] | [-gateway url]
  harness-mcp-gateway version

Subcommands:
  serve     Run the gateway daemon: load config, bind the socket, supervise
            downstream MCP servers, and serve their merged tools over the socket
            (and, when -listen is set, over a streamable-HTTP listener).
  tools     Connect to a running gateway and print the aggregated tool table,
            over its unix socket or (with -gateway) an HTTP listener.
  version   Print the binary's MCP protocol revision.

serve flags:
  -config path      config file (default: `+mcpgateway.DefaultConfigPath(getenv)+`)
  -socket path      unix socket to bind (overrides config; default: `+mcp.DefaultSocketPath(getenv)+`)
  -listen addr      HTTP listen address, e.g. 127.0.0.1:8089 (overrides config;
                    empty = no HTTP listener)
  -log path         log file (overrides config; default: stderr if a TTY, else a
                    gateway.log next to the socket)
  -log-level level  debug|info|warn|error (overrides config; default: info)

tools flags:
  -config path      config file (default: `+mcpgateway.DefaultConfigPath(getenv)+`)
  -socket path      gateway socket (overrides config; default: `+mcp.DefaultSocketPath(getenv)+`)
  -gateway url      query an HTTP gateway, e.g. http://127.0.0.1:8089 (mutually
                    exclusive with -socket)
`)
}

// isTTY reports whether f is a terminal, mirroring cmd/harness's helper. It
// gates the serve default of logging to stderr (interactive) versus a file
// (detached spawn), so a backgrounded gateway never loses its logs.
func isTTY(f *os.File) bool {
	info, err := f.Stat()
	if err != nil {
		return false
	}
	return info.Mode()&os.ModeCharDevice != 0
}

// resolveConfigPath turns the parsed -config flag into the path passed to
// mcpgateway.LoadConfig, mirroring cmd/harness's resolveConfigPath. An explicit
// flag value is used verbatim (a typo surfaces as a "not found" error in Load).
// When the flag was left at its default, a missing file resolves to "" so Load
// returns a valid empty config rather than erroring — the gateway must run with
// no config file present. explicit reports whether the user set -config.
func resolveConfigPath(flagValue string, explicit bool, getenv func(string) string) string {
	if explicit {
		return flagValue
	}
	def := mcpgateway.DefaultConfigPath(getenv)
	if _, err := os.Stat(def); err == nil {
		return def
	}
	return ""
}

// flagWasSet reports whether the named flag was explicitly provided on the
// command line (as opposed to left at its default).
func flagWasSet(fs *flag.FlagSet, name string) bool {
	set := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == name {
			set = true
		}
	})
	return set
}
