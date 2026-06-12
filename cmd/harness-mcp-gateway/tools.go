package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"strings"

	"harness/internal/mcp"
	"harness/internal/mcpgateway"
)

// runTools connects to a running gateway as an MCP client and prints the
// aggregated tool table. It is a debug/status command: if it connects and
// lists, the gateway is up. It targets either the unix socket (-socket > config
// > default) or an HTTP listener (-gateway), which are mutually exclusive.
func runTools(env environment, args []string) int {
	fs := flag.NewFlagSet("tools", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	// -config defaults to "" so a missing default path is non-fatal (resolved
	// below); an explicit -config typo still surfaces as a load error.
	configPath := fs.String("config", "", "config file path")
	socket := fs.String("socket", "", "gateway socket path (overrides config and default)")
	gateway := fs.String("gateway", "", "HTTP gateway URL (mutually exclusive with -socket)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			usage(env.stdout, env.getenv)
			return exitOK
		}
		fmt.Fprintf(env.stderr, "harness-mcp-gateway: %v\n", err)
		return exitUsage
	}

	if *gateway != "" && *socket != "" {
		fmt.Fprintf(env.stderr, "harness-mcp-gateway: -gateway and -socket are mutually exclusive\n")
		return exitUsage
	}

	client, code := toolsClient(env, fs, *gateway, *socket, *configPath)
	if code != exitOK {
		return code
	}
	defer client.Close()

	ctx := context.Background()
	if _, err := client.Initialize(ctx); err != nil {
		fmt.Fprintf(env.stderr, "harness-mcp-gateway: %v\n", err)
		return exitRuntime
	}
	tools, err := client.ListTools(ctx)
	if err != nil {
		fmt.Fprintf(env.stderr, "harness-mcp-gateway: %v\n", err)
		return exitRuntime
	}

	printToolTable(env.stdout, tools)
	return exitOK
}

// toolsClient builds the MCP client for the tools command: an HTTP client when
// -gateway is set, else a unix-socket client. The returned code is exitOK on
// success; on failure the error is already printed.
func toolsClient(env environment, fs *flag.FlagSet, gateway, socket, configFlag string) (*mcp.Client, int) {
	info := mcp.Implementation{Name: "harness-mcp-gateway-tools", Version: "1"}

	if gateway != "" {
		tr := mcp.NewHTTPTransport(mcp.HTTPOptions{Endpoint: gateway})
		return mcp.NewClientTransport(tr, mcp.ClientOptions{Info: info}), exitOK
	}

	socketPath, code := resolveToolsSocket(env, socket, resolveConfigPath(configFlag, flagWasSet(fs, "config"), env.getenv))
	if code != exitOK {
		return nil, code
	}
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		fmt.Fprintf(env.stderr, "harness-mcp-gateway: cannot connect to gateway at %s: %v\n", socketPath, err)
		return nil, exitRuntime
	}
	return mcp.NewClient(conn, mcp.ClientOptions{Info: info}), exitOK
}

// resolveToolsSocket determines which socket the tools command connects to:
// the -socket flag, else the config's socket, else the default. A config that
// fails to load is a runtime error (a typo'd -config should not silently fall
// back to the default socket). The returned code is exitOK on success.
func resolveToolsSocket(env environment, socketFlag, configPath string) (string, int) {
	if socketFlag != "" {
		return socketFlag, exitOK
	}
	cfg, err := mcpgateway.LoadConfig(configPath)
	if err != nil {
		fmt.Fprintf(env.stderr, "harness-mcp-gateway: %v\n", err)
		return "", exitRuntime
	}
	// LoadConfig always fills Socket (config value or default), so this is never
	// empty; guard defensively anyway.
	if cfg.Socket == "" {
		return mcp.DefaultSocketPath(env.getenv), exitOK
	}
	return cfg.Socket, exitOK
}

// printToolTable writes one line per tool as "NAME\tDESCRIPTION-first-line",
// preceded by a count header ("N tools" or "no tools"). Tools arrive sorted by
// name from the gateway, so no local sort is needed.
func printToolTable(w io.Writer, tools []mcp.Tool) {
	if len(tools) == 0 {
		fmt.Fprintln(w, "no tools")
		return
	}
	noun := "tools"
	if len(tools) == 1 {
		noun = "tool"
	}
	fmt.Fprintf(w, "%d %s\n", len(tools), noun)
	for _, t := range tools {
		fmt.Fprintf(w, "%s\t%s\n", t.Name, firstLine(t.Description))
	}
}

// firstLine returns the first line of s with surrounding whitespace trimmed, so
// a multi-line description collapses to a single table cell.
func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	return strings.TrimSpace(s)
}
