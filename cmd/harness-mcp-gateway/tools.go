package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"harness/internal/mcp"
	"harness/internal/mcpgateway"
)

var toolsCommandTimeout = 10 * time.Second

// runTools connects to a running gateway as an MCP client and prints the
// aggregated tool table. It is a debug/status command: if it connects and
// lists, the gateway is up. It targets the HTTP gateway URL from -gateway,
// config gateway.listen, or the default listener.
func runTools(env environment, args []string) int {
	fs := flag.NewFlagSet("tools", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	// -config defaults to "" so a missing default path is non-fatal (resolved
	// below); an explicit -config typo still surfaces as a load error.
	configPath := fs.String("config", "", "config file path")
	gateway := fs.String("gateway", "", "HTTP gateway URL")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			usage(env.stdout, env.getenv)
			return exitOK
		}
		fmt.Fprintf(env.stderr, "harness-mcp-gateway: %v\n", err)
		return exitUsage
	}

	gatewayURL, code := resolveToolsGateway(env, fs, *gateway, *configPath)
	if code != exitOK {
		return code
	}
	client := toolsClient(gatewayURL)
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), toolsCommandTimeout)
	defer cancel()
	if _, err := client.Initialize(ctx); err != nil {
		fmt.Fprintf(env.stderr, "harness-mcp-gateway: cannot connect to gateway at %s: %v\n", gatewayURL, err)
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

// toolsClient builds the HTTP MCP client for the tools command.
func toolsClient(gatewayURL string) *mcp.Client {
	info := mcp.Implementation{Name: "harness-mcp-gateway-tools", Version: "1"}
	tr := mcp.NewHTTPTransport(mcp.HTTPOptions{Endpoint: gatewayURL})
	return mcp.NewClientTransport(tr, mcp.ClientOptions{Info: info})
}

// resolveToolsGateway determines which HTTP URL the tools command connects to:
// the -gateway flag, else the config's gateway.listen, else the default URL. A
// config that fails to load is a runtime error (a typo'd -config should not
// silently fall back to the default URL). The returned code is exitOK on success.
func resolveToolsGateway(env environment, fs *flag.FlagSet, gatewayFlag, configFlag string) (string, int) {
	if gatewayFlag != "" {
		return gatewayFlag, exitOK
	}
	configPath := resolveConfigPath(configFlag, flagWasSet(fs, "config"), env.getenv)
	cfg, err := mcpgateway.LoadConfig(configPath)
	if err != nil {
		fmt.Fprintf(env.stderr, "harness-mcp-gateway: %v\n", err)
		return "", exitRuntime
	}
	return mcpgateway.URLForListen(cfg.Listen), exitOK
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
