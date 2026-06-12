# Smoke / verification matrix

This document records the manual smoke matrix for `harness` (design §13) and how
to re-run each leg. It complements — it does not replace — the unit and golden
suites (`go test ./...`, 711 test functions across 27 packages).

The legs split in two groups:

- **Hermetic legs** drive the real, freshly-built binary as a subprocess against
  a throwaway OpenAI-compatible mock server bound to `127.0.0.1` (no network, no
  API keys). They are automated in `cmd/harness/integration_test.go` and run as
  part of `go test ./...`. The mock lives only in `_test.go`, so it is never
  compiled into the shipped binary.
- **Real-API legs** require provider credentials and are **BLOCKED** in this
  environment (no `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, no local Ollama). They
  are documented below with the exact commands to run them by hand.

## Environment at time of writing

- Go: `go1.26.4 darwin/arm64`
- `ANTHROPIC_API_KEY`: not set
- `OPENAI_API_KEY`: not set
- Ollama: not installed / not running

## Hermetic legs (automated, PASS)

Run them all:

```sh
go test ./cmd/harness/ -run TestSmoke -v
# or under the race detector:
go test -race ./cmd/harness/ -run TestSmoke
```

Last run: all three PASS, no data races.

| Leg | Test | What it asserts |
|---|---|---|
| Local OpenAI-compatible server, tool round-trip | `TestSmokeToolRoundTrip` | The mock streams a `read_file` tool call, the harness executes it, and a **second** request to the mock carries the `role:"tool"` result with the file's content. The assistant's final text lands on **stdout**; a session file is written and passes `ValidateTranscript`. |
| `^C` during a stream | `TestSmokeInterruptMidStream` | The mock streams `partial answer` then stalls (300 ms/line). After the partial text reaches stdout, the test sends `SIGINT` to the subprocess. The process exits **130**; the saved session keeps the partial assistant text and passes `ValidateTranscript` (the §4 cancel-repair: keep streamed text, strip un-executed tool calls). |
| Resume of an interrupted session | `TestSmokeResumeInterrupted` | A crafted session whose transcript ends in a **dangling `tool_use`** is resumed with `-resume`. `session.Load` repairs it with a synthesized `tool_result` (`is_error`, text `interrupted`). The mock's single request is verified to contain that `role:"tool"` / `tool_call_id` message, and the run completes against the mock's text turn. |

### MCP gateway legs (automated, PASS)

These exercise the optional MCP gateway end to end without a network or any real
downstream server: a fake in-process gateway (or the real `harness-mcp-gateway`
serve loop driven against a fake downstream) stands in. They live in
`cmd/harness/mcp_test.go`, `cmd/harness-mcp-gateway/main_test.go`, and
`internal/mcpgateway/daemon_test.go`, and run under `go test ./...`.

```sh
go test ./cmd/harness/ -run TestSetupMCP -v
go test ./cmd/harness-mcp-gateway/ -run 'TestServe|TestTools' -v
go test ./internal/mcpgateway/ -run TestDaemonServesSocketAndHTTP -v
```

| Leg | Test | What it asserts |
|---|---|---|
| Gateway `serve` + `tools` listing | `TestToolsListsAggregatedTools` | `runServe` binds a socket, supervises a fake downstream, and aggregates its tools; the `tools` subcommand connects and prints `2 tools` with the namespaced names `mcp__fake__echo` / `mcp__fake__ping`, descriptions collapsed to their first line. A `SIGINT` shuts the daemon down cleanly. |
| One-shot calling an `mcp__` tool (unix socket) | `TestSetupMCPRegistersToolsAndOneShotCalls` | With `HARNESS_MCP_ENABLE=true` and the gateway socket in env, `harness -p` discovers the gateway's tool, the model calls `mcp__test__echo`, the harness dispatches it over the socket, and the **second** model request carries the `echo:` tool result. The assistant's text lands on **stdout**; stderr shows `mcp: connected`. |
| Gateway down → warn and continue | `TestSetupMCPWarnsAndContinuesWhenUnreachable` | MCP is enabled but the socket is dead (harness never spawns a gateway). The single 250 ms probe fails, startup **proceeds** (exit 0), emits one `[warn] [mcp]` `cannot connect to gateway … MCP tools unavailable` line, registers **zero** `mcp__` tools, and returns a no-op cleanup — MCP never fails startup. |
| HTTP gateway, one-shot round-trip | `TestSetupMCPHTTPGatewayRoundTrip` | With `mcp.gateway` set to an `http://` URL (an in-process streamable-HTTP gateway), harness **skips the probe**, connects directly, registers the tool, and a one-shot model turn calls it; the result flows back over HTTP. |
| Daemon serves socket **and** HTTP together | `TestDaemonServesSocketAndHTTP` | With `gateway.listen` set, one daemon binds both the unix socket and the TCP listener; an MCP client over each transport lists the same aggregated tools. The HTTP side uses an `Mcp-Session-Id` session and JSON-only responses. |
| `tools -gateway <url>` against the HTTP listener | `TestServeListenFlagAndToolsGateway` | `runServe -listen <addr>` brings up the HTTP listener; the `tools` subcommand with `-gateway http://<addr>` connects over HTTP and prints the same aggregated table. |

### Real downstream MCP server (BLOCKED — run by hand)

To smoke a real downstream MCP server, write a gateway config at
`~/.config/harness-mcp-gateway/config.json` (one `mcpServers` entry, stdio or
http; see the README), then:

```sh
go build ./cmd/...

# Start the gateway yourself — harness never spawns it. Leave it running:
./harness-mcp-gateway serve &
./harness-mcp-gateway tools          # prints the mcp__<server>__<tool> table

# Drive a model through an MCP tool:
HARNESS_MCP_ENABLE=true ./harness -model claude-opus-4-8 \
  -p "use an MCP tool to <task>"
```

Expect: `mcp: connected (N servers, M tools): ...` on stderr, the daemon outliving
harness (a second harness reuses it), and downstream stderr/crashes recorded in
`gateway.log` next to the socket. If the gateway is **not** running, harness emits
one `mcp: cannot connect to gateway at <path>: …; MCP tools unavailable` warning
and continues toolless.

To smoke the **HTTP** gateway path, add `"gateway": {"listen": "127.0.0.1:8420"}`
to the gateway config (or pass `serve -listen 127.0.0.1:8420`), then:

```sh
./harness-mcp-gateway serve -listen 127.0.0.1:8420 &
./harness-mcp-gateway tools -gateway http://127.0.0.1:8420   # same table over HTTP

# Point harness at the URL (config mcp.gateway = "http://127.0.0.1:8420", or env):
HARNESS_MCP_ENABLE=true HARNESS_MCP_GATEWAY=http://127.0.0.1:8420 \
  ./harness -model claude-opus-4-8 -p "use an MCP tool to <task>"
```

Expect: the same `mcp: connected` line; harness connects directly (no probe over
HTTP). The tool list is fixed at startup over HTTP — no `[mcp: tool list updated]`
notice fires.

### How the mock works

`recordingMock.ServeHTTP` decodes each `/v1/chat/completions` request body,
records it, and replies with a scripted SSE stream (OpenAI chunk shape:
`choices[].delta` for text, `choices[].delta.tool_calls[]` fragments for a tool
call, `finish_reason`, a trailing usage chunk, then `data: [DONE]`). The model
name `mock-model` infers the OpenAI dialect (design §7); the non-default
`-base-url` makes an empty API key acceptable.

## Real-API legs (BLOCKED — run by hand once credentials exist)

These exercise the live provider dialects end to end. The harness reads keys from
the environment only (design §2/§7); never pass them as flags.

### Anthropic Messages API

```sh
export ANTHROPIC_API_KEY=sk-ant-...
go build ./cmd/harness

# One-shot, assistant text captured to a file (tool summaries/usage go to stderr):
./harness -model claude-opus-4-8 -p "list the Go files in this directory using your tools" > answer.txt

# Interactive REPL (try /help, a prompt that needs a tool, then /usage, /exit):
./harness -model claude-opus-4-8
```

Expect: a per-turn usage line on stderr with a dollar cost (from configured
pricing or models.dev), tool one-liners on stderr, the final answer on stdout,
and a session auto-saved under `~/.local/state/harness/sessions/`.

### OpenAI Responses API

```sh
export OPENAI_API_KEY=sk-...
go build ./cmd/harness

./harness -model gpt-5.5 -p "read README.md and summarize it in two sentences" > answer.txt
./harness -model gpt-5.5            # interactive
```

Expect: same behavior as above. First-party OpenAI models use the Responses
dialect when models.dev identifies them. Cost appears when the model has
configured pricing or pricing can be found through models.dev; unknown model
names show token counts without a dollar figure.

### Local Ollama (OpenAI-compatible, no key)

```sh
ollama serve &                 # if not already running
ollama pull llama3.2

go build ./cmd/harness
./harness -model llama3.2 -base-url http://localhost:11434/v1 \
  -p "what files are in this directory?"
```

Expect: provider inferred as `openai` (non-`claude*` model), empty API key
accepted because the base URL is non-default, token counts with no dollar figure
(localhost skips models.dev pricing lookup). Tool reliability depends on the
local model's tool-calling support.

### Interrupt and resume against a real provider

To reproduce the interrupt/resume legs against a live API rather than the mock:

```sh
# Start a turn that will take a while, then press Ctrl-C once mid-stream:
./harness -model claude-opus-4-8 -session /tmp/s.json
> write a very long essay about distributed systems
# ^C  -> [cancelled], partial text kept; ^C again (or at the idle prompt) -> exit 130

# Resume the saved session and continue:
./harness -model claude-opus-4-8 -resume /tmp/s.json -p "continue"
```

Expect: the resumed transcript is re-sent intact; if the prior run was saved
mid-tool-call, the dangling `tool_use` is repaired with an `interrupted`
`tool_result` before the next request (design §4, §11).
