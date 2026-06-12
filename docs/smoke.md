# Smoke / verification matrix

This document records the manual smoke matrix for `harness` (design §13) and how
to re-run each leg. It complements — it does not replace — the unit and golden
suites (`go test ./...`, 711 test functions across 27 packages).

The legs split in two groups:

- **Hermetic legs** drive the real, freshly-built `harness` binary as a
  subprocess through an in-process `harness-model-proxy` whose provider config
  points at a throwaway OpenAI-compatible mock server bound to `127.0.0.1` (no
  network, no API keys). They are automated in `cmd/harness/integration_test.go`
  and run as part of `go test ./...`. The proxy and mock live only in `_test.go`,
  so they are never compiled into the shipped binaries.
- **Real-API legs** require provider credentials and are **BLOCKED** in this
  environment (no proxy-accessible `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, no
  local Ollama). They are documented below with the exact commands to run them by
  hand.

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

### MCP proxy legs (automated, PASS)

These exercise the optional MCP proxy end to end without a network or any real
downstream server: a fake in-process proxy (or the real `harness-mcp-proxy`
serve loop driven against a fake downstream) stands in. They live in
`cmd/harness/mcp_test.go`, `cmd/harness-mcp-proxy/main_test.go`, and
`internal/mcpproxy/daemon_test.go`, and run under `go test ./...`.

```sh
go test ./cmd/harness/ -run TestSetupMCP -v
go test ./cmd/harness-mcp-proxy/ -run 'TestServe|TestTools' -v
go test ./internal/mcpproxy/ -run TestDaemonServesHTTP -v
```

| Leg | Test | What it asserts |
|---|---|---|
| Proxy `serve` + `tools` listing | `TestToolsListsAggregatedTools` | `runServe` binds an HTTP listener, supervises a fake downstream, and aggregates its tools; the `tools` subcommand connects and prints `2 tools` with the namespaced names `mcp__fake__echo` / `mcp__fake__ping`, descriptions collapsed to their first line. A `SIGINT` shuts the daemon down cleanly. |
| One-shot calling an `mcp__` tool | `TestSetupMCPRegistersToolsAndOneShotCalls` | With `HARNESS_MCP_ENABLE=true` and an HTTP proxy URL in env, `harness -p` discovers the proxy's tool, the model calls `mcp__test__echo`, the harness dispatches it over HTTP, and the **second** model request carries the `echo:` tool result. The assistant's text lands on **stdout**; stderr shows `mcp: connected`. |
| Non-HTTP proxy → warn and continue | `TestSetupMCPRejectsNonHTTPProxyAndContinues` | MCP is enabled but the proxy value is not an `http(s)` URL. Startup **proceeds** (exit 0), emits one `[warn] [mcp]` `cannot connect to proxy … MCP tools unavailable` line, registers **zero** `mcp__` tools, and returns a no-op cleanup — MCP never fails startup. |
| HTTP proxy down → warn and continue | `TestSetupMCPHTTPUnreachableWarnsAndContinues` | With `mcp.proxy` set to a closed `http://` URL, harness attempts registration, emits one warning, and continues without MCP tools. |
| Daemon serves HTTP | `TestDaemonServesHTTP` | With `proxy.listen` set, one daemon binds the TCP listener; an MCP client lists the aggregated tools and calls one. The HTTP side uses an `Mcp-Session-Id` session and JSON-only responses. |
| `tools -proxy <url>` against the HTTP listener | `TestServeListenFlagAndToolsProxy` | `runServe -listen <addr>` brings up the HTTP listener; the `tools` subcommand with `-proxy http://<addr>` connects over HTTP and prints the same aggregated table. |

### Real downstream MCP server (BLOCKED — run by hand)

To smoke a real downstream MCP server, write a proxy config at
`~/.config/harness-mcp-proxy/config.json` (one `mcpServers` entry, stdio or
http; see the README), then:

```sh
go build ./cmd/...

# Start the proxy yourself — harness never spawns it. Leave it running:
./harness-mcp-proxy serve &
./harness-mcp-proxy tools          # prints the mcp__<server>__<tool> table

# Drive a model through an MCP tool:
HARNESS_MCP_ENABLE=true ./harness -model claude-opus-4-8 \
  -p "use an MCP tool to <task>"
```

Expect: `mcp: connected (N servers, M tools): ...` on stderr, the daemon outliving
harness (a second harness reuses it), and downstream stderr/crashes recorded in the
proxy log. If the proxy is **not** running, harness emits one
`mcp: cannot connect to proxy at http://127.0.0.1:8766: …; MCP tools unavailable`
warning and continues toolless.

To smoke a non-default proxy address, add
`"proxy": {"listen": "127.0.0.1:8420"}` to the proxy config (or pass
`serve -listen 127.0.0.1:8420`), then:

```sh
./harness-mcp-proxy serve -listen 127.0.0.1:8420 &
./harness-mcp-proxy tools -proxy http://127.0.0.1:8420

# Point harness at the URL (config mcp.proxy = "http://127.0.0.1:8420", or env):
HARNESS_MCP_ENABLE=true HARNESS_MCP_PROXY=http://127.0.0.1:8420 \
  ./harness -model claude-opus-4-8 -p "use an MCP tool to <task>"
```

Expect: the same `mcp: connected` line. The tool list is fixed at startup over
HTTP — no `[mcp: tool list updated]` notice fires.

### How the mock works

`startModelProxy` creates a temp proxy config with one `openai` provider whose
base URL points at `recordingMock`. The subprocess is invoked with
`-model-proxy-url`, so the tested path is `harness -> harness-model-proxy ->
provider dialect -> mock endpoint`.

`recordingMock.ServeHTTP` decodes each `/v1/chat/completions` request body,
records it, and replies with a scripted SSE stream (OpenAI chunk shape:
`choices[].delta` for text, `choices[].delta.tool_calls[]` fragments for a tool
call, `finish_reason`, a trailing usage chunk, then `data: [DONE]`).

## Real-API legs (BLOCKED — run by hand once credentials exist)

These exercise the live provider dialects end to end through
`harness-model-proxy`. Start the proxy in a separate shell first; `harness`
should receive only the proxy URL, provider id, and model id.

```sh
go build -o harness ./cmd/harness
go build -o harness-model-proxy ./cmd/harness-model-proxy
./harness-model-proxy --setup
./harness-model-proxy
```

### Anthropic Messages API

```sh
export ANTHROPIC_API_KEY=sk-ant-...
# Start or restart harness-model-proxy in this environment after setup.

# One-shot, assistant text captured to a file (tool summaries/usage go to stderr):
./harness -provider anthropic -model claude-opus-4-8 \
  -p "list the Go files in this directory using your tools" > answer.txt

# Interactive REPL (try /help, a prompt that needs a tool, then /usage, /exit):
./harness -provider anthropic -model claude-opus-4-8
```

Expect: a per-turn usage line on stderr with a dollar cost (from configured
pricing or models.dev), tool one-liners on stderr, the final answer on stdout,
and a session auto-saved under `~/.local/state/harness/sessions/`.

### OpenAI Responses API

```sh
export OPENAI_API_KEY=sk-...
# Start or restart harness-model-proxy in this environment after setup.

./harness -provider openai -model gpt-5.5 \
  -p "read README.md and summarize it in two sentences" > answer.txt
./harness -provider openai -model gpt-5.5            # interactive
```

Expect: same behavior as above. First-party OpenAI models use the Responses
dialect when models.dev identifies them. Cost appears when the model has
configured pricing or pricing can be found through models.dev; unknown model
names show token counts without a dollar figure.

### Local Ollama (OpenAI-compatible, no key)

```sh
ollama serve &                 # if not already running
ollama pull llama3.2

mkdir -p ~/.config/harness-model-proxy
cat > ~/.config/harness-model-proxy/config.json <<'JSON'
{
  "provider_configs": ["ollama.json"]
}
JSON
cat > ~/.config/harness-model-proxy/ollama.json <<'JSON'
{
  "name": "ollama",
  "api_type": "openai",
  "base_url": "http://localhost:11434/v1",
  "models": [{"name": "llama3.2", "context_window": 131072}]
}
JSON

./harness-model-proxy
./harness -provider ollama -model llama3.2 -p "what files are in this directory?"
```

Expect: the proxy uses the OpenAI-compatible dialect with an empty API key,
token counts with no dollar figure, and tool reliability depending on the local
model's tool-calling support.

### Interrupt and resume against a real provider

To reproduce the interrupt/resume legs against a live API rather than the mock:

```sh
# Start a turn that will take a while, then press Ctrl-C once mid-stream:
./harness -provider anthropic -model claude-opus-4-8 -session /tmp/s.json
> write a very long essay about distributed systems
# ^C  -> [cancelled], partial text kept; ^C again (or at the idle prompt) -> exit 130

# Resume the saved session and continue:
./harness -provider anthropic -model claude-opus-4-8 -resume /tmp/s.json -p "continue"
```

Expect: the resumed transcript is re-sent intact; if the prior run was saved
mid-tool-call, the dangling `tool_use` is repaired with an `interrupted`
`tool_result` before the next request (design §4, §11).
