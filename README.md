# harness

A minimal agentic coding harness in Go: a plain-text, line-oriented CLI that
drives a tool-using LLM loop against local files, shell commands, and git.

## What it is

- **Small and legible.** The whole system is meant to be readable in an
  afternoon — one purpose per package, no framework.
- **Zero third-party Go dependencies.** Go standard library only. SSE parsing,
  diff application, HTML-to-text reduction, and retries are all small enough to
  own; generic Unix capabilities are delegated to host CLIs where that is the
  simpler, battle-tested path.
- **Generic over providers.** One internal message/streaming model with three
  HTTP dialects: **Anthropic Messages**, **OpenAI Responses**, and **OpenAI Chat
  Completions**. Responses is the default for first-party OpenAI models when
  models.dev identifies them; Chat Completions remains the OpenAI-compatible path
  for vLLM, Ollama, OpenRouter, llama.cpp, and custom base URLs.
- **No sandbox, no permission prompts.** The harness assumes it is launched
  inside an already-sandboxed environment; tools run with the process's
  privileges, immediately.
- **First-class git.** A dedicated `git` tool plus a git summary in the system
  prompt.

The full behavioral specification lives in [`docs/design.md`](docs/design.md).
The end-to-end verification matrix is in [`docs/smoke.md`](docs/smoke.md).

## Build

```sh
go build -o harness ./cmd/harness
go build -o harness-mcp-gateway ./cmd/harness-mcp-gateway
```

`make build` builds the same two binaries. `harness-mcp-gateway` is only needed
for the optional [MCP servers](#mcp-servers-optional) integration; `go build -o
harness ./cmd/harness` alone produces just the main binary.

Requires Go 1.24+ (the toolchain uses range-over-func). Verify a checkout with:

```sh
go build ./... && go vet ./... && go test ./...
```

## Quickstart

API keys are read from environment variables or provider config files — never
from flags or the main config file, because those leak into shell history and
committed dotfiles.

### Anthropic

```sh
export ANTHROPIC_API_KEY=sk-ant-...
./harness -model claude-opus-4-8                       # interactive REPL
./harness -model claude-opus-4-8 -p "summarize README.md"   # one-shot
```

### OpenAI

```sh
export OPENAI_API_KEY=sk-...
./harness -model gpt-5.5
./harness -model gpt-5.5 -p "what files are in internal/?"
```

### A local OpenAI-compatible server (Ollama)

No key is needed when the base URL is non-default (local servers need none):

```sh
ollama serve &
ollama pull llama3.2
./harness -model llama3.2 -base-url http://localhost:11434/v1
```

### Session replay

```sh
./harness session replay ~/.local/state/harness/sessions/20260611T123456Z
```

The base URL supplies scheme/host/prefix only; the dialect appends its standard
path (`/responses`, `/chat/completions`, or `/messages`).

### Provider selection

`-model` is primary. The provider is **inferred** from the model name: anything
starting with `claude` uses the Anthropic dialect, everything else uses the
OpenAI-compatible dialect unless models.dev identifies the selected first-party
OpenAI model, in which case the Responses dialect is used. Custom and local base
URLs stay on Chat Completions unless `-provider responses` or a provider config
with `api_type: "responses"` selects Responses explicitly. A model value like
`openrouter:openai/gpt-5.5` selects the configured `openrouter` provider while
sending `openai/gpt-5.5` as the provider-local model id.

### One-shot mode

In one-shot mode (`-p`) the **assistant's text goes to stdout** while model
progress, tool-call progress, tool summaries, the usage line, notices, and errors
go to stderr — so you can capture exactly the answer:

```sh
./harness -model gpt-5.5 -p "explain this repo in one paragraph" > answer.txt
```

`-p -`, or piping into stdin, reads the prompt from stdin; with both a flag value
and piped stdin they are concatenated, so `harness -p "summarize:" < notes.txt`
works. Exit codes: `0` completed, `1` runtime error, `2` usage error, `130`
interrupted.

## Flags

```
-p <prompt|->     one-shot mode; "-" or piped stdin reads the prompt from stdin
-provider <name>  openai | responses | anthropic, or a configured provider name
                  (default: inferred from -model and models.dev when available)
-model <id>       model id (required)
-base-url <url>   provider base URL (e.g. http://localhost:11434/v1 for Ollama)
-system <text|@file>           append to the system prompt (project notes)
-system-override <text|@file>  replace the builtin instructions
-no-env           omit the environment context block (cwd/os/date/git)
-resume <file>    load a session transcript and continue
-session <file>   explicit session save path
-max-steps <n>    model round-trips per user turn (default 50)
-default-context-window <n>   fallback window for unknown/unconfigured models (default 256000)
-context-window <n>   override the model's context window (tokens)
-reasoning-effort <level>   reasoning/thinking effort when supported
-mode <name>      run mode: auto (default), plan, independent, or a config-defined mode
-v                show tool result snippets (first ~5 lines, dimmed)
-tool-stream      show live tool-call progress (default true; use -tool-stream=false to disable)
-q, --quiet       suppress informational diagnostics
--log-level <level>  diagnostic log level: debug, info, warn, error (also LOG_LEVEL)
-no-color         disable color (also: NO_COLOR env var; color is TTY-only anyway)
-config <file>    alternate config path
--setup           create or update config in ~/.config/harness
--force           with --setup, overwrite existing provider files and defaults
--refresh-models  fetch models.dev and update configured provider model metadata
-h, --help        print this usage screen and exit 0
```

`-system`/`-system-override` accept a `@file` reference (a literal leading `@` is
escaped as `@@`).

### Configuration and environment

Precedence is **flags > environment > config file > built-in defaults**.

- Environment: `OPENAI_API_KEY`, `RESPONSES_API_KEY`, `ANTHROPIC_API_KEY`,
  `OPENAI_BASE_URL`, `RESPONSES_BASE_URL`, `ANTHROPIC_BASE_URL`, plus `HARNESS_*`
  equivalents for most flags
  (`HARNESS_MODEL`, `HARNESS_MAX_STEPS`, `HARNESS_DEFAULT_CONTEXT_WINDOW`, …).
  Environment API keys override keys from provider config files.
- Optional config file at `~/.config/harness/config.json` (override with
  `-config`): `provider`, `model`, `provider_configs`, `mode`, `modes` (see
  [Run modes](#run-modes)), and flag defaults. Provider
  config paths are resolved relative to the config file and may define `api_type`
  (`responses`, `openai`, or `anthropic`), `base_url`, `api_key`, `api_key_env`,
  models, context windows, reasoning metadata, and pricing. The
  `default_context_window` fallback is used only when a model has no
  configured context window; `context_window` forces an override. See
  `examples/config/` for sample files.
- Context-efficiency knobs are config-file only: `agents_md_warn_bytes`
  (default `8192`, warning only; `AGENTS.md` is still included in full),
  `tool_result_max_bytes`, `tool_result_max_lines`, `read_file_default_limit`,
  `compact_keep_turns`, `compact_summary_max_tokens`, and
  `compact_tool_result_max_bytes`. The read-only delegate tool also has
  `delegate_max_steps` (default `20`) as a config-file-only cap.
- If `reasoning_effort` / `HARNESS_REASONING_EFFORT` / `-reasoning-effort` is set,
  harness sends the provider-specific effort field only when requested. Known
  models.dev metadata is used to reject unsupported models or effort values; unknown
  local models are left to the provider.
- If a model is missing context-window, pricing, or needed reasoning metadata locally,
  harness makes a best-effort lookup against `https://models.dev/api.json` and uses
  the discovered model metadata when available. That lookup also promotes
  first-party OpenAI models to the Responses dialect. Localhost base URLs skip it.
- Run `./harness --setup` to create a default config and a provider config from
  models.dev, or append a new provider config to an existing default config
  without changing existing defaults. Setup lists harness-supported providers,
  prompts for the API key, pages the provider's models newest-first, and asks
  which model should be the default. The provider file includes all models known
  for that provider: URL, API type, key env vars, context windows, prices, and
  reasoning metadata come from models.dev. If the live catalog is unreachable,
  setup uses a vendored models.dev snapshot. Existing provider config files and
  existing default provider/model settings are not overwritten unless `--force`
  is set.
- Run `./harness --refresh-models` to fetch the latest live `models.dev` catalog
  and regenerate every provider config referenced by the config file, preserving
  stored API keys. Unlike setup, refresh fails if models.dev is inaccessible.

## Meta-commands (REPL)

Lines starting with `/` are commands; `//` sends a literal leading slash.
In terminals that support bracketed paste, pasted text is submitted as one
literal prompt, preserving embedded newlines; pasted `/commands` are not
executed as meta-commands. For non-interactive large input, prefer `-p -` or
piped stdin.

Press **Ctrl-G** at the prompt, or run `/edit [draft]`, to open an external
editor for a multi-line prompt. Harness uses `$VISUAL`, then `$EDITOR`, then
`vi`. The temp file preloads the visible output from the previous turn, followed
by a delimiter; only text written after the delimiter is sent as the next prompt.

| command | effect |
|---|---|
| `/help` | list commands |
| `/exit`, `/quit` | save and exit |
| `/clear` | reset the conversation; rotate to a fresh session directory |
| `/compact` | force compaction now |
| `/usage` | cumulative session tokens and cost |
| `/edit [draft]` | open an external editor for the next prompt |
| `/save [file]` | force save (optionally elsewhere) |
| `/model` | choose a configured provider, then choose one of its configured models |
| `/model <id>` | switch subsequent turns to model `<id>` |
| `/model <provider>:<id>` | switch to `<id>` on a specific configured provider |
| `/mode` | list run modes, marking the current one |
| `/mode <name>` | switch the active run mode |

## Run modes

A **run mode** bundles a set of allowed tools with extra system-prompt
instructions, so the same harness can plan, work autonomously, or run wide open.
Select one with `-mode <name>`, `HARNESS_MODE`, or `mode` in the config file
(default `auto`); switch mid-session with `/mode <name>`. The active mode is
saved with the session and restored on `-resume` (a `-mode` flag still wins).

Three modes are built in:

| mode | tools | behavior |
|---|---|---|
| `auto` | all available tools, including `delegate` | the default — the model decides what to do |
| `plan` | inspection tools (`read_file`, `list_dir`, `grep`, optional `rg`, `web_fetch`, optional `git_readonly`), `write_tmp_file`, and `delegate` | collaborate on a plan without modifying the project |
| `independent` | all available tools, including `delegate` | complete the task end-to-end without pausing for input; stop only on a hard blocker or the step limit |

Define new modes or override built-ins in the config file under `modes`. Entries
**field-level merge** onto a built-in of the same name — an omitted field keeps
the built-in value, so you can retune just a prompt or just a tool list:

```json
{
  "mode": "plan",
  "modes": {
    "plan":   { "prompt": "@~/.config/harness/plan-prompt.md" },
    "review": {
      "allowed_tools": ["read_file", "list_dir", "grep", "git_readonly"],
      "prompt": "You are a code reviewer. Read the diff and surrounding code, then report findings. Do not modify files."
    }
  }
}
```

A mode with no `allowed_tools` gets the full tool set; a mode `prompt` may be a
`@file` reference. Unknown mode names and unknown tool names are reported at
startup. This tool gating is the one place the harness restricts tools — the
underlying tools still assume an external sandbox for real isolation.

## Sessions

- A session path is a **directory**. `state.json` is the compact resumable state,
  `raw.ndjson` is an append-only replay log, `compactions/` stores raw messages
  removed from active context, and `artifacts/tool-results/` stores full outputs
  omitted from model context.
- The compact state is **saved after every turn**, atomically (write a `.tmp`,
  then rename `state.json`). Auto-save uses
  `~/.local/state/harness/sessions/<timestamp>` (honoring `$XDG_STATE_HOME`);
  the path is printed at startup.
- `-session <dir>` chooses an explicit session directory. `-resume <dir>` loads
  its `state.json` and continues; `/clear` rotates to a fresh directory.
- Transcripts are **provider-neutral**, so a session started against Anthropic
  resumes against an OpenAI-compatible server and vice versa. When flags disagree
  with a resumed directory's provider/model, the flags win with a warning.
- A session saved mid-turn (a dangling `tool_use`) is repaired on load by
  synthesizing an `interrupted` tool result, so the resumed transcript is always
  valid for both APIs.
- `harness session replay <session-dir>` prints the user-facing session view to
  stdout for inspection or grep.

## Compaction

When a turn's reported input tokens reach **78%** of the model's context window
(or on `/compact`), the harness summarizes the conversation to free context. The
trigger uses an approximate full-request footprint (system prompt, tools, and
messages). It keeps the system prompt and the configured number of recent turns
(`compact_keep_turns`, default `4`) verbatim, sends everything older to the model
with a summarization instruction, and replaces the old messages with a single
summary message. Summary output is capped by `compact_summary_max_tokens`
(default `2048`). The summary call's tokens and cost are folded into the session
totals and reported:

```
[compacted: 38 messages → summary · 9.1k in / 0.4k out · $0.05]
```

Before summarization, large old tool results are reduced to previews
(`compact_tool_result_max_bytes`, default `4096`) and the raw removed messages are
archived under `compactions/`. If the old history is too large for one summary
call, harness summarizes chunks and then summarizes the chunk summaries. If the
transcript is still over budget it degrades further (keep only the last turn,
then hard-truncate the largest tool results) — it never wedges. If the summary or
archive step fails, compaction is aborted and the full transcript is kept.

Turn summaries include an approximate context footprint, for example:

```text
[turn: 3 steps · 12.4k in / 1.8k out · $0.071 · ctx 42.0k/256.0k (sys 2.0k tools 1.5k msgs 38.5k) · 4.3s]
```

## Interrupts

- **Ctrl-C during a turn**, or **Esc twice in short succession during a REPL
  turn**, cancels the turn (aborting the HTTP stream and killing any
  `run_command` process group); streamed partial text is kept and un-executed
  tool calls are stripped. Prints `[cancelled]` and returns to the prompt.
- **A second Ctrl-C within ~1s, or Ctrl-C at the idle prompt** saves and exits
  130. **Ctrl-D** at the prompt saves and exits 0.

## Tools

`read_file`, `list_dir`, `grep`, optional `rg` when ripgrep is installed,
`edit`, `write_file`, `apply_patch`, `run_command`, `exec`, optional `git`
when git is installed, `web_fetch`, `delegate`, plus two used by restricted modes: optional
`git_readonly` (read-only git
subcommands — `status`, `log`, `diff`, `show`, `grep`, `blame`, `bisect`) and
`write_tmp_file` (write scratch files under a private per-run temp directory).
`delegate` starts a child read-only agent with inspection tools only and returns
its final report as a normal tool result; the child transcript is not persisted
into the parent session, but child token usage is included in turn/session usage.
Missing optional CLI-backed tools are reported on stderr at startup, e.g.
`[warn] [cli_tools] Tool "rg" is disabled. Reason: "rg" binary not found.`
unless `-q`/`--quiet` or `--log-level error` suppresses the warning. `grep`,
`rg`, and `git` are thin argv wrappers around the host CLIs. When the optional
[MCP servers](#mcp-servers-optional) integration is enabled, downstream MCP tools
also appear, namespaced as `mcp__<server>__<tool>`. See
[`docs/design.md`](docs/design.md) §9 for each tool's schema and exact
behavior.

Tool results are centrally capped (default 64 KB or 1000 lines; configurable via
`tool_result_max_bytes` / `tool_result_max_lines`). Truncated results include a
marker in the model-visible text, a warning in the UI, and the full output is
archived under the session directory when available.

## MCP servers (optional)

Harness can expose tools from [Model Context Protocol](https://modelcontextprotocol.io)
servers. A second binary, `harness-mcp-gateway`, owns every downstream MCP server
(spawning stdio children, dialing streamable-HTTP endpoints) and aggregates their
tools into one namespaced surface; harness connects to that gateway — over a unix
socket by default, or [over HTTP](#talking-to-a-gateway-over-http) — and registers
each tool as an ordinary harness tool. Harness and the gateway speak MCP to each
other (JSON-RPC 2.0, revision `2025-06-18`), so the gateway is a single shared
daemon that many harness sessions reuse. You start the gateway yourself; harness
never spawns it.

### Enabling it

MCP is **opt-in** and off by default. Turn it on in `~/.config/harness/config.json`:

```json
{
  "mcp": {
    "enable": true,
    "gateway": ""
  }
}
```

or via environment: `HARNESS_MCP_ENABLE=true` and (optionally)
`HARNESS_MCP_GATEWAY=/path/to/gateway.sock`. There are no flags. An empty
`gateway` resolves the shared default socket path (below). Precedence is the usual
**env > config file > default**.

`gateway` accepts either a unix socket path or an `http(s)://` gateway URL (see
[Talking to a gateway over HTTP](#talking-to-a-gateway-over-http)). A unix path
may point anywhere you like; the daemon never touches permissions — it creates a
missing parent directory (umask default) and uses a pre-existing parent as-is.
The only hard limit is the OS `sun_path` length (~104 bytes), so keep the path
short — the default `/tmp/harness-mcp-gateway-$UID/gateway.sock` stays well
under it.

### Configuring downstream servers

The gateway has its own config file, **separate from harness**, at
`$XDG_CONFIG_HOME/harness-mcp-gateway/config.json` (else
`~/.config/harness-mcp-gateway/config.json`). It is Claude Code-compatible:

```json
{
  "mcpServers": {
    "fs": {
      "command": "mcp-server-filesystem",
      "args": ["--root", "/srv/data"],
      "env": { "LOG_LEVEL": "info" }
    },
    "search": {
      "type": "http",
      "url": "https://mcp.example.com/mcp",
      "headers": { "Authorization": "Bearer ${SEARCH_TOKEN}" }
    }
  },
  "gateway": {
    "socket": "",
    "listen": "",
    "logFile": "",
    "logLevel": "info"
  }
}
```

`gateway.listen` is empty by default (unix socket only); set it to an address such
as `127.0.0.1:8420` to also serve over HTTP (see
[Talking to a gateway over HTTP](#talking-to-a-gateway-over-http)). A server with no
`type` (or `"stdio"`) is a child process (`command`/`args`/`env`);
`"http"` is a streamable-HTTP endpoint (`url`/`headers`). `${NAME}` references in
any string are expanded from the gateway's environment (a literal `$` is preserved;
an unset variable warns and expands to empty). Invalid server entries are skipped
with a warning, never fatal — the gateway still serves the valid ones. See
`examples/config/harness-mcp-gateway.json` for a copyable starting point.

Stdio servers inherit the gateway's **full environment** — whatever environment
the `harness-mcp-gateway serve` process was started with — plus the per-server
`env` overrides. Do not configure untrusted stdio servers when secrets live in the
environment, since the child process can read them.

### Running the gateway

Harness never starts the gateway for you — you run it yourself, once, and leave it
up. The daemon is a single shared process that **outlives harness** and is **shared
across sessions**: starting a second harness reuses the same gateway, and a second
`serve` exits quietly if one is already running.

```sh
harness-mcp-gateway serve &        # foreground without the & to watch its logs
```

For a persistent setup, run it from your shell profile, a launchd agent (macOS),
or a systemd user unit (Linux) so it comes up at login.

When MCP is enabled, harness does a **single 250 ms probe** of the gateway socket
at startup. If the probe succeeds it connects and registers the gateway's tools,
logging a line such as `mcp: connected (2 servers, 5 tools): fs=3 search=2`. If
the probe fails it emits **one** warning and continues with no MCP tools:

```
mcp: cannot connect to gateway at /tmp/harness-mcp-gateway-1000/gateway.sock: <err>; MCP tools unavailable
```

MCP **never fails harness startup**. The startup cost is bounded by that 250 ms
probe for a dead unix socket; the only longer wait is the 5 s registration timeout,
reachable only when the socket binds but the gateway then hangs during
`initialize`/`tools/list` (an HTTP gateway, which skips the probe, connects directly
under the same 5 s bound).

Default paths (all derived per-user):

- **Socket:** `$XDG_RUNTIME_DIR/harness-mcp-gateway/gateway.sock`, else
  `<tmpdir>/harness-mcp-gateway-<uid>/gateway.sock` (a short hashed name under
  `<tmpdir>` if that would exceed the OS socket-path limit).
- **Config:** `$XDG_CONFIG_HOME/harness-mcp-gateway/config.json`, else
  `~/.config/harness-mcp-gateway/config.json`.
- **Log:** stderr when serving interactively, otherwise `gateway.log` next to the
  socket (so a detached daemon never loses output).

Inspect the live surface without harness with:

```sh
harness-mcp-gateway tools                       # over the unix socket
harness-mcp-gateway tools -gateway http://127.0.0.1:8420   # over an HTTP gateway
```

### Talking to a gateway over HTTP

The gateway can serve its merged MCP surface over **streamable HTTP** alongside the
unix socket, so harness sessions on another host (or behind a proxy) can reach it.
Set `gateway.listen` in the gateway config, or pass `serve -listen`:

```json
{ "gateway": { "listen": "127.0.0.1:8420" } }
```

```sh
harness-mcp-gateway serve -listen 127.0.0.1:8420 &
```

The listener speaks **plain HTTP only** — put a reverse proxy (nginx, Caddy) in
front for TLS and any stronger auth. Each session is carried by an `Mcp-Session-Id`
header with a 30-minute idle TTL; responses are `application/json` only, and there
is **no server-push channel**, so a client re-lists on its own rather than being
told of tool changes.

On the harness side, point `mcp.gateway` at the URL and (for a proxied gateway that
wants auth) add a config-file-only `mcp.headers` map sent on **every** request:

```json
{
  "mcp": {
    "enable": true,
    "gateway": "https://mcp.internal.example/mcp",
    "headers": { "Authorization": "Bearer ${TOKEN}" }
  }
}
```

`headers` has no environment variable — it lives in the config file alongside the
URL it authenticates to. Header values expand `${VAR}` and `${VAR:-default}`;
literal dollar forms such as `$5` and `$$` are preserved. An unset `${VAR}` is a
config error. Harness does **not** probe an `http(s)` gateway (it
connects directly, bounded by the 5 s registration timeout). Because HTTP has no
notifications channel, the tool list is **fixed at startup over HTTP**: the
`[mcp: tool list updated]` notice never fires for an HTTP gateway.

### Tools, modes, and limits

Aggregated tools are named `mcp__<server>__<tool>` (the charset/length must fit
`[a-zA-Z0-9_-]{1,64}`; names that do not are dropped with a warning). They are
plain harness tools, so they flow through the normal truncation, artifact, and
session paths. Modes that inherit the default tool set (`auto`, `independent`, and
config modes without an explicit `allowed_tools`) expose the MCP tools; a mode
with an explicit `allowed_tools` whitelist does **not** (it may list `mcp__` names
manually). Over a **unix-socket** gateway, if a server announces a tool-list
change the REPL applies it at the next prompt boundary with a
`[mcp: tool list updated; N tools]` notice; one-shot runs ignore mid-run changes,
and an HTTP gateway has no notifications channel so its tool list is fixed at
startup.

One-shot users should note the startup cost is small: a single 250 ms probe for a
dead unix socket, or the 5 s registration timeout if the gateway binds but then
hangs during `initialize`/`tools/list` (an HTTP gateway connects directly under
the same 5 s bound). Leave MCP off (the default) for latency-sensitive one-shot
invocations that do not need it.

**v1 non-goals:** tools only (no MCP resources or prompts); streamable-HTTP only
for remote servers (no legacy HTTP+SSE transport); header-based auth only (no
OAuth flow); the HTTP gateway listener is plain HTTP (TLS via a reverse proxy).
