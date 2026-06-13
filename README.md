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
- **Provider access is isolated.** `harness` talks to a local
  `harness-model-proxy` over HTTP. The proxy owns API keys, provider configs,
  model metadata, and the Anthropic/OpenAI dialects; the main CLI only sees a
  provider/model catalog and normalized stream events.
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
go build -o harness-model-proxy ./cmd/harness-model-proxy
go build -o harness-mcp-proxy ./cmd/harness-mcp-proxy
```

`make build` builds the same binaries. `harness-mcp-proxy` is only needed
for the optional [MCP servers](#mcp-servers-optional) integration; `go build -o
harness ./cmd/harness` alone produces just the main binary.

Requires Go 1.24+ (the toolchain uses range-over-func). Verify a checkout with:

```sh
go build ./... && go vet ./... && go test ./...
```

## Quickstart

Configure and start the model proxy first. API keys are read by
`harness-model-proxy` from its environment or provider config files; the
`harness` process does not need direct access to them.

```sh
harness-model-proxy --setup
harness-model-proxy
```

Then run `harness` against the proxy catalog:

```sh
./harness -provider anthropic -model claude-opus-4-8
./harness -model openrouter:openai/gpt-5.5
./harness -provider openai -model gpt-5.5 -p "summarize README.md"
```

`harness-model-proxy` listens on `127.0.0.1:8765` by default. Use
`harness -model-proxy-url http://host:port` when the proxy runs elsewhere.

### Session replay

```sh
./harness session replay ~/.local/state/harness/sessions/20260611T123456Z
```

### Provider selection

`harness` fetches providers and models from `harness-model-proxy`. A model value
like `openrouter:openai/gpt-5.5` selects the proxy provider `openrouter` while
sending `openai/gpt-5.5` as the provider-local model id. Model selection belongs
to `harness`: use `-model`, `HARNESS_MODEL`, config `model`, or `/model` in the
REPL.

### One-shot mode

In one-shot mode (`-p`) the **assistant's text goes to stdout** while model
progress, tool-call progress, tool summaries, the usage line, notices, and errors
go to stderr. Bracketed status lines are timestamped by default; disable them
when you want untimestamped diagnostics:

```sh
./harness -model gpt-5.5 -timestamps=none -p "explain this repo in one paragraph" > answer.txt
```

`-p -`, or piping into stdin, reads the prompt from stdin; with both a flag value
and piped stdin they are concatenated, so `harness -p "summarize:" < notes.txt`
works. Exit codes: `0` completed, `1` runtime error, `2` usage error, `130`
interrupted.

## Flags

```
-p <prompt|->     one-shot mode; "-" or piped stdin reads the prompt from stdin
-provider <name>  model proxy provider id
-model <id>       model id
-model-proxy-url <url>   harness-model-proxy URL (default http://127.0.0.1:8765)
-system <text|@file>           append to the system prompt (project notes)
-system-override <text|@file>  replace the builtin instructions
-no-env           omit the environment context block (cwd/os/date/git)
-resume <file>    load a session transcript and continue
-session <file>   explicit session save path
-max-turns <n>    model turns per user prompt; <=0 means unlimited (default 250)
-default-context-window <n>   fallback window for unknown/unconfigured models (default 256000)
-context-window <n>   override the model's context window (tokens)
-reasoning-effort <level>   reasoning/thinking effort when supported
-mode <name>      run mode: auto (default), plan, independent, or a config-defined mode
-v                show tool result snippets (first ~5 lines, dimmed)
-tool-stream      show live tool-call progress (default true; use -tool-stream=false to disable)
-q, --quiet       suppress informational diagnostics
--log-level <level>  diagnostic log level: debug, info, warn, error (also LOG_LEVEL)
-no-color         disable color (also: NO_COLOR env var; color is TTY-only anyway)
-timestamps <mode>  bracketed status timestamps: short (default), full/long, or none
-no-timestamps   alias for -timestamps=none
-prompt <text>    REPL input prompt string (default "> ")
-config <file>    alternate config path
-h, --help        print this usage screen and exit 0
```

`-system`/`-system-override` accept a `@file` reference (a literal leading `@` is
escaped as `@@`).

### Configuration and environment

Precedence is **flags > environment > config file > built-in defaults**.

- Environment: `HARNESS_MODEL_PROXY_URL`, `HARNESS_PROVIDER`, `HARNESS_MODEL`,
  `HARNESS_MAX_TURNS`, `HARNESS_DEFAULT_CONTEXT_WINDOW`, `HARNESS_TIMESTAMPS`,
  and other `HARNESS_*` equivalents for user-facing flags. `HARNESS_NO_TIMESTAMPS`
  is an alias for `HARNESS_TIMESTAMPS=none`. Provider API-key environment variables are
  read only by `harness-model-proxy`.
- Optional config file at `~/.config/harness/config.json` (override with
  `-config`): `model_proxy_url`, `provider`, `model`, `mode`, `modes` (see
  [Run modes](#run-modes)), and flag defaults. The `default_context_window`
  fallback is used only when proxy metadata has no configured context window;
  `context_window` forces an override. See `examples/harness/config.json`
  for a complete schema reference with the effective defaults.
- Context-efficiency knobs are config-file only: `agents_md_warn_bytes`
  (default `8192`, warning only; `AGENTS.md` is still included in full),
  `tool_result_max_bytes`, `tool_result_max_lines`, `read_file_default_limit`,
  `compact_keep_turns`, `compact_summary_max_tokens`, and
  `compact_tool_result_max_bytes`. The read-only delegate tool also has
  `delegate_max_turns` (default `20`) as a config-file-only cap.
- If `reasoning_effort` / `HARNESS_REASONING_EFFORT` / `-reasoning-effort` is set,
  harness sends it to the proxy only when requested. Proxy catalog metadata is
  used to reject unsupported models or effort values.
- Run `./harness-model-proxy --setup` to create a proxy config and a provider
  config from models.dev, append a new provider config to an existing proxy config,
  or update an existing configured provider without configuring a proxy default
  model. Setup lists harness-supported providers and marks existing providers with
  bold text and `*`, prompts for the API key, then lets you choose which provider
  models are available locally. New providers start with no models enabled;
  existing providers start with their configured models enabled and all other
  catalog models disabled. The model selector marks enabled rows with bold text and
  `*`; enter a
  number or id to toggle a model, `all` or `none` to bulk-select the model list,
  `save` to write the allowlist, or `cancel` to quit without saving. The provider
  file includes only enabled models, with URL, API type, key env vars, context
  windows, prices, and reasoning metadata from models.dev. If the live catalog is
  unreachable, setup uses a vendored models.dev snapshot. Existing provider config
  files that are not already referenced by the proxy config are not overwritten
  unless `--force` is set.
- For hand-written model-proxy config shape references, see
  `examples/harness-model-proxy/config.json` and
  `examples/harness-model-proxy/providers.json`; setup remains the
  recommended way to create real provider allowlists.
- Run `./harness-model-proxy --refresh-models` to fetch the latest live
  `models.dev` catalog and refresh metadata for the currently configured model
  allowlists, preserving stored API keys. Unlike setup, refresh fails if models.dev
  is inaccessible.

## Meta-commands (REPL)

Lines starting with `/` are commands; `//` sends a literal leading slash.
In terminals that support bracketed paste, pasted text is submitted as one
literal prompt, preserving embedded newlines; pasted `/commands` are not
executed as meta-commands. For non-interactive large input, prefer `-p -` or
piped stdin.

At an interactive terminal, the prompt supports basic line editing: left/right
arrows move the cursor, Backspace and Delete remove text around the cursor, and
typed text inserts at the cursor position. Cursor movement is rune-aware but not
full grapheme- or emoji-width aware.

Press **Ctrl-G** at the prompt, or run `/edit [draft]`, to open an external
editor for a multi-line prompt. Harness uses `$VISUAL`, then `$EDITOR`, then
`vi`. The temp file preloads the visible output from the previous turn, followed
by a delimiter; only text written after the delimiter is sent as the next prompt.

| command | effect |
|---|---|
| `/help` | list commands |
| `/exit`, `/quit` | save, print a session token summary, and exit |
| `/clear` | reset the conversation; rotate to a fresh session directory |
| `/compact` | force compaction now |
| `/usage` | cumulative input, cached input, output, reasoning tokens, and cost |
| `/tools` | list enabled built-in and MCP tools with descriptions, plus disabled optional tools |
| `/edit [draft]` | open an external editor for the next prompt |
| `/save [file]` | force save (optionally elsewhere) |
| `/model` | choose a configured provider, then choose one of its configured models |
| `/model <id>` | switch subsequent turns to model `<id>` |
| `/model <provider>:<id>` | switch to `<id>` on a specific configured provider |
| `/mode` | list run modes, marking the current one |
| `/mode <name>` | switch the active run mode |

Anthropic usage does not currently expose a separate reasoning-token field;
extended thinking is counted in output tokens, so the reasoning total remains
zero for Anthropic sessions.

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
| `independent` | all available tools, including `delegate` | complete the task end-to-end without pausing for input; stop only on a hard blocker or the model-turn limit |

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
[turn: 3 model turns · 12.4k in / 1.8k out · $0.071 · ctx 42.0k/256.0k (sys 2.0k tools 1.5k msgs 38.5k) · 4.3s]
```

## Interrupts

- **Ctrl-C during a turn**, or **Esc twice in short succession during a REPL
  turn**, cancels the turn (aborting the HTTP stream and killing any
  `run_command` process group); streamed partial text is kept and un-executed
  tool calls are stripped. Prints `[cancelled]` and returns to the prompt.
- **A second Ctrl-C within ~1s, or Ctrl-C at the idle prompt** saves, prints the
  session token summary, and exits 130. **Ctrl-D** at the prompt saves, prints
  the summary, and exits 0.

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
behavior. In the REPL, `/tools` lists the active mode's enabled built-in and MCP
tools with their descriptions, followed by optional built-ins that are currently
disabled.

Tool results are centrally capped (default 64 KB or 1000 lines; configurable via
`tool_result_max_bytes` / `tool_result_max_lines`). Truncated results include a
marker in the model-visible text, a warning in the UI, and the full output is
archived under the session directory when available.

## MCP servers (optional)

Harness can expose tools from [Model Context Protocol](https://modelcontextprotocol.io)
servers. A second binary, `harness-mcp-proxy`, owns every downstream MCP server
(spawning stdio children, dialing streamable-HTTP endpoints) and aggregates their
tools into one namespaced surface; harness connects to that proxy over HTTP and
registers each tool as an ordinary harness tool. Harness and the proxy speak
MCP streamable HTTP (JSON-RPC 2.0, revision `2025-06-18`), so the proxy is a
single shared daemon that many harness sessions reuse. You start the proxy
yourself; harness never spawns it.

### Enabling it

MCP is **opt-in** and off by default. Turn it on in `~/.config/harness/config.json`:

```json
{
  "mcp": {
    "enable": true,
    "proxy": ""
  }
}
```

or via environment: `HARNESS_MCP_ENABLE=true` and (optionally)
`HARNESS_MCP_PROXY=http://127.0.0.1:8766`. There are no flags. An empty
`proxy` resolves to `http://127.0.0.1:8766`. Precedence is the usual
**env > config file > default**. `proxy` must be an `http(s)://` URL.

### Configuring downstream servers

The proxy has its own config file, **separate from harness**, at
`$XDG_CONFIG_HOME/harness-mcp-proxy/config.json` (else
`~/.config/harness-mcp-proxy/config.json`). It is Claude Code-compatible:

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
  "proxy": {
    "listen": "127.0.0.1:8766",
    "logFile": "",
    "logLevel": "info"
  }
}
```

`proxy.listen` defaults to `127.0.0.1:8766`; set it to another address such as
`127.0.0.1:8420` when you need a different port or host. A server with no `type`
(or `"stdio"`) is a child process (`command`/`args`/`env`); `"http"` is a
streamable-HTTP endpoint (`url`/`headers`). `${NAME}` references in any string are
expanded from the proxy's environment (a literal `$` is preserved; an unset
variable warns and expands to empty). Invalid server entries are skipped with a
warning, never fatal — the proxy still serves the valid ones. See
`examples/harness-mcp-proxy/config.json` for a copyable starting point.

Stdio servers inherit the proxy's **full environment** — whatever environment
the `harness-mcp-proxy serve` process was started with — plus the per-server
`env` overrides. Do not configure untrusted stdio servers when secrets live in the
environment, since the child process can read them.

### Running the proxy

Harness never starts the proxy for you — you run it yourself, once, and leave it
up. The daemon is a single shared process that **outlives harness** and is **shared
across sessions**. A second `serve` on the same address fails with the normal HTTP
bind error, matching `harness-model-proxy`.

```sh
harness-mcp-proxy serve &        # foreground without the & to watch its logs
```

For a persistent setup, run it from your shell profile, a launchd agent (macOS),
or a systemd user unit (Linux) so it comes up at login.

When MCP is enabled, harness connects to the proxy and registers the proxy's
tools under a 5 s timeout, logging a line such as
`mcp: connected (2 servers, 5 tools): fs=3 search=2`. If the connection or
registration fails it emits **one** warning and continues with no MCP tools:

```
mcp: cannot connect to proxy at http://127.0.0.1:8766: <err>; MCP tools unavailable
```

MCP **never fails harness startup**. The startup cost is bounded by the 5 s
registration timeout if the proxy is unreachable or hangs during
`initialize`/`tools/list`.

Default paths (all derived per-user):

- **Proxy URL:** `http://127.0.0.1:8766`.
- **Config:** `$XDG_CONFIG_HOME/harness-mcp-proxy/config.json`, else
  `~/.config/harness-mcp-proxy/config.json`.
- **Log:** stderr unless `proxy.logFile` or `serve -log` is set.

Inspect the live surface without harness with:

```sh
harness-mcp-proxy tools
harness-mcp-proxy tools -proxy http://127.0.0.1:8420
```

### Proxy HTTP details

The proxy serves its merged MCP surface over **streamable HTTP**. Set
`proxy.listen` in the proxy config, or pass `serve -listen`, to change the
default listener:

```json
{ "proxy": { "listen": "127.0.0.1:8420" } }
```

```sh
harness-mcp-proxy serve -listen 127.0.0.1:8420 &
```

The listener speaks **plain HTTP only** — put a reverse proxy (nginx, Caddy) in
front for TLS and any stronger auth. Each session is carried by an `Mcp-Session-Id`
header with a 30-minute idle TTL; responses are `application/json` only, and there
is **no server-push channel**, so a client re-lists on its own rather than being
told of tool changes.

On the harness side, point `mcp.proxy` at the URL and (for an MCP proxy behind
a reverse proxy that wants auth) add a config-file-only `mcp.headers` map sent
on **every** request:

```json
{
  "mcp": {
    "enable": true,
    "proxy": "https://mcp.internal.example/mcp",
    "headers": { "Authorization": "Bearer ${TOKEN}" }
  }
}
```

`headers` has no environment variable — it lives in the config file alongside the
URL it authenticates to. Header values expand `${VAR}` and `${VAR:-default}`;
literal dollar forms such as `$5` and `$$` are preserved. An unset `${VAR}` is a
config error. Because HTTP has no notifications channel, the tool list is
**fixed at startup**: the `[mcp: tool list updated]` notice never fires for the
proxy.

### Tools, modes, and limits

Aggregated tools are named `mcp__<server>__<tool>` (the charset/length must fit
`[a-zA-Z0-9_-]{1,64}`; names that do not are dropped with a warning). They are
plain harness tools, so they flow through the normal truncation, artifact, and
session paths. Modes that inherit the default tool set (`auto`, `independent`, and
config modes without an explicit `allowed_tools`) expose the MCP tools; a mode
with an explicit `allowed_tools` whitelist does **not** (it may list `mcp__` names
manually). The HTTP proxy has no notifications channel, so the tool list is
fixed at startup.

One-shot users should note the startup cost is bounded by the 5 s registration
timeout when the proxy is unavailable or hangs during `initialize`/`tools/list`.
Leave MCP off (the default) for latency-sensitive one-shot invocations that do
not need it.

**v1 non-goals:** tools only (no MCP resources or prompts); streamable-HTTP only
for remote servers (no legacy HTTP+SSE transport); header-based auth only (no
OAuth flow); the HTTP proxy listener is plain HTTP (TLS via a reverse proxy).
