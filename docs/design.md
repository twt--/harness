# harness — design

A minimal agentic coding harness in Go: a plain-text, line-oriented CLI that drives a
tool-using LLM loop against local files, shell commands, and git.

## 1. Goals

- **Small and legible.** The whole system should be readable in an afternoon. One purpose
  per package; no framework.
- **Zero third-party Go dependencies.** Go stdlib only. SSE, diff application, HTML
  stripping, and retries are all small enough to own.
- **Unix philosophy for tools.** When the job is already owned by a mature host CLI
  (`grep`, `rg`, `git`, shell commands), expose a thin argv wrapper instead of
  reimplementing optimized search or command semantics in the harness.
- **Generic over providers.** One internal message/streaming model with two HTTP dialects:
  Anthropic Messages and OpenAI Chat Completions. "OpenAI-style" means the ecosystem
  standard — the same code path must work against OpenAI, vLLM, Ollama, OpenRouter, and
  llama.cpp via a configurable base URL.
- **No sandboxing or permission prompts.** The harness assumes it is launched inside an
  already-sandboxed environment. Tools run with the process's privileges, immediately.
- **First-class git.** A dedicated `git` tool plus git context in the system prompt.

### Non-goals (v1)

- CLI-subprocess backends (`codex`, `claude -p`) — cut from scope; they run their own
  agent loops and are fundamentally different from a model API.
- OpenAI Responses API (future work; Chat Completions is the compatibility standard).
- Markdown rendering, MCP, sub-agents.
- Adopted in v1.1 (no longer a non-goal; see
  `docs/superpowers/specs/2026-06-11-roadmap-items-design.md`): parallel
  dispatch of read-only tool calls.

## 2. Constraints

| Constraint | Choice |
|---|---|
| Language | Go 1.24+ (`iter` / range-over-func used; toolchain 1.26 available) |
| Dependencies | stdlib only |
| Module / binary | `module harness`, binary built from `cmd/harness` |
| Interface | line-oriented plain text; optional dim ANSI color only when stdout is a TTY; `NO_COLOR` and `-no-color` disable |
| Secrets | API keys from environment variables only — never flags or config files |

## 3. Architecture

```
                 ┌────────────────────────────────────────────┐
 stdin ──────►   │ internal/ui        REPL / one-shot driver  │
                 │   meta-commands, rendering, usage line     │
                 └──────────────┬─────────────────────────────┘
                                │ user prompt
                 ┌──────────────▼─────────────────────────────┐
                 │ internal/agent     turn loop               │
                 │   interrupt handling, compaction           │
                 └────┬──────────────────────────┬────────────┘
                      │ Request                  │ ToolCall
        ┌─────────────▼────────────┐   ┌─────────▼────────────┐
        │ internal/llm             │   │ internal/tools       │
        │   Provider interface     │   │   registry+dispatch  │
        │   message model, prices  │   │   built-in tools     │
        ├───────────┬──────────────┤   └──────────────────────┘
        │ llm/openai│ llm/anthropic│
        └───────────┴──────────────┘
              │ HTTP + SSE (internal/sse, internal/retry)
              ▼
        provider endpoint
```

### Package layout

```
cmd/harness/main.go      flags, config load, wiring, signal setup, REPL-vs-oneshot dispatch
internal/llm             provider-agnostic types, Provider interface, model/price registry, factory
internal/llm/openai      Chat Completions dialect: wire structs, request builder, stream decode, tool-call assembly
internal/llm/anthropic   Messages dialect: same responsibilities
internal/sse             generic SSE frame reader
internal/retry           backoff + jitter + Retry-After parsing
internal/agent           turn loop, interrupt state machine, compaction
internal/tools           Tool interface, registry, dispatch (recover + central truncation), built-in tools
internal/session         session state, replay log, compaction archives, tool artifacts
internal/config          flags > env > config-file resolution
internal/modelsdev       optional models.dev catalog reduction for setup/pricing metadata
internal/ui              REPL, streaming renderer, tool summaries, usage line
internal/sysprompt       builtin instructions + environment context (cwd/os/date/git summary)
```

`internal/llm` is the shared contract between the agent loop and the dialects. The loop
imports only `internal/llm`; a small factory package (`internal/llm/factory`,
`factory.New(opts)`) selects the dialect. The factory is its own package — not a file in
`internal/llm` — because it imports both dialect packages, which themselves import
`internal/llm` (an import cycle otherwise).

## 4. Message model (`internal/llm`)

The internal model is Anthropic-shaped — a content-block list — because it is a lossless
superset of OpenAI's flat fields: collapsing blocks into OpenAI's shape is mechanical,
while the reverse direction would lose structure.

```go
type Role string

const (
    RoleUser      Role = "user"
    RoleAssistant Role = "assistant"
    // No tool role: tool results are content blocks on a user message.
    // No system role: the system prompt is a Request field, not a message.
)

type Message struct {
    Role    Role           `json:"role"`
    Content []ContentBlock `json:"content"`
}

type BlockKind string

const (
    BlockText       BlockKind = "text"
    BlockToolUse    BlockKind = "tool_use"
    BlockToolResult BlockKind = "tool_result"
)

// ContentBlock is a tagged union; exactly the fields for Kind are set.
type ContentBlock struct {
    Kind BlockKind `json:"kind"`

    // BlockText
    Text string `json:"text,omitempty"`

    // BlockToolUse (assistant calls a tool)
    ToolUseID string          `json:"tool_use_id,omitempty"` // provider-issued call id
    ToolName  string          `json:"tool_name,omitempty"`
    ToolInput json.RawMessage `json:"tool_input,omitempty"`  // complete JSON object

    // BlockToolResult (we answer a tool call)
    ResultForID string `json:"result_for_id,omitempty"` // matches a ToolUseID
    ResultText  string `json:"result_text,omitempty"`
    ResultError bool   `json:"result_error,omitempty"`
}
```

Design notes:

- **System prompt lives on `Request.System`,** not in the message list. This is the
  natural Anthropic shape, trivially becomes a leading `role:"system"` message for
  OpenAI, and means compaction can never accidentally summarize it away.
- **`ToolInput` is `json.RawMessage`,** not `map[string]any`: it arrives as a byte stream,
  the tool layer decodes it into its own typed struct anyway, and raw bytes round-trip
  through session files without re-encoding surprises.
- **JSON tags are provider-neutral** (`kind`, `tool_use_id`, …). Session files never
  contain raw provider wire JSON, so a session started against Anthropic resumes
  against an OpenAI-compatible server and vice versa.

Two small seam types carry tool traffic between the agent loop and the tool layer;
they are flat views of the corresponding content blocks:

```go
type ToolCall struct { // from a BlockToolUse
    ID    string
    Name  string
    Input json.RawMessage
}

type ToolResult struct { // becomes a BlockToolResult
    ForID   string
    Text    string
    IsError bool
}
```

### Transcript invariant

> Every assistant `tool_use` block has exactly one matching `tool_result` block in the
> following user message, and no `tool_result` is orphaned.

Both APIs hard-reject conversations that violate this. A `ValidateTranscript([]Message) error`
helper encodes the invariant; tests assert it after every operation that mutates a
transcript (cancel, compact, resume, max-steps stop). Repair rules:

- **Cancel mid-turn:** keep streamed partial text as an assistant text-only message;
  strip un-executed `tool_use` blocks. If nothing streamed, drop the partial message.
- **Resume with a dangling `tool_use`** (session saved mid-turn): synthesize a
  `tool_result` with `ResultError: true`, `ResultText: "interrupted"`.

### Wire mapping

| Internal | OpenAI Chat Completions | Anthropic Messages |
|---|---|---|
| `Request.System` | leading `{"role":"system","content":…}` message | top-level `"system"` string |
| user text | `{"role":"user","content":"…"}` | `{"role":"user","content":[{"type":"text",…}]}` |
| assistant text + tool_use | `{"role":"assistant","content":"…","tool_calls":[{"id","type":"function","function":{"name","arguments":<JSON-string>}}]}` | `{"role":"assistant","content":[{"type":"text",…},{"type":"tool_use","id","name","input":<object>}]}` |
| tool_result | separate `{"role":"tool","tool_call_id":…,"content":…}` message per result | `{"type":"tool_result","tool_use_id":…,"content":…,"is_error":…}` block inside a user message |

Mapping subtleties that must be handled:

- OpenAI `function.arguments` is a JSON **string** (`"{\"path\":\"x\"}"`); Anthropic
  `input` is a JSON **object**. A call with no arguments must serialize as `"{}"` for
  OpenAI, never `""`.
- OpenAI tool results are **sibling messages, not blocks**: each `BlockToolResult` is
  hoisted into its own `role:"tool"` message, placed immediately after the assistant
  message that issued the calls, in call order.
- OpenAI has no `is_error` field on tool messages; error results are prefixed
  `ERROR: ` in the content string. Anthropic gets `is_error: true`.
- An assistant message with tool calls but no text serializes with `content` omitted
  (OpenAI) / no text block (Anthropic).

## 5. Provider layer

### 5.1 Interface and stream events

```go
type Provider interface {
    Name() string // "openai" | "anthropic"

    // Stream runs one model call. The iterator yields events until a Done
    // event or a terminal error (yielded at most once, last). Consumer break
    // or ctx cancellation aborts the underlying HTTP request.
    Stream(ctx context.Context, req Request) iter.Seq2[StreamEvent, error]
}

type Request struct {
    Model       string
    System      string
    Messages    []Message
    Tools       []ToolSchema
    MaxTokens   int      // 0 = provider policy (see §5.4)
    Temperature *float64 // nil = omit
    Reasoning   ReasoningConfig
    StopSeqs    []string
}

type ReasoningConfig struct { Effort string } // empty = provider default

type ToolSchema struct {
    Name        string
    Description string
    Parameters  json.RawMessage // JSON Schema object, owned by the tool layer
}
```

`iter.Seq2[StreamEvent, error]` (range-over-func) was chosen over channels: the consumer
is a plain `for ev, err := range stream` with natural early-`break` cancellation, and the
producer keeps stream state on its own stack — no goroutine lifecycle to leak.

```go
type EventKind int

const (
    EventTextDelta     EventKind = iota // incremental assistant text
    EventToolCallStart                  // tool_use began: ID + Name known
    EventToolCallDelta                  // partial JSON args (rendering only)
    EventToolCallDone                   // one call fully assembled
    EventUsage                          // usage snapshot (may arrive >1x)
    EventDone                           // turn end: StopReason + final Usage
)

type StreamEvent struct {
    Kind EventKind

    Text string // EventTextDelta

    // EventToolCall*; Index disambiguates parallel calls within one turn.
    Index     int
    ToolID    string          // Start/Done
    ToolName  string          // Start/Done
    ArgsDelta string          // Delta
    ToolInput json.RawMessage // Done only: complete, valid JSON

    Usage      *Usage     // EventUsage / EventDone
    StopReason StopReason // EventDone
}

type StopReason string

const (
    StopEndTurn   StopReason = "end_turn"
    StopToolUse   StopReason = "tool_use"
    StopMaxTokens StopReason = "max_tokens"
    StopStop      StopReason = "stop" // stop sequence matched
)
```

StopReason normalization: OpenAI `stop|length|tool_calls` and Anthropic
`end_turn|max_tokens|tool_use|stop_sequence` map onto the four constants. Unknown or
provider-specific reasons (e.g. `content_filter`) map to `end_turn` — the turn is over
either way — and are noted on the rendered usage line.

### 5.2 SSE client (`internal/sse`)

A dialect-agnostic frame reader over `io.Reader`:

```go
type Event struct {
    Type string // from "event:" lines; "" for OpenAI (it never sends them)
    Data string // "data:" lines joined with \n
}

func Read(ctx context.Context, r io.Reader) iter.Seq2[Event, error]
```

- `bufio.Scanner` with an enlarged buffer (max token ~1 MB — default 64 KB is too small
  for large tool-argument frames).
- Accumulates `event:`/`data:` lines; yields on blank line; strips one leading space
  after the colon per the SSE spec; ignores comment (`:`) lines.
- Dialect handling stays in the providers:
  - **OpenAI:** every frame is `data:` JSON; the literal `data: [DONE]` terminates.
  - **Anthropic:** typed frames — `message_start`, `content_block_start`,
    `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`,
    `ping` (ignored), `error` (turn-fatal).
- **Truncated stream:** body EOF without `[DONE]` / `message_stop` →
  `ErrTruncatedStream`; the turn fails cleanly. No reconnection — retries never apply
  mid-stream (§5.5).
- Cancellation rides on the HTTP request context: cancelling unblocks the body read and
  the iterator yields `ctx.Err()` as its terminal error.

### 5.3 Streaming tool-call assembly

Providers emit granular `Start`/`Delta` events for live rendering **and** guarantee that
`EventToolCallDone.ToolInput` is complete, valid JSON. The agent loop consumes only
`Done`; the renderer may consume `Delta`. Assembly is per-turn state inside each
provider's `Stream`:

- **OpenAI:** `choices[].delta.tool_calls[]` arrive with an `index`; the first delta for
  an index carries `id` + `function.name` (emit `Start`), subsequent deltas carry
  `function.arguments` string fragments (emit `Delta`). All buffered calls flush as
  `Done` when `finish_reason: "tool_calls"` arrives.
- **Anthropic:** `content_block_start` with `type:"tool_use"` gives `id` + `name` at a
  block index (`Start`); `content_block_delta` with `input_json_delta` carries
  `partial_json` fragments (`Delta`); `content_block_stop` flushes that call (`Done`).

Edge cases:

- **Empty arguments:** OpenAI may send zero fragments; an empty buffer flushes as `{}`.
- **Validation on flush:** `json.Valid` is checked before emitting `Done`; invalid
  accumulated JSON (truncated stream) is a turn-fatal error, never a garbage `Done`.
- **Parallel calls:** both dialects interleave multiple calls; `Index` keeps them
  distinct and emission order is preserved into the transcript.
- **Interleaved text and tool_use** (Anthropic): text blocks share the index space but
  bypass the assembler.

### 5.4 Request building

| Concern | OpenAI Chat Completions | Anthropic Messages |
|---|---|---|
| Endpoint default | `https://api.openai.com/v1/chat/completions` | `https://api.anthropic.com/v1/messages` |
| Auth | `Authorization: Bearer <key>` | `x-api-key: <key>` + `anthropic-version: 2023-06-01` |
| Tool schemas | `tools[].function = {name, description, parameters}` (`type:"function"`) | `tools[] = {name, description, input_schema}` |
| `max_tokens` | sent only if user-set (compatible servers pick their own defaults) | **required** — if unset, default `min(8192, contextWindow/4)` |
| Streaming usage | `"stream_options":{"include_usage":true}` (always set) | automatic: input tokens in `message_start`, output in `message_delta` |
| Stop sequences | `stop` | `stop_sequences` |
| Temperature | omitted when nil (never send a spurious 0) | same |
| Reasoning effort | OpenAI: `reasoning_effort`; OpenRouter: `reasoning.effort` | `output_config.effort` |

The same `ToolSchema.Parameters` bytes go into `parameters` vs `input_schema` —
schemas are never transformed.

**Anthropic prompt caching (v1):** `cache_control: {"type":"ephemeral"}` breakpoints on
the system block and on the last content block of the final message, refreshed every
call. An agentic loop re-sends a growing prefix every step; caching makes that prefix
~10× cheaper. OpenAI caches automatically; no opt-in exists or is needed.

### 5.5 Errors and retries (`internal/retry`)

```go
type APIError struct {
    StatusCode int
    Code       string        // provider error code/type if parseable
    Message    string
    Retryable  bool
    RetryAfter time.Duration // parsed Retry-After, 0 if absent
}
```

- **Retryable:** HTTP 429, 500, 502, 503, 529 (Anthropic overloaded), and transport
  errors (timeouts, resets, DNS).
- **Fatal, no retry:** 400, 401, 403, 404, 422 — surfaced immediately with the
  provider's error message.
- **Backoff:** full jitter — `sleep = rand(0, min(30s, 500ms·2^attempt))`, 5 attempts.
  `Retry-After` (seconds or HTTP-date) is honored as a floor. The policy is a pure
  function (`retry.Next(attempt, retryAfter) time.Duration`); the retry loop lives in
  each provider, which injects a `sleep` function so tests run instantly.
- **Provider retries apply only before the first response byte.** Once tokens have
  streamed, the provider treats failure as terminal — mid-stream Anthropic `error`
  frames and truncated bodies fail the provider call. The agent loop re-requests the
  step from scratch when such a failure is retryable (§8.1; spec
  `docs/superpowers/specs/2026-06-11-roadmap-items-design.md` §2), so a transient
  mid-stream failure no longer ends the turn.
- **Cancellation wins:** `ctx.Err()` is checked before every attempt and every backoff
  sleep, and is distinguished from `APIError` so the UI renders "cancelled" vs "failed".

## 6. Usage, cost, and the model registry

```go
type Usage struct {
    InputTokens      int // uncached input, billed at full rate
    OutputTokens     int
    CacheReadTokens  int
    CacheWriteTokens int
}
```

Normalization: OpenAI's `prompt_tokens` **includes** cached tokens
(`prompt_tokens_details.cached_tokens` is subtracted); Anthropic's `input_tokens`
already excludes them. After normalization `InputTokens` means the same thing on both.

`internal/llm/model.go` holds a small registry:

```go
type Price struct{ Input, Output, CacheRead, CacheWrite float64 } // USD per 1M tokens
type ModelInfo struct {
    ContextWindow int
    Price         Price
    Reasoning     *ReasoningInfo
}
func Cost(model string, u Usage) (usd float64, known bool)
func ContextWindow(model string) int // registry hit, else default 256_000
func Models() []string // sorted configured model ids
```

Unknown models (arbitrary names on OpenAI-compatible servers) display token counts
without a dollar figure, and use a conservative 256k context-window default,
configurable with `-default-context-window` and overridable for a run with
`-context-window`. Model prices and context windows are loaded from provider
config files referenced by the main config, then filled from a best-effort
`https://models.dev/api.json` lookup when local metadata is missing. When
`-reasoning-effort` is set, the same metadata is used to validate known
provider/model reasoning support and effort values. Localhost base URLs skip the
online lookup.

## 7. Configuration and provider selection

Precedence: **flags > environment > config file > built-in defaults.**

- Environment: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_BASE_URL`,
  `ANTHROPIC_BASE_URL`, plus `HARNESS_*` equivalents for most flags. Provider configs
  may also name provider-specific `api_key_env` variables such as
  `OPENROUTER_API_KEY`. Environment API keys override provider-config keys.
- Config file (optional): `~/.config/harness/config.json` — provider, model,
  provider_configs, run modes, flag defaults, and config-only context-efficiency knobs:
  `agents_md_warn_bytes`, `tool_result_max_bytes`, `tool_result_max_lines`,
  `read_file_default_limit`, `compact_keep_turns`, `compact_summary_max_tokens`, and
  `compact_tool_result_max_bytes`. Provider config paths are resolved relative to the
  config file and may define api_type, base_url, api_key, api_key_env, models, context
  windows, reasoning metadata, and pricing.
- `--setup` creates a config in the default directory, or appends a new provider config
  to an existing default config. It fetches models.dev provider metadata, falls back to a
  vendored models.dev snapshot when the live catalog is unreachable, lists
  harness-supported providers, prompts for the API key, pages the selected provider's
  models newest-first, and asks which model should be the default. The provider config
  is generated from models.dev with all known models for that provider: base URL,
  api_type, key env vars, context windows, pricing, and reasoning metadata. Without
  `--force`, setup refuses to overwrite existing provider files and preserves existing
  default provider/model fields; `--force` opts into those overwrites.
- `--refresh-models` fetches the latest live models.dev catalog and regenerates each
  configured provider file, preserving stored API keys. It errors if models.dev is
  inaccessible or a configured provider is missing/unsupported.
- **Selection rule:** `-model` is primary. A `provider:model` value sets the provider and
  strips the prefix before sending requests. Otherwise provider is inferred — model names
  starting with `claude` → Anthropic, everything else → OpenAI-compatible (the right
  fallback for arbitrary local model names). Explicit `-provider` overrides inference and
  may name a provider config whose `api_type` selects the OpenAI or Anthropic dialect. An
  empty API key is allowed when the base URL is non-default (local servers need none).
- A custom base URL supplies scheme/host/prefix only; the dialect appends its standard
  path (`/chat/completions`, `/messages`) — so `-base-url http://localhost:11434/v1`
  works for Ollama.
- `internal/config` resolves the user-facing settings, then hands the provider factory a
  small `factory.Options` struct (provider, model, base URL, API key, max tokens,
  temperature, context window, reasoning mode). This keeps `internal/llm` free of any
  dependency on the flag/env/file machinery.

## 8. Agent loop (`internal/agent`)

### 8.1 Turn loop

One user turn runs model steps until the model stops asking for tools:

```
append user message
for step := 0; step < maxSteps; step++ {           // -max-steps, default 50
    stream := provider.Stream(ctx, request)
    accumulate: print text deltas live; collect assembled tool calls;
                capture usage + stop reason
    append assistant message (text blocks + tool_use blocks, emission order)
    if stopReason != tool_use { break }
    for each tool call, in order:                  // concurrent iff all read-only, else sequential
        result := registry.Dispatch(ctx, call)     // always returns a result
        print one-line tool summary
    append ONE user message carrying all tool_result blocks, in call order
}
print turn usage line; save session
```

- **Mostly-sequential tool execution.** Coding tools mutate a shared filesystem; deterministic
  ordering matching the model's emission order is worth far more than parallelism. A step
  whose calls are all read-only (2+ of them) dispatches concurrently, bounded at 8; results,
  sink events, and transcript blocks stay in emission order. Mixed steps stay sequential.
- **One result per call, always.** Required by both APIs (§4 invariant). `Dispatch`
  produces a result even on panic.
- **Max-steps guard:** on hit, print
  `[stopped: reached max steps (50); say "continue" to keep going]`, keep the
  transcript (it is valid — the last step's results are appended), return to the prompt.

### 8.2 Tool failure handling

`Dispatch` never lets the loop crash. Each failure mode becomes an `is_error` result
string fed back to the model so it can self-correct:

| Failure | Result text |
|---|---|
| unknown tool name | `error: unknown tool "<name>"` |
| invalid JSON args | `error: invalid arguments: <detail>` |
| tool returned error | `error: <message>` |
| tool panicked | `error: tool panicked: <recovered>` (also logged to stderr) |

### 8.3 Output truncation

A central cap in `Dispatch` (backstop for every tool): **64 KB or 1000 lines per
result** by default, configurable with `tool_result_max_bytes` and
`tool_result_max_lines`. The first cap hit adds a teaching marker:

```
[truncated: showing first 1000 of 4213 lines; use read_file offset/limit or grep to narrow]
```

Individual tools may also apply their own natural limits, but the central cap is the
backstop for every result. Truncated results carry metadata so the UI can warn and write
the full output to the session's `artifacts/tool-results/` directory.

### 8.4 Interrupts

A single SIGINT handler plus a per-turn `context.CancelFunc`:

- **^C during a turn** → cancel the turn context (aborts the HTTP stream; kills
  `run_command` process groups). Apply the cancel repair rule (§4): keep streamed
  partial text, strip un-executed tool calls. Print `[cancelled]`, return to prompt.
- **Second ^C within ~1 s, or ^C at the idle prompt** → save session, exit 130.
- **^D at the prompt** → save session, exit 0.

### 8.5 System prompt (`internal/sysprompt`)

`system = builtinInstructions + "\n\n" + envContext`

- **Builtin instructions** (a constant): concise agentic-coding guidance — read before
  editing, prefer `edit` with unique context, use tools rather than guessing file
  contents, use `rg` when available or `grep`/`list_dir` for search, run builds/tests
  via `run_command`, stop when done.
- **Environment context**, computed at startup:

  ```
  <env>
  cwd: /Users/twt/project
  os: darwin/arm64
  date: 2026-06-09
  git: branch=main, 2 modified, 1 untracked
  </env>
  ```

  Git summary via `git branch --show-current` + parsed `git status --porcelain`;
  `git: (not a git repository)` otherwise.
- Flags: `-system <text|@file>` **appends** (the common case — project notes);
  `-system-override <text|@file>` replaces the builtin; `-no-env` drops the env block.

## 9. Tool set (`internal/tools`)

```go
type Tool interface {
    Name() string
    Description() string     // model-facing, one line
    Schema() json.RawMessage // JSON Schema for the input object
    ReadOnly() bool
    Run(ctx context.Context, input json.RawMessage) (string, error)
}

type Registry struct{ /* ordered map */ }
func (r *Registry) Register(t Tool)
func (r *Registry) Specs() []llm.ToolSchema
func (r *Registry) Dispatch(ctx context.Context, call llm.ToolCall) llm.ToolResult
```

- **Schemas are hand-written JSON Schema constants.** The schema is the model-facing
  contract; descriptions, enums, and required-ness deserve hand tuning, and reflection
  fights you on exactly those fields.
- **Tools self-validate args** after `json.Unmarshal` into a private struct (no stdlib
  JSON Schema validator; unknown extra keys are tolerated — models hallucinate them).
- Relative paths resolve against the process cwd. No path restrictions — the harness is
  honest about its no-sandbox assumption.

### 9.1 `read_file`

> Read a file from disk. Returns line-numbered content; supports offset/limit for large files.

| param | type | notes |
|---|---|---|
| `path` | string, required | file path |
| `offset` | int | 1-based starting line |
| `limit` | int | max lines, default 1000 or `read_file_default_limit` |

- Output is line-numbered (`cat -n` style: right-aligned number, tab, line). Line
  numbers make `edit` targeting and grep cross-referencing far more reliable.
- Binary sniff: first 8 KB containing NUL → `error: <path> appears to be binary`.
- Files >10 MB read only the first window unless offset/limit given.
- Directory → `error: <path> is a directory; use list_dir`. Offset past EOF → error
  stating the file's line count. Empty file → `(empty file)`.

### 9.2 `list_dir`

> List directory entries with type and size. Non-recursive; pass a glob to filter.

| param | type | notes |
|---|---|---|
| `path` | string | default `"."` |
| `glob` | string | `path.Match` filter on base names |

- Non-recursive by design — recursion belongs to `grep`/`rg`/host commands, and
  `run_command` (`find`) is the escape hatch. No separate `find` tool: fewer tools
  means better model reliability.
- One entry per line: type char, human-readable size, name (`/` suffix for dirs);
  dirs-first, then alphabetical. 1000-entry cap with truncation marker.
- Unreadable entries shown with `?` size; listing continues.

### 9.3 `grep` and optional `rg`

> `grep`: Run the host grep command directly. Pass grep options and operands as args, e.g. ["-R","-n","TODO","."]. No shell; returns combined stdout+stderr and the exit code.

> `rg`: Run the host rg (ripgrep) command directly. Pass ripgrep options and operands as args, e.g. ["-n","TODO","."]. No shell; returns combined stdout+stderr and the exit code.

| param | type | notes |
|---|---|---|
| `args` | array of strings, required | arguments passed after the program name |
| `stdin` | string | written to the program's standard input |
| `cwd` | string | default process cwd |
| `timeout_seconds` | int | default 120, cap 600 |

- `grep` is always registered and invokes `grep` from the harness process PATH.
- `rg` is registered immediately after `grep` only when `exec.LookPath("rg")` succeeds
  at registry construction time. If `rg` is not installed, the model never sees that
  tool name.
- Missing optional CLI-backed tools are reported once at startup through the plaintext
  slog handler, e.g.
  `[warn] [cli_tools] Tool "rg" is disabled. Reason: "rg" binary not found.`
  `-q`/`--quiet` suppresses these diagnostics, and `--log-level`/`LOG_LEVEL` filters
  them by level.
- Both tools use `exec.Command(program, args...)`: no shell, glob expansion, pipes,
  redirection, `$VAR`, or `~` expansion. Each argument arrives byte-for-byte.
- Search semantics are the host tool's semantics. Regex syntax, recursion,
  gitignore/default ignore behavior, binary handling, hidden files, and output shape are
  selected with native CLI flags (`grep -R -n`, `grep -F`, `rg -n`, `rg --hidden`,
  `rg --no-ignore`, etc.), not reimplemented by the harness.
- Same process conventions as `run_command` (§9.7): own process group, timeout or ^C
  kills the group, combined stdout+stderr, `[exit code: N]` trailer, and non-zero exit
  is NOT an error result. For search this matters because no matches is commonly exit
  code 1.

### 9.4 `edit`

> Replace an exact string in a file. old_string must appear exactly once unless replace_all is set.

| param | type | notes |
|---|---|---|
| `path` | string, required | must exist (use write_file to create) |
| `old_string` | string, required | exact byte match, whitespace included |
| `new_string` | string, required | |
| `replace_all` | bool | default false |

- 0 occurrences → `error: old_string not found in <path>`.
- N>1 without `replace_all` → `error: old_string appears N times; add surrounding
  context to make it unique, or set replace_all`.
- Empty `old_string` or `old_string == new_string` → error.
- Success reports `edited <path> (N replacement(s))`.

### 9.5 `write_file`

> Create or overwrite a file with the given content. Creates parent directories.

| param | type | notes |
|---|---|---|
| `path` | string, required | |
| `content` | string, required | empty allowed |

- `os.MkdirAll` parents (0755), write 0644, overwrite without ceremony (no permission
  system by design). Reports `created`/`overwrote`, bytes, lines.
- Existing directory at path, or trailing `/` → error.

### 9.6 `apply_patch`

> Apply a unified-diff patch. May span multiple files; supports create, delete, and rename.

| param | type | notes |
|---|---|---|
| `patch` | string, required | full unified diff text |

- Hand-written parser: `--- a/x` / `+++ b/y` headers, `@@ -l,s +l,s @@` hunks,
  `--- /dev/null` create, `+++ /dev/null` delete, git extended `rename from/to`.
- **Per-hunk matching, three levels tried in order:**
  1. exact match at the header's stated position;
  2. offset search — same lines found within ±200 lines (later hunks shift accordingly);
  3. whitespace-normalized comparison (leading/trailing whitespace stripped), applied
     while preserving the file's actual lines.
- **Atomic per file, best-effort across files:** all of a file's hunks apply to an
  in-memory copy; any hunk failure leaves that file untouched and rejected. Other files
  still apply. The result lists `applied: …` and `rejected: <file> (hunk i of n did not
  match)` so the model can retry just the failures. Per-file atomicity prevents
  half-edited files; cross-file best-effort keeps the model's correct work.
- Create-where-exists, delete-content-mismatch, missing target → that file rejected.

### 9.7 `run_command`

> Run a shell command. Returns combined stdout+stderr and the exit code. For arguments containing quotes, spaces, or newlines, prefer exec to avoid shell-quoting issues.

| param | type | notes |
|---|---|---|
| `command` | string, required | |
| `stdin` | string | written to the command's standard input |
| `cwd` | string | default process cwd |
| `timeout_seconds` | int | default 120, cap 600 |

- Executed via `bash -lc` (fallback `sh -c` if bash is absent); `-l` picks up the user's
  PATH/toolchain for build and test commands.
- **Combined stdout+stderr** — the model reads a terminal transcript the way a human
  does; interleaving beats separation.
- `[exit code: N]` always appended. **Non-zero exit is NOT an error result** — a failing
  build is exactly the signal the model needs; only infrastructure failures (shell
  couldn't start) set `is_error`.
- Runs in its own process group under the turn context; timeout or ^C kills the group
  (children included) and reports output captured so far.
- Environment inherited unmodified.
- `stdin`, when provided, is written verbatim to the command's standard input; absent
  means `/dev/null` (programs see immediate EOF, never hang on input). Prefer it over
  `echo`/heredocs when feeding content to a command (`git commit -F -`, `python -`,
  `tee file`) — content travels with zero shell escaping.

### 9.8 `exec`

> Run a program directly with an argv array (no shell). Use when arguments contain quotes, spaces, or newlines; no globbing/pipes/$VAR — use run_command for those. Returns combined stdout+stderr and the exit code.

| param | type | notes |
|---|---|---|
| `argv` | array of strings, required | program + literal arguments |
| `stdin` | string | written to the program's standard input |
| `cwd` | string | default process cwd |
| `timeout_seconds` | int | default 120, cap 600 |

- `exec.Command(argv[0], argv[1:]...)` — no shell anywhere, so arguments arrive
  byte-for-byte: nothing to quote, nothing to escape, nothing to inject.
- **Why it exists:** shell quoting is the dominant model failure when generated content
  (commit messages with apostrophes, `python -c` one-liners, sed programs, JSON) travels
  through `run_command` as part of a command line. The argv form eliminates that failure
  class; `git` (§9.9) proved the pattern works with models.
- No globbing, `$VAR` expansion, `~`, pipes, or redirection — the tool descriptions
  cross-steer: `exec` for tricky arguments, `run_command` for shell features. argv[0]
  resolves against the harness process PATH, not the login-shell PATH `bash -lc` sees.
- Same conventions as `run_command` (§9.7), sharing its implementation: own process
  group, timeout or ^C kills the group, combined stdout+stderr, `[exit code: N]`
  trailer, non-zero exit is NOT an error result. A missing binary is a normal tool
  error naming the program so the model can correct the call.

### 9.9 `git`

> Run a git command. Pass arguments as an array, e.g. ["status","--porcelain"]. No shell; no pager.

| param | type | notes |
|---|---|---|
| `args` | array of strings, required | argv after `git` |

- `git` is registered only when `exec.LookPath("git")` succeeds at registry
  construction time. If git is not installed, the model never sees the `git` tool name.
- `exec.CommandContext(ctx, <resolved-git-path>, append([]string{"--no-pager"}, args...)...)`
  — no shell, so no quoting ambiguity. `GIT_TERMINAL_PROMPT=0` prevents auth hangs.
- **One argv tool, not narrow per-subcommand tools:** a single stable schema covers the
  entire git surface (status, diff, log, blame, stash, rebase, commit) that the model
  already knows from training; enumerating subcommands multiplies schemas and still
  misses the long tail.
- Combined output + exit code, same conventions as `run_command`. Interactive flows
  (`rebase -i`) fail fast (no TTY) rather than hang.

### 9.10 `web_fetch`

> Fetch a URL (http/https) and return its text content. HTML is reduced to readable text.

| param | type | notes |
|---|---|---|
| `url` | string, required | http/https only |
| `max_bytes` | int | default 1 MB, cap 5 MB |

- 30 s timeout; up to 5 redirects, each hop re-validated as http/https.
- `text/html` → hand-rolled reduction: drop `<script>`/`<style>` blocks, strip tags,
  `html.UnescapeString` (stdlib), collapse whitespace. Explicitly "readable-ish text",
  not a renderer — good enough for docs and articles. Other `text/*`,
  `application/json`, `application/xml` → raw. Binary content types → error.
- Output prefixed `# <final-url> (<status>, <content-type>)`. Non-2xx responses return
  status + body as content (not `is_error` — the model may want the error page).

### 9.11 `git_readonly`

> Run a read-only git command: status, log, diff, show, grep, blame, or bisect.

| param | type | notes |
|---|---|---|
| `args` | array of strings, required | argv after `git`, starting with the subcommand |

- A read-only sibling of `git` (§9.9) used by restricted run modes (§14). It is
  registered only when git is installed and reuses the same `--no-pager` /
  `GIT_TERMINAL_PROMPT=0` plumbing.
- **Allowlist by bare subcommand:** `args[0]` must be one of `status`, `log`, `diff`,
  `show`, `grep`, `blame`, `bisect` and must not start with `-`. Because global git
  options (`-c`, `-C`, `--git-dir`, `--exec-path`, `--paginate`) precede the
  subcommand, requiring a non-flag first argument blocks every global-option
  injection. Subcommand-local flags after `args[0]` pass through.
- A few local flags still break the read-only boundary and are rejected:
  `--output`/`--output-directory` (write a file) and `-O`/`--open-files-in-pager`
  (launch a pager/editor). `bisect run <cmd>` is rejected (it executes commands);
  other `bisect` operations are allowed even though they move HEAD.

### 9.12 `write_tmp_file`

> Write a scratch file under this run's private temp directory and return its absolute path.

| param | type | notes |
|---|---|---|
| `name` | string, required | relative file name (subdirectories allowed) |
| `content` | string, required | full file content (empty allowed) |

- Gives read-only run modes (§14, `plan`) a place to draft notes without project
  write access. Files are written under one `os.MkdirTemp` directory created lazily on
  first use and shared across calls; they are kept after exit.
- `name` must be relative and stay inside the temp directory: absolute paths and any
  `..` escape (after `filepath.Clean`) are rejected. Returns the absolute path written.

## 10. CLI / REPL (`internal/ui`)

### Rendering

- Assistant text streams raw as deltas arrive. No markdown rendering.
- Tool calls render as one-liners:
  `[grep] args=["-R","-n","func main","."] → 14 lines, 2.1KB`
  built from the tool name, key args, and a result summary. `-v` adds the first ~5 lines
  of each result, dimmed.
- Per-turn usage line: `[turn: 3 steps · 12.4k in / 1.8k out · $0.071 · 4.3s]`
  (cost omitted for unknown models).
- Dim color only when stdout is a TTY (`os.Stdout.Stat()` mode check); `NO_COLOR` env or
  `-no-color` disables. Everything is legible without color.
- Startup diagnostics use `log/slog` with a plaintext handler: `[level] [category]
  message`. Default level is `info`; `--log-level` or `LOG_LEVEL` accepts `debug`,
  `info`, `warn`, or `error`; `-q`/`--quiet` suppresses non-error slog-backed
  diagnostics.

### Terminal reset on REPL start

Before the first prompt the REPL restores the controlling terminal to a usable state
(`internal/term`, stdlib-only): kernel termios to the platform's `stty sane` equivalent
(GNU semantics on Linux; BSD `f_sane` flag semantics plus the `cfmakesane` control-char
reset on macOS), then an emulator soft reset (DECSTR; mouse tracking, focus reporting,
and bracketed paste off; leave alt screen; show cursor; charset and SGR reset). This
repairs a terminal left in raw/no-echo/mouse-reporting state by a crashed program. It
targets `/dev/tty` directly, is a silent no-op without a controlling terminal, and —
unlike the RIS (`\033c`) it replaced — never clears the screen or scrollback.

### Meta-commands

Lines starting with `/` are commands; `//` escapes a literal slash.

| command | effect |
|---|---|
| `/help` | list commands |
| `/exit`, `/quit` | save and exit |
| `/clear` | reset conversation; rotate to a fresh session file |
| `/compact` | force compaction now |
| `/usage` | cumulative session tokens + cost |
| `/save [file]` | force save (optionally elsewhere) |
| `/model` | show current provider/model/base-url and configured models |
| `/model <id>` | switch subsequent turns to model `<id>` |
| `/model <provider>:<id>` | switch to `<id>` on a specific configured provider |

### Flags

```
-p <prompt|->     one-shot mode; "-" or piped stdin reads the prompt from stdin
-provider <name>  openai | anthropic (default: inferred from -model)
-model <id>
-base-url <url>
-system <text|@file>           append to system prompt
-system-override <text|@file>  replace builtin instructions
-no-env           omit environment context block
-resume <file>    load a session transcript and continue
-session <file>   explicit session save path
-max-steps <n>    model round-trips per user turn (default 50)
-default-context-window <n>
-context-window <n>
-reasoning-effort <level>
-v                show tool result snippets
-q, --quiet       suppress informational diagnostics
--log-level <level>  diagnostic log level: debug, info, warn, error (also LOG_LEVEL)
-no-color
-config <file>    alternate config path
--setup           create or update config in ~/.config/harness
--force           with --setup, overwrite existing provider files and defaults
--refresh-models  fetch models.dev and update configured provider model metadata
```

### One-shot mode (`-p`)

- Prompt from the flag value; `-p -` or piped stdin reads stdin (both → flag text, then
  stdin — enables `harness -p "summarize:" < notes.txt`).
- **Assistant text → stdout; tool summaries, usage, errors → stderr.** So
  `harness -p "…" > answer.txt` captures exactly the model's answer.
- Exit codes: `0` completed, `1` runtime error, `2` usage error, `130` interrupted.
- Runs exactly one user turn, saves the session, exits.

## 11. Session persistence (`internal/session`)

```go
type Session struct {
    Version  int           `json:"version"` // 2
    Provider string        `json:"provider"`
    Model    string        `json:"model"`
    Created  time.Time     `json:"created"`
    Updated  time.Time     `json:"updated"`
    System   string        `json:"system"`
    Mode     string        `json:"mode,omitempty"`
    Turn     int           `json:"turn,omitempty"`
    Messages []llm.Message `json:"messages"`
    Usage    UsageTotals   `json:"usage"`
}

type UsageTotals struct {
    llm.Usage         // cumulative token counts
    CostUSD   float64 `json:"cost_usd"` // 0 when the model has no price entry
}
```

- A session path is a directory. `state.json` is the compact resumable state,
  `raw.ndjson` is append-only replay data, `compactions/` stores raw messages removed
  from active context, and `artifacts/tool-results/` stores full truncated tool output.
- **Saved after every turn**, atomically (write `state.json.tmp`, `os.Rename`). Cheap
  relative to a model call; crash-safe for long sessions.
- Auto-save to `~/.local/state/harness/sessions/<timestamp>`; the path is printed at
  startup. `-session` chooses a directory; `-resume` loads `state.json` (applying the
  dangling-tool-use repair, §4). `/clear` rotates to a fresh directory.
- `harness session replay <session-dir>` prints `raw.ndjson` as the familiar
  user-facing terminal view.
- Transcripts are provider-neutral; resuming under a different provider/model works.
  When flags disagree with the state, flags win with a warning.

## 12. Compaction (`internal/agent/compact.go`)

- **Trigger:** after a turn whose reported input tokens or estimated full-request
  footprint reaches **78%** of the model's context window (headroom for the summary
  call plus the next turn). Also manual `/compact`.
- **Mechanism:** keep the system prompt and the configured number of recent turns
  verbatim (`compact_keep_turns`, default 4; a turn = a user message through the
  following end-turn). Send everything older to the model with a summarization
  instruction: preserve the task/goal, decisions made, files created/modified and their
  current state, key facts learned, open TODOs; do not invent. Summary output is capped
  by `compact_summary_max_tokens` (default 2048). Replace the old messages with a
  single user message:
  `=== Summary of earlier conversation ===\n<summary>`.
- Before summarization, large old tool results are reduced to previews
  (`compact_tool_result_max_bytes`, default 4096). If older history is too large for
  one summary request, it is summarized in chunks, then the chunk summaries are
  summarized.
- Before replacing active history, raw removed messages are archived under
  `compactions/`; the active summary includes the archive reference.
- The summary call's tokens and cost are added to session totals and reported:
  `[compacted: 38 messages → summary · 9.1k in / 0.4k out · $0.05]`.
- **Degradation:** if still over budget, keep only the last turn; if still over,
  hard-truncate the largest tool results in place with markers. Never wedge.
- **Failure:** if the summary or archive step errors, abort compaction, warn, and keep
  the full transcript — the next call may fail visibly on context length, which beats
  silent data loss.
- Compacted transcripts must still satisfy the §4 invariant (kept turns are whole turns,
  so no tool_use/tool_result pair is ever split).

## 13. Testing strategy

Seams that make the system testable: the `Provider` interface (scripted `FakeProvider`),
the `Tool` interface + registry, REPL via injected `io.Reader`/`io.Writer` (TTY detection
injectable), the retry clock, and `ValidateTranscript`.

| Layer | Tests |
|---|---|
| `internal/sse` | frame parsing tables; huge frames; truncated input |
| providers | `httptest.Server` replaying `.sse` golden fixtures per dialect → assert ordered events; golden request-JSON tests (role:tool hoisting, args-string vs object, system placement, `stream_options`, cache_control); tool-call reassembly tables (fragment splits, empty args → `{}`, interleaved parallel calls, invalid tail → turn-fatal); truncated stream; mid-stream cancellation; retry loop via injected sleeper (429-then-200, 400 immediate failure, budget exhaustion) |
| `internal/retry` | `Next`: jitter bounds, 30s cap, Retry-After floor |
| tools | table-driven against `t.TempDir()`; `grep` wrapper against the host CLI; optional `rg` registration with a fake executable on PATH; `git` against a scratch `git init` repo (skipped if git absent); `run_command` timeout via `sleep`; `apply_patch` table: exact/offset/whitespace fuzz, create, delete, rename, multi-file with one rejected file (rejected file untouched) |
| agent loop | `FakeProvider` scripts: multi-tool batches, error-result feedback (next request carries the error), max-steps stop, cancellation → transcript still re-sendable |
| session | save→load→save round-trip; atomic rename leaves no `.tmp`; resume repair; cross-provider resume |
| compaction | canned summary via FakeProvider; old messages collapse, last 4 turns kept; invariant holds |
| ui | scripted REPL input (`/help`, prompt, `/exit`); rendering goldens with fake clock/usage |

Cross-cutting: `ValidateTranscript` is asserted after every transcript mutation in every
test that touches one.

## 14. Run modes (`internal/mode`)

A **run mode** is a named bundle of an allowed-tool set and extra system-prompt
instructions. It lets one harness behave as a collaborative planner, an autonomous
worker, or the wide-open default without separate binaries.

- **Selection** follows the standard precedence (§7): `-mode` flag > `HARNESS_MODE`
  > `mode` in the config file > the built-in default `auto`. An empty value means
  "unspecified", so a resumed session's saved mode (§11) can supply it before the
  `auto` fallback. `/mode <name>` switches at runtime; `/mode` lists.
- **Built-ins:** `auto` (all available tools, no extra prompt — current behavior),
  `plan` (read-only tools including optional `rg` when installed, plus optional
  `git_readonly` when git is installed and `write_tmp_file`, a planning prompt), and
  `independent` (all available tools, a
  complete-without-asking prompt).
- **Config `modes`** entries **field-level merge** onto a built-in of the same name:
  a non-empty `allowed_tools` or `prompt` replaces, an omitted field inherits. A new
  name defines a new mode (no `allowed_tools` ⇒ the full default set). Mode prompts
  accept `@file` and are expanded once at startup (fail-fast).
- **Tool gating** is the harness's one departure from the no-sandbox stance (§2): the
  mode's tool set is realized by `tools.Registry.Subset`, building a registry that
  holds only the allowed tools. Because the agent advertises (`Specs`) and dispatches
  from the same registry, an excluded tool is neither offered nor callable. The
  underlying tools still assume an external sandbox for real isolation; gating only
  shapes what each mode exposes. `Agent.SetTools` swaps the registry for `/mode`.
- The mode prompt is appended to the system prompt as the final section, so it layers
  on top of the builtin instructions, env block, AGENTS.md, and `-system` text. The
  active mode is saved with the session and restored on `-resume` (flags win).

## 15. Future work

- CLI-subprocess backends (codex / claude) behind a separate "delegate" abstraction.
- OpenAI Responses API dialect.
- Markdown rendering.
- MCP client support; sub-agent spawning.
- Smarter prompt-cache breakpoint placement (the fourth allowed breakpoint is still
  unused; dynamic placement could help compaction-heavy sessions).
