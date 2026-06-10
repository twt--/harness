# harness — design

A minimal agentic coding harness in Go: a plain-text, line-oriented CLI that drives a
tool-using LLM loop against local files, shell commands, and git.

## 1. Goals

- **Small and legible.** The whole system should be readable in an afternoon. One purpose
  per package; no framework.
- **Zero third-party dependencies.** Go stdlib only. SSE, diff application, HTML
  stripping, and retries are all small enough to own.
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
- `.gitignore`-aware search, markdown rendering, parallel tool execution, MCP, sub-agents.

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
        │   message model, prices  │   │   9 tools            │
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
internal/tools           Tool interface, registry, dispatch (recover + central truncation), the 9 tools
internal/session         transcript persistence (atomic save/load)
internal/config          flags > env > config-file resolution
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
    StopSeqs    []string
}

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
- **Retries apply only before the first response byte.** Once tokens have streamed,
  failure is turn-fatal — retrying mid-stream would duplicate partial output and
  double-charge. Mid-stream Anthropic `error` frames and truncated bodies fail the turn.
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
}
func Cost(model string, u Usage) (usd float64, known bool)
func ContextWindow(model string) int // registry hit, else default 128_000
```

Unknown models (arbitrary names on OpenAI-compatible servers) display token counts
without a dollar figure, and use a conservative 128k context-window default,
overridable with `-context-window`. The price table is maintained by hand and is
explicitly best-effort.

## 7. Configuration and provider selection

Precedence: **flags > environment > config file > built-in defaults.**

- Environment: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_BASE_URL`,
  `ANTHROPIC_BASE_URL`, plus `HARNESS_*` equivalents for most flags.
- Config file (optional): `~/.config/harness/config.json` — provider, model, base_url,
  and flag defaults. **API keys are env-only** (never config or flags — they leak into
  shell history and committed dotfiles).
- **Selection rule:** `-model` is primary. Provider is inferred — model names starting
  with `claude` → Anthropic, everything else → OpenAI-compatible (the right fallback for
  arbitrary local model names). Explicit `-provider` overrides inference. An empty API
  key is allowed when the base URL is non-default (local servers need none).
- A custom base URL supplies scheme/host/prefix only; the dialect appends its standard
  path (`/chat/completions`, `/messages`) — so `-base-url http://localhost:11434/v1`
  works for Ollama.
- `internal/config` resolves the user-facing settings, then hands the provider factory a
  small `factory.Options` struct (provider, model, base URL, API key, max tokens,
  temperature, context window). This keeps `internal/llm` free of any dependency on the
  flag/env/file machinery.

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
    for each tool call, in order:                  // sequential, never parallel
        result := registry.Dispatch(ctx, call)     // always returns a result
        print one-line tool summary
    append ONE user message carrying all tool_result blocks, in call order
}
print turn usage line; save session
```

- **Sequential tool execution.** Coding tools mutate a shared filesystem; deterministic
  ordering matching the model's emission order is worth far more than parallelism.
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
result**, whichever hits first, with a teaching marker:

```
[truncated: showing first 1000 of 4213 lines; use read_file offset/limit or grep to narrow]
```

Individual tools also pre-cap with tool-specific advice (e.g. `grep` match caps).

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
  contents, run builds/tests via `run_command`, stop when done.
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
| `limit` | int | max lines, default 1000 |

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

- Non-recursive by design — recursion is `grep`'s job, and `run_command` (`find`) is the
  escape hatch. No separate `find` tool: fewer tools means better model reliability.
- One entry per line: type char, human-readable size, name (`/` suffix for dirs);
  dirs-first, then alphabetical. 1000-entry cap with truncation marker.
- Unreadable entries shown with `?` size; listing continues.

### 9.3 `grep`

> Search file contents with a Go (RE2) regular expression. Recurses from a path; prints path:line:text.

| param | type | notes |
|---|---|---|
| `pattern` | string, required | Go `regexp` syntax |
| `path` | string | default `"."`; may be a single file |
| `glob` | string | base-name filter |
| `ignore_case` | bool | prepends `(?i)` |
| `max_matches` | int | default 200 |

- Walks with `filepath.WalkDir`, skipping a **fixed denylist**: `.git`, `node_modules`,
  `vendor`, `dist`, `build`, `target`, `.venv`, `__pycache__`, `.svn`, `.hg`. No
  `.gitignore` parser — a correct one (nesting, negation, precedence) is a real
  subsystem; the denylist is predictable and stdlib-trivial. Documented behavior.
- Skips binary files (NUL sniff) and files >5 MB.
- Output `relpath:lineno:text`, line text capped at 300 chars; `[truncated at N
  matches]` when capped; `(no matches)` on zero.
- Invalid pattern → `error: invalid pattern: <detail>` (model fixes and retries).

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

> Run a shell command. Returns combined stdout+stderr and the exit code.

| param | type | notes |
|---|---|---|
| `command` | string, required | |
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

### 9.8 `git`

> Run a git command. Pass arguments as an array, e.g. ["status","--porcelain"]. No shell; no pager.

| param | type | notes |
|---|---|---|
| `args` | array of strings, required | argv after `git` |

- `exec.CommandContext(ctx, "git", append([]string{"--no-pager"}, args...)...)` — no
  shell, so no quoting ambiguity. `GIT_TERMINAL_PROMPT=0` prevents auth hangs.
- **One argv tool, not narrow per-subcommand tools:** a single stable schema covers the
  entire git surface (status, diff, log, blame, stash, rebase, commit) that the model
  already knows from training; enumerating subcommands multiplies schemas and still
  misses the long tail.
- Combined output + exit code, same conventions as `run_command`. Interactive flows
  (`rebase -i`) fail fast (no TTY) rather than hang.

### 9.9 `web_fetch`

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

## 10. CLI / REPL (`internal/ui`)

### Rendering

- Assistant text streams raw as deltas arrive. No markdown rendering.
- Tool calls render as one-liners:
  `[grep] pattern="func main" path=. → 14 matches, 2.1KB`
  built from the tool name, key args, and a result summary. `-v` adds the first ~5 lines
  of each result, dimmed.
- Per-turn usage line: `[turn: 3 steps · 12.4k in / 1.8k out · $0.071 · 4.3s]`
  (cost omitted for unknown models).
- Dim color only when stdout is a TTY (`os.Stdout.Stat()` mode check); `NO_COLOR` env or
  `-no-color` disables. Everything is legible without color.

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
| `/model` | print provider/model/base-url |

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
-context-window <n>
-v                show tool result snippets
-no-color
-config <file>    alternate config path
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
    Version  int           `json:"version"` // 1
    Provider string        `json:"provider"`
    Model    string        `json:"model"`
    Created  time.Time     `json:"created"`
    Updated  time.Time     `json:"updated"`
    System   string        `json:"system"`
    Messages []llm.Message `json:"messages"`
    Usage    UsageTotals   `json:"usage"`
}

type UsageTotals struct {
    llm.Usage         // cumulative token counts
    CostUSD   float64 `json:"cost_usd"` // 0 when the model has no price entry
}
```

- **Saved after every turn**, atomically (write `<path>.tmp`, `os.Rename`). Cheap
  relative to a model call; crash-safe for long sessions.
- Auto-save to `~/.local/state/harness/sessions/<timestamp>.json`; the path is printed
  at startup. `-session` chooses the path; `-resume` loads any transcript (applying the
  dangling-tool-use repair, §4). `/clear` rotates to a fresh file.
- Transcripts are provider-neutral; resuming under a different provider/model works.
  When flags disagree with the file, flags win with a warning.

## 12. Compaction (`internal/agent/compact.go`)

- **Trigger:** after a turn whose reported input tokens ≥ **78%** of the model's context
  window (headroom for the summary call plus the next turn). Also manual `/compact`.
- **Mechanism:** keep the system prompt and the **last 4 turns verbatim** (a turn = a
  user message through the following end-turn). Send everything older to the model with
  a summarization instruction: preserve the task/goal, decisions made, files
  created/modified and their current state, key facts learned, open TODOs; do not
  invent. Replace the old messages with a single user message:
  `=== Summary of earlier conversation ===\n<summary>`.
- The summary call's tokens and cost are added to session totals and reported:
  `[compacted: 38 messages → summary · 9.1k in / 0.4k out · $0.05]`.
- **Degradation:** if still over budget, keep only the last turn; if still over,
  hard-truncate the largest tool results in place with markers. Never wedge.
- **Failure:** if the summary call errors, abort compaction, warn, and keep the full
  transcript — the next call may fail visibly on context length, which beats silent
  data loss.
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
| tools | table-driven against `t.TempDir()`; `git` against a scratch `git init` repo (skipped if git absent); `run_command` timeout via `sleep`; `apply_patch` table: exact/offset/whitespace fuzz, create, delete, rename, multi-file with one rejected file (rejected file untouched) |
| agent loop | `FakeProvider` scripts: multi-tool batches, error-result feedback (next request carries the error), max-steps stop, cancellation → transcript still re-sendable |
| session | save→load→save round-trip; atomic rename leaves no `.tmp`; resume repair; cross-provider resume |
| compaction | canned summary via FakeProvider; old messages collapse, last 4 turns kept; invariant holds |
| ui | scripted REPL input (`/help`, prompt, `/exit`); rendering goldens with fake clock/usage |

Cross-cutting: `ValidateTranscript` is asserted after every transcript mutation in every
test that touches one.

## 14. Future work

- CLI-subprocess backends (codex / claude) behind a separate "delegate" abstraction.
- OpenAI Responses API dialect.
- `.gitignore`-aware grep; markdown rendering; parallel tool execution where safe.
- MCP client support; sub-agent spawning.
- Smarter prompt-cache breakpoint placement.
