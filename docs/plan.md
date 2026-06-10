# harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the agentic coding harness specified in `docs/design.md` — a stdlib-only Go CLI that drives a tool-using LLM loop over Anthropic- and OpenAI-style streaming APIs.

**Architecture:** Provider-agnostic message/streaming model in `internal/llm` with two HTTP dialect packages; a sequential tool-dispatch agent loop; nine filesystem/exec/git/web tools behind one interface; line-oriented REPL plus `-p` one-shot mode. See `docs/design.md` for all behavioral contracts — every task below cites the design section it implements.

**Tech Stack:** Go 1.24+ (`iter`, range-over-func), standard library only. Tests: `go test`, `httptest`, `t.TempDir()`.

**Conventions for every task:**
- TDD: write the failing test, see it fail, implement, see it pass, commit.
- Conventional-commit messages (`feat:`, `test:`, `docs:`, `refactor:`).
- After every phase: `go build ./... && go vet ./... && go test ./...` must be green.
- The transcript invariant helper `llm.ValidateTranscript` (Phase 1) must be asserted in every later test that mutates a transcript.

---

## Phase 1 — Module scaffolding and core types

Implements design §4 (message model), §5.1 (Provider contract), §6 (usage/cost/model registry).

**Files:**
- Create: `go.mod` (`module harness`, `go 1.24`)
- Create: `.gitignore` (`/harness`, `*.test`)
- Create: `internal/llm/message.go` — `Role`, `Message`, `BlockKind`, `ContentBlock`, plus the seam types `ToolCall`/`ToolResult`, exactly as design §4
- Create: `internal/llm/provider.go` — `Provider`, `Request`, `ToolSchema`, `StreamEvent`, `EventKind`, `StopReason`, `Usage` exactly as design §5.1
- Create: `internal/llm/validate.go` — `func ValidateTranscript(msgs []Message) error`
- Create: `internal/llm/model.go` — `Price`, `ModelInfo`, registry map, `Cost(model string, u Usage) (float64, bool)`, `ContextWindow(model string) int`
- Test: `internal/llm/validate_test.go`, `internal/llm/model_test.go`

- [ ] **Step 1:** `git init` is done; run `go mod init harness` and commit `chore: initialize Go module`.
- [ ] **Step 2:** Write `validate_test.go` — table-driven cases:
  - valid: empty transcript; user→assistant text; assistant with 2 tool_use followed by user with 2 matching tool_result.
  - invalid: tool_use with no following tool_result; tool_result with no preceding tool_use (orphan); tool_result IDs not matching; two results for one call; tool_result appearing in an assistant message.
- [ ] **Step 3:** Run `go test ./internal/llm/` — expect FAIL (undefined types).
- [ ] **Step 4:** Implement `message.go`, `provider.go`, `validate.go`. `ValidateTranscript` walks messages tracking the set of open tool_use IDs; any assistant message closes the previous set only if empty.
- [ ] **Step 5:** Run `go test ./internal/llm/` — expect PASS.
- [ ] **Step 6:** Write registry tests: known model returns `(cost>0, true)`; unknown model returns `(0, false)`; `ContextWindow("unknown-model")` returns 256000; provider config files load models, context windows, and prices.
- [ ] **Step 7:** Implement runtime model registry loading from provider config files; tests PASS.
- [ ] **Step 8:** Commit `feat: core llm types, transcript validation, model registry`.

## Phase 2 — SSE reader and retry policy

Implements design §5.2 (SSE) and §5.5 (retries).

**Files:**
- Create: `internal/sse/sse.go` — `Event{Type, Data string}`, `Read(ctx, io.Reader) iter.Seq2[Event, error]`, `ErrTruncatedStream` (exported for providers to wrap)
- Create: `internal/retry/retry.go` — `Next(attempt int, retryAfter time.Duration) time.Duration` (full jitter: `rand(0, min(30s, 500ms·2^attempt))`, Retry-After as floor). The retry *loop* lives in each provider (it owns `APIError.Retryable` and the ctx); providers take a `sleep func(time.Duration)` field defaulting to `time.Sleep` so tests run instantly.
- Test: `internal/sse/sse_test.go`, `internal/retry/retry_test.go`

- [ ] **Step 1:** Write `sse_test.go` — table-driven over raw byte inputs:
  - single `data:` frame; multi-line data joined with `\n`; `event:` + `data:` pair; leading-space stripping (`data: x` vs `data:x`); comment lines (`: ping`) ignored; two frames separated by blank line; CRLF line endings; a 700KB single data line (buffer sizing); input ending mid-frame without blank line yields the partial frame then clean EOF; context cancelled mid-read yields `ctx.Err()`.
- [ ] **Step 2:** Run — FAIL. Implement with `bufio.Scanner` (1MB max token). PASS.
- [ ] **Step 3:** Write `retry_test.go` for `Next`: growth bounded by the 30s cap; jitter within `[0, base·2^n]` (sample many draws); `retryAfter=2s` floors the result. (Loop behavior — stop on success, give up after budget, no retry on fatal errors, ctx short-circuit — is tested in the provider tests, Phases 3–4, via the injected sleeper.)
- [ ] **Step 4:** Run — FAIL. Implement. PASS.
- [ ] **Step 5:** Commit `feat: sse reader and retry policy`.

## Phase 3 — Anthropic provider

Implements design §5.3, §5.4, §5.5 for the Messages dialect, including prompt caching.

**Files:**
- Create: `internal/llm/anthropic/wire.go` — request/response/event JSON structs
- Create: `internal/llm/anthropic/provider.go` — `New(cfg)`, `Stream` (HTTP + sse.Read + event decode), retry-before-first-byte, `APIError` mapping
- Create: `internal/llm/anthropic/assemble.go` — per-index tool_use assembler
- Create: `internal/llm/apierror.go` — shared `APIError` type (design §5.5); move here now, both dialects use it
- Test: `internal/llm/anthropic/provider_test.go`, `request_test.go`; fixtures under `internal/llm/anthropic/testdata/*.sse` and `*_request.json`

- [ ] **Step 1:** Write `request_test.go` golden tests. Build a `Request` covering: system prompt; multi-turn with assistant tool_use and user tool_result (incl. `is_error`); 2 tools; MaxTokens unset (assert default `min(8192, window/4)` is sent); nil Temperature omitted; cache_control present on system block and last message block. Compare marshaled JSON to `testdata/basic_request.json` (write the golden by hand from the design §4/§5.4 tables — it documents the wire format).
- [ ] **Step 2:** FAIL → implement `wire.go` request building → PASS.
- [ ] **Step 3:** Create stream fixtures (hand-written from the design, verified against real API docs at implementation time):
  - `text_only.sse`: message_start → content_block text deltas → message_delta (usage, end_turn) → message_stop.
  - `tool_call.sse`: text block + tool_use block with input_json_delta fragments split mid-token.
  - `parallel_tools.sse`: two interleaved tool_use blocks.
  - `empty_args.sse`: tool_use with no input_json_delta (assert `{}`).
  - `error_frame.sse`: mid-stream `error` event (overloaded).
  - `truncated.sse`: ends after a content_block_delta, no message_stop.
- [ ] **Step 4:** Write `provider_test.go`: serve each fixture from `httptest.Server`; collect `[]StreamEvent` and terminal error; assert exact ordered events, assembled `ToolInput` JSON, final Usage (input from message_start + accumulated output from message_delta, cache fields mapped), StopReason mapping (`end_turn`/`tool_use`/`max_tokens`/`stop_sequence`→`stop`). Also: 429-then-200 retries (fake sleeper); 400 fails immediately with APIError; ctx cancel mid-stream yields ctx.Err(); `ping` frames ignored.
- [ ] **Step 5:** FAIL → implement `provider.go` + `assemble.go` → PASS. Headers: `x-api-key`, `anthropic-version: 2023-06-01`, `content-type: application/json`.
- [ ] **Step 6:** Commit `feat: anthropic messages provider`.

## Phase 4 — OpenAI provider

Implements design §5.3, §5.4, §5.5 for Chat Completions.

**Files:**
- Create: `internal/llm/openai/wire.go`, `provider.go`, `assemble.go`
- Test: `internal/llm/openai/provider_test.go`, `request_test.go`; fixtures `internal/llm/openai/testdata/`

- [ ] **Step 1:** Golden request test mirroring Phase 3, asserting the OpenAI mapping rules (design §4): system → leading system message; tool_result blocks hoisted to sibling `role:"tool"` messages in call order; `is_error` results prefixed `ERROR: `; tool_use → `tool_calls[]` with `arguments` as JSON string (`"{}"` for empty); assistant message with no text omits `content`; `stream_options.include_usage:true` always present; MaxTokens omitted when unset.
- [ ] **Step 2:** FAIL → implement request building → PASS.
- [ ] **Step 3:** Stream fixtures: `text_only.sse` (role chunk, content deltas, finish_reason stop, usage chunk, `[DONE]`); `tool_call.sse` (indexed tool_call deltas, fragmented arguments, finish_reason tool_calls); `parallel_tools.sse`; `empty_args.sse`; `no_usage.sse` (compatible server sends no usage chunk — Usage zero, no crash); `truncated.sse` (no `[DONE]`).
- [ ] **Step 4:** Provider tests as Phase 3, plus: cached-token normalization (`prompt_tokens_details.cached_tokens` subtracted from InputTokens); unknown finish_reason → `end_turn`.
- [ ] **Step 5:** FAIL → implement → PASS.
- [ ] **Step 6:** Add `internal/llm/factory.go`: define `Options{Provider, Model, BaseURL, APIKey string; MaxTokens int; Temperature *float64; ContextWindow int}` and `New(opts Options) (Provider, error)` with provider inference (`claude*` prefix → anthropic, else openai; explicit `Provider` wins). Note: the dialect packages register constructors with `internal/llm` (or `New` lives in a tiny `internal/llm/factory` package importing both dialects) to avoid an import cycle — decide at implementation, document the choice in a comment. Add `factory_test.go` (inference table, missing-key validation, empty key allowed with custom base URL). Commit `feat: openai chat-completions provider and provider factory`.

## Phase 5 — Tool foundation and file tools

Implements design §9 interface/registry/dispatch and tools §9.1–§9.5.

**Files:**
- Create: `internal/tools/tool.go` — `Tool` interface, `Registry`, `Register`, `Specs`, `Dispatch` (panic recovery, unknown-tool, invalid-args, central truncation)
- Create: `internal/tools/truncate.go` — 64KB/1000-line cap with marker
- Create: `internal/tools/readfile.go`, `listdir.go`, `grep.go`, `edit.go`, `writefile.go`
- Test: one `_test.go` per file, all table-driven against `t.TempDir()`

- [ ] **Step 1:** `tool_test.go`: Dispatch returns is_error result for unknown tool / invalid JSON args / tool error / tool panic (register a deliberately panicking fake); successful result passes through; >64KB output truncated with marker text containing original size; >1000-line output truncated by lines.
- [ ] **Step 2:** FAIL → implement `tool.go` + `truncate.go` → PASS. Commit `feat: tool registry and dispatch`.
- [ ] **Step 3:** Per tool, write tests then implement (one commit each, `feat: read_file tool` etc.). Required cases per design:
  - **read_file:** line numbering format; offset/limit windowing; offset past EOF error names line count; missing file; directory → "use list_dir"; NUL-sniff binary rejection; empty file marker; default 1000-line cap noted.
  - **list_dir:** dirs-first ordering with `/` suffix; glob filter; 1000-entry cap; non-dir error; unreadable entry shows `?` and continues.
  - **grep:** basic match `path:line:text`; `(?i)` via ignore_case; glob filter; max_matches cap marker; denylist dirs skipped (create `.git/`, `node_modules/` in fixture); binary skipped; single-file path; invalid pattern error text; 300-char line cap; `(no matches)`.
  - **edit:** single replacement; 0 matches error; N>1 error message mentions replace_all; replace_all path; empty old_string rejected; old==new rejected; missing file directs to write_file.
  - **write_file:** create with parent mkdir; overwrite reports `overwrote`; reports bytes/lines; path-is-dir error; trailing `/` error.

## Phase 6 — apply_patch

Implements design §9.6. The largest isolated chunk — keep it its own phase.

**Files:**
- Create: `internal/tools/patch/parse.go` — unified-diff parser (`FilePatch{Old, New string, Hunks []Hunk, IsCreate, IsDelete, IsRename bool}`)
- Create: `internal/tools/patch/apply.go` — 3-level hunk matching (exact → ±200-line offset search → whitespace-normalized), per-file atomic apply
- Create: `internal/tools/applypatch.go` — the Tool wrapper, applied/rejected report
- Test: `internal/tools/patch/parse_test.go`, `apply_test.go`, `internal/tools/applypatch_test.go`

- [ ] **Step 1:** `parse_test.go`: single hunk; multi-hunk; multi-file; create (`--- /dev/null`); delete (`+++ /dev/null`); rename (git extended headers); `a/`+`b/` prefix stripping; malformed hunk header error; no-newline-at-EOF marker (`\ No newline`).
- [ ] **Step 2:** FAIL → implement parser → PASS. Commit `feat: unified diff parser`.
- [ ] **Step 3:** `apply_test.go`: exact application; hunk applies at shifted offset (insert lines above target first); whitespace-drift application (tabs→spaces in file); failing hunk leaves file untouched; later hunks shift after earlier insertions; create/delete/rename end-to-end; delete with mismatched content rejected.
- [ ] **Step 4:** FAIL → implement → PASS. Commit `feat: patch application with fuzzy matching`.
- [ ] **Step 5:** `applypatch_test.go`: multi-file patch with one bad file — good files applied, report lists `applied:` and `rejected: <file> (hunk i of n did not match)`; create-where-exists rejected. PASS → commit `feat: apply_patch tool`.

## Phase 7 — Exec tools: run_command, git, web_fetch

Implements design §9.7–§9.9.

**Files:**
- Create: `internal/tools/runcommand.go`, `git.go`, `webfetch.go`
- Test: matching `_test.go` files

- [ ] **Step 1:** **run_command** tests: echo round-trip with `[exit code: 0]`; non-zero exit reported but NOT is_error (`false` → `[exit code: 1]`); stdout+stderr interleaved; cwd honored; timeout kills a `sleep 30` process group within the limit and reports partial output (use 1s timeout); missing cwd error. Implement: `bash -lc` with `sh -c` fallback, `Setpgid`, group kill on ctx/timeout. Commit `feat: run_command tool`.
- [ ] **Step 2:** **git** tests against a scratch repo (`exec git init` in `t.TempDir()`, skip if git absent): status/add/commit/log round-trip; `--no-pager` injected; non-repo error surfaces git's message; `GIT_TERMINAL_PROMPT=0` set (assert via env inspection seam). Commit `feat: git tool`.
- [ ] **Step 3:** **web_fetch** tests against `httptest.Server`: text/plain raw; JSON raw; HTML reduced (script/style dropped, tags stripped, entities unescaped); max_bytes stops reading; redirect followed and final URL in header line; non-2xx returns body as content; non-http scheme rejected; binary content-type rejected. Commit `feat: web_fetch tool`.

## Phase 8 — Agent loop, system prompt, interrupts

Implements design §8 entirely.

**Files:**
- Create: `internal/agent/agent.go` — `Agent{provider, registry, transcript, opts}`, `RunTurn(ctx, userText string, sink EventSink) error`; `EventSink` is the UI callback interface (text deltas, tool start/result, usage)
- Create: `internal/agent/interrupt.go` — SIGINT watcher state machine (first ^C cancels turn, second/idle exits)
- Create: `internal/llm/llmtest/fake.go` — scripted `FakeProvider` (each scripted step yields canned StreamEvents; records the Requests it receives). A real package, not a `_test.go` file, because Phases 10–11 (`internal/ui`, compaction tests) need it too.
- Create: `internal/sysprompt/sysprompt.go` — builtin instructions constant + env block builder
- Test: `internal/agent/agent_test.go`, `internal/sysprompt/sysprompt_test.go`

- [ ] **Step 1:** `agent_test.go` with FakeProvider + a recording fake tool:
  - text-only turn appends user+assistant, returns at end_turn.
  - tool turn: 2 parallel calls executed sequentially in emission order; one user message with 2 results in order; loop re-calls provider; `ValidateTranscript` passes after every step.
  - failing tool: error string fed back as is_error result; next request (captured by FakeProvider) carries it.
  - max-steps: FakeProvider always returns tool_use; loop stops at the limit, transcript valid, sink told.
  - cancellation: cancel ctx mid-stream; partial text kept as text-only assistant message, no dangling tool_use, `ValidateTranscript` passes.
  - usage accumulation across steps reported to sink.
- [ ] **Step 2:** FAIL → implement `agent.go` → PASS. Commit `feat: agent turn loop`.
- [ ] **Step 3:** `sysprompt_test.go`: env block contains cwd/os/date; git summary against scratch repo (branch, modified/untracked counts); non-repo line; append vs override vs `-no-env` composition. Implement → PASS → commit `feat: system prompt builder`.
- [ ] **Step 4:** Interrupt watcher: unit-test the state machine with injected signal channel (first signal → cancel func called; second within window → exit request; signal at idle → exit request). Implement → commit `feat: interrupt handling`.

## Phase 9 — Config and session persistence

Implements design §7 and §11.

**Files:**
- Create: `internal/config/config.go` — `Config` struct, `Load(args []string, getenv func(string) string, configPath string) (Config, error)` implementing flags > env > file > defaults
- Create: `internal/session/session.go` — `Session` struct (design §11), `Save` (tmp+rename), `Load`, `DefaultPath()`, resume repair (synthesize `interrupted` tool_result for dangling tool_use)
- Test: `internal/config/config_test.go`, `internal/session/session_test.go`

- [ ] **Step 1:** Config tests: each precedence pairing (flag beats env beats file beats default) for model/provider/base-url; API key read from env only; `HARNESS_*` mapping; bad flag → usage error. Implement with `flag.NewFlagSet` (testable, no globals). Commit `feat: configuration resolution`.
- [ ] **Step 2:** Session tests: save→load round-trip equality; no `.tmp` file left after save; load of transcript ending in dangling tool_use produces repaired transcript passing `ValidateTranscript`; cross-provider neutrality (saved tags contain no `function`/`tool_calls` strings). Implement → commit `feat: session persistence`.

## Phase 10 — REPL, one-shot mode, main wiring

Implements design §10. First end-to-end dogfood point.

**Files:**
- Create: `internal/ui/render.go` — streaming printer, tool one-liners, usage line, TTY/color detection (injectable `isTTY`)
- Create: `internal/ui/repl.go` — `Run(in io.Reader, out, errw io.Writer, agent *agent.Agent, ...) int`; meta-commands `/help /exit /quit /clear /compact /usage /save /model`; `//` escape
- Create: `internal/ui/oneshot.go` — one user turn; assistant→out, noise→errw; exit codes 0/1/2/130
- Create: `cmd/harness/main.go` — config load, provider factory, registry assembly, signal wiring, REPL-vs-oneshot dispatch
- Test: `internal/ui/repl_test.go`, `render_test.go`, `oneshot_test.go` (all via injected readers/writers + FakeProvider)

- [ ] **Step 1:** Render tests: tool summary line format; usage line with known model (cost shown) and unknown model (cost omitted); color suppressed when not TTY.
- [ ] **Step 2:** REPL tests: scripted stdin (`/help`, a prompt, `/exit`) asserts help text, agent invoked once, clean return; `/clear` resets transcript and rotates session; unknown `/cmd` message; `//literal` sent as prompt.
- [ ] **Step 3:** One-shot tests: assistant text only on stdout, tool noise on stderr; `-p -` reads stdin; flag text + piped stdin concatenated; exit code 1 on provider error.
- [ ] **Step 4:** Implement, then wire `main.go`. Manual smoke: `go run ./cmd/harness -p "hello" -model <real model>` against a real key (document in commit message that this was done). Commit `feat: repl, one-shot mode, main entrypoint`.

## Phase 11 — Compaction and usage/cost display

Implements design §12 and finishes §6 display.

**Files:**
- Create: `internal/agent/compact.go` — threshold check (≥78% of the injected model registry context window), summary request, transcript replacement, degradation ladder
- Modify: `internal/agent/agent.go` — post-turn trigger; `internal/ui/repl.go` — `/compact`, `/usage`
- Test: `internal/agent/compact_test.go`

- [ ] **Step 1:** Tests with FakeProvider returning a canned summary: transcript above threshold compacts to [summary message + last 4 turns], `ValidateTranscript` passes, kept turns are whole turns; below threshold untouched; summary-call failure leaves transcript intact and reports warning; degradation: oversized kept-turns drop to 1; compaction usage added to totals.
- [ ] **Step 2:** FAIL → implement → PASS. Commit `feat: context compaction`.

## Phase 12 — Polish

- [ ] **Step 1:** Write `README.md`: what it is, build (`go build ./cmd/harness`), quickstart for both providers + one local server (Ollama), flag reference, session/compaction behavior, design-doc pointer.
- [ ] **Step 2:** Flag `-h` output reviewed for accuracy against design §10.
- [ ] **Step 3:** Manual smoke matrix (document results in the commit message): Anthropic real API (tool round-trip: ask it to read a file); OpenAI real API; one OpenAI-compatible local server; ^C during a stream; resume of an interrupted session.
- [ ] **Step 4:** `go vet ./... && go test ./...` green. Commit `docs: readme and final polish`.

---

## Self-review checklist (run at the end of every phase)

1. Does the phase's behavior match the cited design section exactly? Divergence → fix code or amend `docs/design.md` in the same commit, never silently.
2. `ValidateTranscript` asserted in every test that mutates a transcript?
3. No test sleeps on real time (fake clocks/sleepers everywhere)?
4. `go build ./... && go vet ./... && go test ./...` green?
