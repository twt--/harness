# Roadmap items 2–9: design

Approved design for implementing every open item in `docs/roadmap.md`
(items 2–9), including the two items flagged against v1 non-goals, which
are explicitly adopted here. Work is executed by parallel subagents in
isolated worktrees, grouped into waves by file-conflict analysis, and
merged to local main between waves.

References: `docs/design.md` (the v1 design), `docs/roadmap.md`,
`AGENTS.md`. Code references are to the tree at commit `421bb63`.

## 1. Execution architecture

### File-conflict map

| Item | Files touched |
|---|---|
| #2 mid-stream retry | `internal/agent/agent.go` (`stream`, `RunTurn`) |
| #3 proactive compaction | `internal/agent/agent.go` (step loop), `internal/agent/compact.go` |
| #4 maxSteps auto-continue | `internal/agent/agent.go` (loop exit), `internal/config` |
| #5 dispatch timeout | `internal/tools/tool.go` (`Dispatch`) |
| #6 defensive usage | `internal/agent/agent.go` (`stream`), `internal/agent/compact.go` (`summarize`), both providers |
| #7 cache breakpoints | `internal/llm/anthropic/wire.go` |
| #8 parallel read-only dispatch | `internal/agent/agent.go` (tool loop), `internal/tools/tool.go`, docs |
| #9 gitignore-aware grep | `internal/tools/grep.go`, docs |

`agent.go` is the contention hotspot (5 of 8 items); #2 and #6 touch the
same function (`stream`). Lanes are drawn so that no two concurrent
lanes touch the same file.

### Waves

- **Wave 0** (orchestrator, directly on main): one `docs:` commit
  amending the non-goals (§10 below). Removes the docs conflict between
  #8 and #9 and makes the scope change explicit rather than incidental.
- **Wave 1** — four parallel lanes, zero file overlap:
  - Lane 1: #2 then #6 (sequential within the lane; same function).
  - Lane 2: #5.
  - Lane 3: #7.
  - Lane 4: #9.
- **Wave 2** — one lane: #3 then #4, rebased on merged wave 1.
- **Wave 3** — one lane: #8 (needs the post-#2 `stream`/loop and the
  post-#5 `tools` package stable).

### Process per lane

- Isolated git worktree on a feature branch (`feat/<topic>`),
  conventional commits, regression tests included with each change.
- Gate before merge: `go build ./... && go vet ./... && go test ./...`
  green in the worktree.
- The orchestrator reviews each lane's diff, merges to local main with
  a merge commit (matching existing history style), and re-runs the
  full test suite on main after each wave. No PRs.

## 2. Item #2 — mid-stream retry

**Problem.** `internal/retry` protects only connection setup. Once the
first byte streams, any failure (truncated body, mid-stream `error`
frame, transport read error) is turn-fatal: `Agent.stream` returns the
error and `RunTurn` ends the turn (`agent.go:174-189`).

**Design.** Retry at the agent level — not in the providers — so both
dialects are covered once. Wrap the per-step stream consumption in a
new `streamWithRetry` helper that `RunTurn` calls in place of `stream`:

- Budget: **2 retries per step** (3 attempts total), reset each step.
  Backoff between attempts via the existing `retry.Next(attempt, 0)`.
- Retryable: errors wrapping `sse.ErrTruncatedStream`; `*llm.APIError`
  with `Retryable == true`; transport/read errors that are neither
  `*llm.APIError` nor context errors.
- Not retryable: `context.Canceled` / `context.DeadlineExceeded`
  (existing cancel repair applies unchanged); `*llm.APIError` with
  `Retryable == false` (for example invalid_request).
- On a retryable failure: discard the partial `stepResult` (nothing was
  committed to the transcript — the existing invariant), emit
  `sink.Notice("[stream interrupted: <err>; retrying step]")`, sleep
  the backoff, and re-request the step from scratch. The re-attempt
  streams text deltas to the sink again from the beginning.
- Usage accounting: usage reported by failed attempts (Anthropic sends
  input tokens on `message_start`) is folded into the turn total — the
  tokens were paid for — but never drives `lastInput` (the compaction
  trigger), which only ever reflects a completed step.
- When the budget is exhausted, behavior is exactly today's: the last
  error propagates, cancel-repair text handling applies, the turn ends.

**Tests.** Fake provider that fails mid-stream N times then succeeds:
assert transcript correctness, notice emission, attempt cap, usage
accumulation, and that context cancellation is not retried.

## 3. Item #6 — defensive usage accounting

**Problem.** `Agent.stream` overwrites `res.usage` on every
`EventUsage` (`agent.go:256-258`), trusting providers to send
cumulative snapshots. Correct for both current dialects; brittle for
OpenAI-compatible servers that send deltas or a zeroed/partial final
frame.

**Design.**

- Document the contract on `llm.Provider`: usage events carry
  **cumulative snapshots**, never deltas.
- Make consumers defensive anyway: `Agent.stream` and
  `compact.summarize` merge each incoming usage with **element-wise
  max** instead of overwrite. A zeroed or partially-populated late
  frame can no longer erase earlier numbers. (A true delta-sending
  server would undercount, but max never makes today's correct inputs
  wrong, and undercounting beats the current overwrite-to-zero
  failure.)
- Verify the OpenAI request sets `stream_options.include_usage`; add it
  if missing (`openai/wire.go`).

**Tests.** Usage frames arriving out of order, zeroed final frame,
fields populated in different frames; per-provider tests confirming
cumulative emission.

## 4. Item #3 — proactive compaction

**Problem.** `MaybeCompact` fires only after a turn, using the last
step's reported input tokens (`agent.go:225`, `compact.go:42-48`). A
turn whose tool results balloon the context can overflow the next
request before compaction ever runs.

**Design.** At the top of the step loop in `RunTurn`, before building
each request: compute `est = estimateTokens(a.transcript)` (the
existing bytes/4 heuristic in `compact.go`) and trigger
`a.Compact(ctx, sink)` when

```
max(lastInput, est) * 100 >= window * compactThresholdPct
```

- Reuses the existing 78% threshold and the existing `Compact`, whose
  keep-whole-turns rule already preserves the tool_use/tool_result
  invariant mid-turn: the in-flight turn is the newest turn, so it is
  kept verbatim.
- The post-turn `MaybeCompact` call stays (it catches the
  reported-tokens signal, which is more accurate than the estimate).
- Mid-turn compaction usage folds into the turn total, as post-turn
  compaction usage does today.
- A mid-turn compaction failure follows `Compact`'s existing contract:
  warning notice, transcript intact, the step proceeds (a visible
  context-length failure beats silent data loss).

**Tests.** A turn whose tool results inflate the transcript past the
threshold triggers compaction before the next step; under-threshold
turns do not compact mid-turn; usage totals include the summary call.

## 5. Item #4 — maxSteps auto-continue

**Problem.** Exhausting the 50-step cap stops with a "say continue"
notice (`agent.go:214-217`, `292-294`).

**Design.** New config knob in `internal/config`:

- `on_max_steps = "stop" | "continue"`, default `"stop"` (today's
  behavior, notice text unchanged).
- `"continue"`: when the step budget exhausts while the model still
  wants tools, grant a fresh budget and keep looping — no synthetic
  user message is needed; the transcript is already valid — emitting
  `sink.Notice("[max steps reached; auto-continuing (n/3)]")`.
- Hard cap: **3 auto-continues** (so at most 4 budgets per turn), then
  the existing stop notice. The cap is a constant, not config.
- No summarize-and-continue: proactive compaction (#3) already bounds
  context growth, so summarizing here would be redundant.
- Plumbing: `config` → `agent.Options` → `Agent`, following the
  existing pattern used by `MaxSteps`/`ContextWindow`. Wire through CLI
  startup the same way run modes were.

**Tests.** Default stops exactly as today (regression); continue mode
loops past the cap, emits the per-continue notice, and stops after the
hard cap.

## 6. Item #5 — per-tool-call timeout ceiling in Dispatch

**Problem.** `run_command`/`exec` self-limit, but a hanging `web_fetch`
(or any future tool) blocks the turn until ^C
(`tools/tool.go:139-177`).

**Design.** `Registry.Dispatch` wraps the call's context:
`context.WithTimeout(ctx, r.dispatchTimeout)`.

- Default **5 minutes**, stored as a `Registry` field; exported setter
  `SetDispatchTimeout(d time.Duration)` for tests and future config; no
  config-file knob yet.
- On expiry the result is the standard `is_error` shape:
  `error: tool timed out after 5m0s`. Distinguish the ceiling firing
  (`ctx.Err() == context.DeadlineExceeded` from the derived context)
  from an outer cancellation, which must keep propagating as
  cancellation, not as a tool error.
- Tools already honor context (`grep` checks `ctx.Err()`, exec tools
  use `CommandContext`), so no per-tool changes.
- Verify the ceiling exceeds `run_command`/`exec` self-limits so it
  never fires first for well-behaved tools; adjust the default upward
  if any self-limit is higher.

**Tests.** A stub tool that blocks on `ctx.Done()` returns a timeout
`is_error` result; outer cancellation still surfaces as cancellation;
fast tools are unaffected.

## 7. Item #7 — Anthropic cache-breakpoint tuning

**Problem.** Only two breakpoints today: the system block and the last
content block of the final message (`anthropic/wire.go:140-145`,
`204-213`). Anthropic's cache prefix order is tools → system →
messages, so the system breakpoint covers tools+system together:
anything that changes the system text (a `/mode` run-mode switch,
sysprompt edits between sessions) invalidates the cached tool schemas
too.

**Design.** In `buildRequest`, set `CacheControl: ephemeral` on the
**last entry of the tools array** when tools are present — a third
breakpoint (3 of the 4 allowed), preserving the tools segment across
system-prompt changes. No other behavior change.

**Tests.** Wire-level request tests assert breakpoint placement with
and without tools; existing two-breakpoint assertions updated.

## 8. Item #8 — parallel dispatch of read-only tool calls

Adopted from the flagged list; the non-goal is amended in wave 0.

**Design.**

- `Tool` gains a `ReadOnly() bool` method (full interface extension —
  backward compatibility is a non-concern). Classification:
  - `true`: `read_file`, `list_dir`, `grep`, `git_readonly`,
    `web_fetch` (network GET; mutates no workspace state).
  - `false`: `edit`, `write_file`, `apply_patch`, `run_command`,
    `exec`, `git`, `write_tmp_file`.
- In `RunTurn`: when a step emits **2+ tool calls and all are
  read-only**, dispatch them concurrently — goroutines +
  `sync.WaitGroup`, concurrency bounded at **8** — collecting results
  into a slice indexed by emission position. Any step containing a
  non-read-only call runs fully sequential, exactly as today. Simple,
  predictable v1.1 semantics: reads parallelize, everything else is
  serial.
- Sink ordering stays deterministic: all `ToolStart` events are emitted
  in emission order before dispatch begins; all `ToolResult` events are
  emitted in emission order after the group completes. The sink is
  never called from more than one goroutine.
- Transcript: results append in emission order exactly as today; the §4
  invariant is untouched.
- Docs: update design §8 prose to describe the read-only parallel path.

**Tests.** A step with N read-only calls dispatches concurrently
(observable via a stub tool that blocks until all have started) and
appends results in emission order; mixed steps stay sequential
(observable via a stub recording dispatch interleaving); sink event
order is deterministic.

## 9. Item #9 — gitignore-aware grep via git file listing

Adopted from the flagged list; the non-goal is amended in wave 0.
Direction chosen with the user: delegate ignore semantics to git
instead of implementing a gitignore matcher.

**Design.** Source the candidate file list from git when possible;
keep the existing RE2 match loop, binary sniffing, caps, and
`path:line:text` output unchanged.

- When the search root is inside a git work tree and `no_ignore` is not
  set, list candidates with:

  ```
  git -C <root> ls-files --cached --others --exclude-standard -z -- .
  ```

  (tracked + untracked-but-not-ignored — `git grep --untracked`
  semantics). Run the existing `grepFile` over that list. Files listed
  but missing from the worktree (deleted-but-tracked) already fall out:
  `grepFile`'s `os.Stat` failure skips them.
- Glob filtering, match caps, line truncation, and relative-path
  display all behave exactly as the walker does today.
- Fallbacks — used when any of these hold: the root is not in a git
  work tree (detect via `git rev-parse --is-inside-work-tree`, or
  simply on `ls-files` failure); the `git` binary is missing; or the
  new optional arg `no_ignore: true` is set. The fallback is today's
  `WalkDir` + denylist walk, unchanged.
- The `.git` directory is never searched in either path; the denylist
  continues to apply in the fallback path only. Single-file roots
  bypass listing entirely (today's behavior).
- Why not `git grep` directly: the tool's schema advertises Go RE2
  patterns; `git grep` speaks POSIX ERE (PCRE only if compiled in).
  Model-written patterns like `(?i)` or `\d` would silently change
  meaning, and the existing caps/output shaping would be lost.
- Docs: update design §9.3 to describe git-backed listing with the
  denylist as fallback.

**Tests.** In a temp git repo: ignored files are excluded, untracked
non-ignored files are included, nested/negated `.gitignore` rules are
honored (free via git), `no_ignore` restores the walker, non-repo
directories use the walker, and output format/caps are unchanged
(regression).

## 10. Wave 0 — non-goals amendment

- `docs/design.md` §1: remove "parallel tool execution" and
  "`.gitignore`-aware search" from the v1 non-goals, noting they are
  adopted as v1.1 scope by this spec (MCP, sub-agents, markdown
  rendering, and the rest stay non-goals).
- `AGENTS.md` (line 103): same amendment, pointing at this spec.
- `docs/roadmap.md` is updated as items land (each lane moves its item
  to "Done" in its final commit).

## 11. Error handling summary

- Mid-stream retry never retries cancellation and never commits partial
  output; exhausted budget degrades to today's behavior.
- Mid-turn compaction failure keeps the transcript intact and proceeds.
- Dispatch timeout converts only the ceiling's own expiry into a tool
  error; outer cancellation stays cancellation.
- Grep falls back to the walker whenever git is unavailable or the
  listing fails — never an error surfaced to the model for a non-repo
  path.

## 12. Testing strategy

- Every item ships table-driven regression tests in the existing style
  (fake providers / `httptest` for stream behavior, stub tools for
  dispatch, temp git repos for grep).
- Per-lane gate: `go build ./... && go vet ./... && go test ./...`.
- Orchestrator re-runs the full suite on main after each wave's merges
  and reviews each lane's diff before merging.
