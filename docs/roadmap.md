# Roadmap

Prioritized improvements to the agent loop and the default command set,
from a June 2026 review of `internal/agent`, `internal/llm`, and
`internal/tools`. Items reference the code as of that review; re-verify
line-level claims before starting one.

## Done

1. **`exec` tool + `stdin` on `run_command`.** Shell quoting was the dominant
   model failure when generated content (commit messages, `python -c`
   one-liners, sed programs, JSON) traveled through `run_command` as part of a
   command line. `exec` (argv array, no shell — design §9.8) eliminates the
   failure class for arguments; `stdin` on both tools eliminates it for
   documents (`git commit -F -`, `python -`, `tee file`).
2. **Mid-stream retry.** The agent loop re-requests a step from scratch when
   its stream fails after the first byte (truncated body, retryable error
   frame, transport reset), 2 retries per step with backoff. Failed-attempt
   usage still counts toward the turn total.
3. **Defensive usage accounting.** Usage events are documented as cumulative
   snapshots and merged element-wise (max) in the loop and compaction, so a
   zeroed or partial late frame cannot erase earlier numbers.
4. **Per-tool-call timeout ceiling in `Dispatch`.** Every tool call gets an
   11-minute ceiling (largest self-limit + grace); Dispatch unblocks even for
   tools that ignore ctx, returning a timed-out is_error result.
5. **Anthropic cache-breakpoint tuning.** A third breakpoint after the static
   tool-schema array preserves the cached tools segment across system-prompt
   changes (e.g. run-mode switches).
6. **gitignore-aware grep.** Inside git repos, grep's candidate files come
   from `git ls-files --cached --others --exclude-standard`, delegating all
   ignore semantics to git; the denylist walk remains the non-repo /
   `no_ignore` fallback. The RE2 contract and output caps are unchanged.
7. **Proactive compaction.** The step loop estimates the transcript before
   every request and compacts mid-turn at the same 78% threshold; the
   reported-tokens signal now counts cached tokens too.
8. **maxSteps auto-continue.** `on_max_steps = continue` grants up to 3 fresh
   step budgets before stopping; default behavior is unchanged.
9. **Parallel dispatch of read-only tool calls.** Steps whose calls are all
   read-only dispatch concurrently (bounded at 8); ordering of sink events,
   results, and transcript blocks is unchanged. Mixed steps stay sequential.
