// Package ui drives the harness from stdin: a streaming renderer implementing
// the agent's EventSink, an interactive REPL with meta-commands, and a one-shot
// mode for piping a single prompt (design §10).
package ui

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	"harness/internal/agent"
	"harness/internal/llm"
	"harness/internal/tools"
)

// dim/reset are the only ANSI codes used; rendering is legible without color
// (design §2, §10). They are emitted only when RenderOptions.Color is set.
const (
	ansiDim              = "\x1b[2m"
	ansiReset            = "\x1b[0m"
	TimestampShortLayout = "15:04:05"
	TimestampFullLayout  = "2006-01-02 15:04:05"
)

// snippetLines caps the verbose result preview (design §10: "first ~5 lines").
const snippetLines = 5

// RenderOptions configures a Renderer. Color is decided by the caller (TTY check
// plus NO_COLOR / -no-color); Now is injected so the per-turn duration is
// deterministic in tests (design §10, §13).
type RenderOptions struct {
	Color           bool
	Verbose         bool
	ToolStream      bool
	Model           string
	Registry        *llm.Registry
	Now             func() time.Time
	TimestampLayout string
}

// Renderer implements agent.EventSink: assistant text streams to out, while tool
// one-liners, the usage line, and notices go to errw so one-shot stdout carries
// only the model's answer (design §10).
type Renderer struct {
	out             io.Writer
	errw            io.Writer
	color           bool
	verbose         bool
	toolStream      bool
	model           string
	registry        *llm.Registry
	now             func() time.Time
	timestampLayout string

	turnStart         time.Time
	assistantLineOpen bool
	pending           map[string]llm.ToolCall // tool_use id -> call, awaiting its result

	cumInput  int
	cumOutput int
	cumCost   float64
}

// NewRenderer builds a Renderer. A nil Now defaults to time.Now.
func NewRenderer(out, errw io.Writer, opts RenderOptions) *Renderer {
	now := opts.Now
	if now == nil {
		now = time.Now
	}
	return &Renderer{
		out:             out,
		errw:            errw,
		color:           opts.Color,
		verbose:         opts.Verbose,
		toolStream:      opts.ToolStream,
		model:           opts.Model,
		registry:        opts.Registry,
		now:             now,
		timestampLayout: opts.TimestampLayout,
		pending:         make(map[string]llm.ToolCall),
	}
}

// StartTurn records the turn's start instant for the duration in the usage line.
// The driver calls it immediately before agent.RunTurn.
func (r *Renderer) StartTurn() { r.turnStart = r.now() }

// SetModel updates the model used for subsequent usage/cost summaries.
func (r *Renderer) SetModel(model string) { r.model = model }

// SetCumulativeUsage seeds the session totals used by per-turn usage lines.
func (r *Renderer) SetCumulativeUsage(inputTokens, outputTokens int, costUSD float64) {
	r.cumInput = inputTokens
	r.cumOutput = outputTokens
	r.cumCost = costUSD
}

func (r *Renderer) TextDelta(text string) {
	if text == "" {
		return
	}
	io.WriteString(r.out, text)
	r.assistantLineOpen = !strings.HasSuffix(text, "\n")
}

func (r *Renderer) ModelTurnStart(modelTurn, attempt int, _ agent.ContextEstimate) {
	if attempt <= 1 {
		r.dimLine(fmt.Sprintf("[model: turn %d waiting]", modelTurn))
		return
	}
	r.dimLine(fmt.Sprintf("[model: turn %d retry %d waiting]", modelTurn, attempt-1))
}

func (r *Renderer) ToolUseStart(call llm.ToolCall) {
	if !r.toolStream {
		return
	}
	r.dimLine(fmt.Sprintf("[tool-call: %s id=%s]", call.Name, call.ID))
}

func (r *Renderer) ToolUseDelta(_ int, _ string) {}

// ToolStart stashes the call so ToolResult can render name+args+summary on one
// line once the result is known.
func (r *Renderer) ToolStart(call llm.ToolCall) {
	r.pending[call.ID] = call
	r.dimLine(fmt.Sprintf("[tool: %s started%s]", call.Name, formatArgs(call.Input)))
}

func (r *Renderer) ToolResult(result llm.ToolResult) {
	call := r.pending[result.ForID]
	delete(r.pending, result.ForID)

	r.dimLine(ToolResultLine(call, result))

	if r.verbose {
		for _, s := range snippet(result.Text) {
			r.dimLine("  " + s)
		}
	}
}

func (r *Renderer) Notice(msg string) { r.dimLine(msg) }

func (r *Renderer) TurnComplete(usage agent.TurnUsage) {
	r.finishAssistantLine()
	elapsed := r.now().Sub(r.turnStart)

	// Accumulate session totals for the cumulative readout.
	var cost float64
	if r.registry != nil {
		cost, _ = r.registry.Cost(r.model, usage.Usage)
	}
	r.cumInput += usage.Usage.InputTokens
	r.cumOutput += usage.Usage.OutputTokens
	r.cumCost += cost

	r.dimLine(usageLine(r.registry, r.model, usage, elapsed, r.cumInput, r.cumOutput, r.cumCost))
}

// dimLine writes one line to errw, wrapping it in dim ANSI codes when color is
// enabled.
func (r *Renderer) dimLine(s string) {
	r.finishAssistantLine()
	s = r.timestampStatusLine(s)
	if r.color {
		fmt.Fprintf(r.errw, "%s%s%s\n", ansiDim, s, ansiReset)
		return
	}
	fmt.Fprintln(r.errw, s)
}

func (r *Renderer) timestampStatusLine(s string) string {
	if r.timestampLayout == "" || !strings.HasPrefix(s, "[") {
		return s
	}
	return "[" + r.now().Format(r.timestampLayout) + " " + strings.TrimPrefix(s, "[")
}

func (r *Renderer) finishAssistantLine() {
	if !r.assistantLineOpen {
		return
	}
	fmt.Fprintln(r.out)
	r.assistantLineOpen = false
}

// formatArgs renders a tool call's input object as space-prefixed key=value
// pairs in a stable (sorted) order. String values are quoted when they contain
// whitespace; non-scalar values (objects, arrays) are summarized by their JSON
// so the line stays one row.
func formatArgs(input json.RawMessage) string {
	if len(input) == 0 {
		return ""
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(input, &obj); err != nil {
		return ""
	}
	keys := make([]string, 0, len(obj))
	for k := range obj {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var b strings.Builder
	for _, k := range keys {
		fmt.Fprintf(&b, " %s=%s", k, formatValue(obj[k]))
	}
	return b.String()
}

// formatValue renders one JSON value compactly for an args line. Strings with
// whitespace are quoted; long strings are clipped.
func formatValue(raw json.RawMessage) string {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		s = clip(s, 60)
		if strings.ContainsAny(s, " \t\r\n") {
			return fmt.Sprintf("%q", s)
		}
		return s
	}
	return clip(strings.TrimSpace(string(raw)), 60)
}

// resultSummary describes a tool result for the arrow target: an error marker
// for is_error results, else a line count (when multi-line) and byte size.
func resultSummary(result llm.ToolResult) string {
	if result.IsError {
		return "error: " + clip(firstLine(result.Text), 80)
	}
	n := len(result.Text)
	lines := countLines(result.Text)
	size := tools.HumanBytes(n)
	prefix := ""
	if result.Truncated {
		if result.OriginalBytes > 0 {
			prefix = fmt.Sprintf("truncated %s of %s, ", tools.HumanBytes(result.ShownBytes), tools.HumanBytes(result.OriginalBytes))
		} else {
			prefix = "truncated, "
		}
	}
	if lines <= 1 {
		if n == 0 {
			return prefix + "(empty), " + size
		}
		return prefix + size
	}
	return fmt.Sprintf("%s%d lines, %s", prefix, lines, size)
}

// ToolResultLine renders the one-line tool summary used by live output and
// session replay.
func ToolResultLine(call llm.ToolCall, result llm.ToolResult) string {
	return fmt.Sprintf("[%s]%s → %s", call.Name, formatArgs(call.Input), resultSummary(result))
}

// usageLine renders the per-turn summary with cumulative totals (design §10):
//
//	[turn: 3 model turns · 12.4k (15.0k) in / 1.8k (2.0k) out · $0.071 ($0.102) · 4.3s]
//
// Per-turn values are shown first; parenthesised values are cumulative across
// the session. Cumulative cost is omitted for models with no price entry;
// per-turn cost is also omitted when the model has no price entry.
func usageLine(registry *llm.Registry, model string, u agent.TurnUsage, elapsed time.Duration, cumIn, cumOut int, cumCost float64) string {
	var b strings.Builder
	fmt.Fprintf(&b, "[turn: %s · %s (%s) in / %s (%s) out",
		modelTurnPhrase(u.ModelTurns),
		humanTokens(u.Usage.InputTokens), humanTokens(cumIn),
		humanTokens(u.Usage.OutputTokens), humanTokens(cumOut))
	if registry != nil {
		if _, known := registry.Cost(model, u.Usage); known {
			fmt.Fprintf(&b, " · $%.3f ($%.3f)", turnCost(registry, model, u.Usage), cumCost)
		}
	}
	if u.Context.Total > 0 {
		fmt.Fprintf(&b, " · ctx %s/%s", humanTokens(u.Context.Total), humanTokens(u.Context.Window))
		if u.Context.System > 0 || u.Context.Tools > 0 || u.Context.Messages > 0 {
			fmt.Fprintf(&b, " (sys %s tools %s msgs %s)",
				humanTokens(u.Context.System), humanTokens(u.Context.Tools), humanTokens(u.Context.Messages))
		}
	}
	fmt.Fprintf(&b, " · %s]", humanDuration(elapsed))
	return b.String()
}

func modelTurnPhrase(n int) string {
	if n == 1 {
		return "1 model turn"
	}
	return fmt.Sprintf("%d model turns", n)
}

// turnCost returns the USD cost for a single turn's usage. It returns 0 when
// the model is unknown.
func turnCost(registry *llm.Registry, model string, u llm.Usage) float64 {
	usd, _ := registry.Cost(model, u)
	return usd
}

// snippet returns the first snippetLines lines of s for the verbose preview.
func snippet(s string) []string {
	if s == "" {
		return nil
	}
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) > snippetLines {
		lines = lines[:snippetLines]
	}
	return lines
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// countLines counts text lines: a trailing newline does not add an empty line.
func countLines(s string) int {
	if s == "" {
		return 0
	}
	n := strings.Count(s, "\n")
	if !strings.HasSuffix(s, "\n") {
		n++
	}
	return n
}

func clip(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

// humanTokens renders a token count compactly: 12400 -> "12.4k".
func humanTokens(n int) string {
	if n < 1000 {
		return fmt.Sprintf("%d", n)
	}
	return fmt.Sprintf("%.1fk", float64(n)/1000)
}

// humanDuration renders an elapsed turn duration: "4.3s" or "850ms".
func humanDuration(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	return fmt.Sprintf("%.1fs", d.Seconds())
}
