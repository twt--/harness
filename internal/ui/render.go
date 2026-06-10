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
)

// dim/reset are the only ANSI codes used; rendering is legible without color
// (design §2, §10). They are emitted only when RenderOptions.Color is set.
const (
	ansiDim   = "\x1b[2m"
	ansiReset = "\x1b[0m"
)

// snippetLines caps the verbose result preview (design §10: "first ~5 lines").
const snippetLines = 5

// RenderOptions configures a Renderer. Color is decided by the caller (TTY check
// plus NO_COLOR / -no-color); Now is injected so the per-turn duration is
// deterministic in tests (design §10, §13).
type RenderOptions struct {
	Color   bool
	Verbose bool
	Model   string
	Now     func() time.Time
}

// Renderer implements agent.EventSink: assistant text streams to out, while tool
// one-liners, the usage line, and notices go to errw so one-shot stdout carries
// only the model's answer (design §10).
type Renderer struct {
	out     io.Writer
	errw    io.Writer
	color   bool
	verbose bool
	model   string
	now     func() time.Time

	turnStart time.Time
	pending   map[string]llm.ToolCall // tool_use id -> call, awaiting its result
}

// NewRenderer builds a Renderer. A nil Now defaults to time.Now.
func NewRenderer(out, errw io.Writer, opts RenderOptions) *Renderer {
	now := opts.Now
	if now == nil {
		now = time.Now
	}
	return &Renderer{
		out:     out,
		errw:    errw,
		color:   opts.Color,
		verbose: opts.Verbose,
		model:   opts.Model,
		now:     now,
		pending: make(map[string]llm.ToolCall),
	}
}

// StartTurn records the turn's start instant for the duration in the usage line.
// The driver calls it immediately before agent.RunTurn.
func (r *Renderer) StartTurn() { r.turnStart = r.now() }

func (r *Renderer) TextDelta(text string) { io.WriteString(r.out, text) }

// ToolStart stashes the call so ToolResult can render name+args+summary on one
// line once the result is known.
func (r *Renderer) ToolStart(call llm.ToolCall) { r.pending[call.ID] = call }

func (r *Renderer) ToolResult(result llm.ToolResult) {
	call := r.pending[result.ForID]
	delete(r.pending, result.ForID)

	line := fmt.Sprintf("[%s]%s → %s", call.Name, formatArgs(call.Input), resultSummary(result))
	r.dimLine(line)

	if r.verbose {
		for _, s := range snippet(result.Text) {
			r.dimLine("  " + s)
		}
	}
}

func (r *Renderer) Notice(msg string) { r.dimLine(msg) }

func (r *Renderer) TurnComplete(usage agent.TurnUsage) {
	elapsed := r.now().Sub(r.turnStart)
	r.dimLine(usageLine(r.model, usage, elapsed))
}

// dimLine writes one line to errw, wrapping it in dim ANSI codes when color is
// enabled.
func (r *Renderer) dimLine(s string) {
	if r.color {
		fmt.Fprintf(r.errw, "%s%s%s\n", ansiDim, s, ansiReset)
		return
	}
	fmt.Fprintln(r.errw, s)
}

// formatArgs renders a tool call's input object as space-prefixed key=value
// pairs in a stable (sorted) order. String values are quoted when they contain
// spaces; non-scalar values (objects, arrays) are summarized by their JSON so
// the line stays one row.
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
// spaces are quoted; long strings are clipped.
func formatValue(raw json.RawMessage) string {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		s = clip(s, 60)
		if strings.ContainsAny(s, " \t") {
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
	size := humanBytes(n)
	if lines <= 1 {
		if n == 0 {
			return "(empty), " + size
		}
		return size
	}
	return fmt.Sprintf("%d lines, %s", lines, size)
}

// usageLine renders the per-turn summary (design §10):
//
//	[turn: 3 steps · 12.4k in / 1.8k out · $0.071 · 4.3s]
//
// The cost segment is omitted for models with no price entry.
func usageLine(model string, u agent.TurnUsage, elapsed time.Duration) string {
	var b strings.Builder
	fmt.Fprintf(&b, "[turn: %d steps · %s in / %s out", u.Steps, humanTokens(u.Usage.InputTokens), humanTokens(u.Usage.OutputTokens))
	if usd, known := llm.Cost(model, u.Usage); known {
		fmt.Fprintf(&b, " · $%.3f", usd)
	}
	fmt.Fprintf(&b, " · %s]", humanDuration(elapsed))
	return b.String()
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

// humanBytes renders a byte count as a short size: 2150 -> "2.1KB".
func humanBytes(n int) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%dB", n)
	}
	div, exp := int64(unit), 0
	for v := int64(n) / unit; v >= unit; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f%cB", float64(n)/float64(div), "KMGTPE"[exp])
}

// humanDuration renders an elapsed turn duration: "4.3s" or "850ms".
func humanDuration(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	return fmt.Sprintf("%.1fs", d.Seconds())
}
