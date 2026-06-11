// Package tools defines the Tool interface, an ordered registry, and a Dispatch
// entry point that turns every failure mode (unknown tool, invalid arguments,
// tool error, tool panic) into an is_error result and caps oversized output.
// Tools resolve relative paths against the process cwd; there are no path
// restrictions, in keeping with the harness's no-sandbox stance (design §2, §9).
package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"maps"
	"slices"
	"strings"
	"time"

	"harness/internal/llm"
)

// Tool is one model-callable capability. Schema is hand-written JSON Schema for
// the input object; Run decodes input into its own typed struct and self-validates.
type Tool interface {
	Name() string
	Description() string     // model-facing, one line
	Schema() json.RawMessage // JSON Schema for the input object
	// ReadOnly reports that Run never mutates workspace or repo state, so
	// calls may dispatch concurrently with other read-only calls (spec §8).
	ReadOnly() bool
	Run(ctx context.Context, input json.RawMessage) (string, error)
}

// Registry is an ordered set of tools. Order is preserved so Specs and the
// model-facing tool list are stable across runs.
type Registry struct {
	order           []string
	tools           map[string]Tool
	dispatchTimeout time.Duration // zero = defaultDispatchTimeout
}

// defaultDispatchTimeout caps any single tool call: the largest tool
// self-limit (run_command/exec cap timeout_seconds at 600s) plus a one-minute
// grace, so the ceiling never fires first for well-behaved tools. It exists to
// stop tools with no self-limit from hanging the turn (spec §6).
const defaultDispatchTimeout = 11 * time.Minute

// DisabledTool describes an optional built-in tool that was not registered.
type DisabledTool struct {
	Name   string
	Reason string
}

// Message renders a concise user-facing disabled-tool diagnostic.
func (d DisabledTool) Message() string {
	return fmt.Sprintf("Tool %q is disabled. Reason: %s.", d.Name, d.Reason)
}

func missingBinaryTool(name, binary string) DisabledTool {
	return DisabledTool{Name: name, Reason: fmt.Sprintf("%q binary not found", binary)}
}

// SetDispatchTimeout overrides the per-call ceiling applied by Dispatch.
// Non-positive values reset to the default.
func (r *Registry) SetDispatchTimeout(d time.Duration) { r.dispatchTimeout = d }

// RegisterFileTools registers the built-in file tools (read_file, list_dir,
// grep, optional rg, edit, write_file, apply_patch) on r, in that order. It is
// the only exported path to these tools; their types are unexported by design.
func RegisterFileTools(r *Registry) {
	registerFileTools(r, nil)
}

func registerFileTools(r *Registry, disabled *[]DisabledTool) {
	r.Register(readFile{})
	r.Register(listDir{})
	r.Register(grep{})
	if rg, ok := newRipgrep(); ok {
		r.Register(rg)
	} else if disabled != nil {
		*disabled = append(*disabled, missingBinaryTool("rg", "rg"))
	}
	r.Register(edit{})
	r.Register(writeFile{})
	r.Register(applyPatch{})
}

// RegisterExecTools registers the exec tools (run_command, exec, git,
// web_fetch) on r, in that order. It is the only exported path to these tools;
// their types are unexported by design.
func RegisterExecTools(r *Registry) {
	registerExecTools(r, nil)
}

func registerExecTools(r *Registry, disabled *[]DisabledTool) {
	r.Register(runCommand{})
	r.Register(execTool{})
	if git, ok := newGitTool(); ok {
		r.Register(git)
	} else if disabled != nil {
		*disabled = append(*disabled, missingBinaryTool("git", "git"))
	}
	r.Register(webFetch{})
}

// Default returns a Registry preloaded with every built-in tool.
func Default() *Registry {
	r, _ := DefaultWithDiagnostics()
	return r
}

// DefaultWithDiagnostics returns the default tool registry plus diagnostics for
// optional tools that were not registered.
func DefaultWithDiagnostics() (*Registry, []DisabledTool) {
	r := &Registry{}
	var disabled []DisabledTool
	registerFileTools(r, &disabled)
	registerExecTools(r, &disabled)
	return r, disabled
}

// DefaultNames returns the names of the Default tool set in registration
// order. Run-mode definitions use it as the baseline allowed-tool list.
func DefaultNames() []string { return Default().Names() }

// Catalog returns a Registry with every constructible tool: the Default set
// plus the mode-oriented tools (git_readonly, write_tmp_file), which run modes
// select from by name. Build it once per process — write_tmp_file holds the
// per-run temp directory.
func Catalog() *Registry {
	r, _ := CatalogWithDiagnostics()
	return r
}

// CatalogWithDiagnostics returns the complete constructible tool catalog plus
// diagnostics for optional tools that were not registered.
func CatalogWithDiagnostics() (*Registry, []DisabledTool) {
	r, disabled := DefaultWithDiagnostics()
	if git, ok := newGitReadonly(); ok {
		r.Register(git)
	} else {
		disabled = append(disabled, missingBinaryTool("git_readonly", "git"))
	}
	r.Register(newWriteTmpFile())
	return r, disabled
}

// Names returns the registered tool names in registration order.
func (r *Registry) Names() []string {
	return append([]string(nil), r.order...)
}

// Subset returns a new Registry containing exactly the named tools, in this
// registry's order. Unknown names are an error so a config typo fails fast
// instead of silently dropping a tool.
func (r *Registry) Subset(names []string) (*Registry, error) {
	want := make(map[string]bool, len(names))
	for _, name := range names {
		want[name] = true
	}
	sub := &Registry{}
	for _, name := range r.order {
		if want[name] {
			sub.Register(r.tools[name])
			delete(want, name)
		}
	}
	if len(want) > 0 {
		unknown := slices.Sorted(maps.Keys(want))
		return nil, fmt.Errorf("unknown tools: %s (valid tools: %s)",
			strings.Join(unknown, ", "), strings.Join(r.Names(), ", "))
	}
	return sub, nil
}

// Register adds a tool. A later registration with the same name replaces the
// earlier one but keeps its position in the order.
func (r *Registry) Register(t Tool) {
	if r.tools == nil {
		r.tools = make(map[string]Tool)
	}
	name := t.Name()
	if _, ok := r.tools[name]; !ok {
		r.order = append(r.order, name)
	}
	r.tools[name] = t
}

// Specs returns the registered tools' schemas in registration order.
func (r *Registry) Specs() []llm.ToolSchema {
	specs := make([]llm.ToolSchema, 0, len(r.order))
	for _, name := range r.order {
		t := r.tools[name]
		specs = append(specs, llm.ToolSchema{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  t.Schema(),
		})
	}
	return specs
}

// AllReadOnly reports whether every call resolves to a read-only tool.
// Unknown names count as not read-only: they dispatch to an error result,
// and serializing them is the conservative choice.
func (r *Registry) AllReadOnly(calls []llm.ToolCall) bool {
	for _, c := range calls {
		t, ok := r.tools[c.Name]
		if !ok || !t.ReadOnly() {
			return false
		}
	}
	return true
}

// Dispatch runs one tool call and always returns a result (design §8.2). It
// runs Tool.Run under a per-call timeout ceiling in a goroutine, recovers
// panics (inside that goroutine), maps unknown tools and decode/run errors to
// is_error result strings, and applies the central output cap (design §8.3).
// On ceiling expiry it returns a timeout is_error result even for a tool that
// ignores its context; an outer cancellation is reported as cancellation, not
// a timeout (spec §6).
func (r *Registry) Dispatch(parent context.Context, call llm.ToolCall) (res llm.ToolResult) {
	res.ForID = call.ID

	t, ok := r.tools[call.Name]
	if !ok {
		res.Text = fmt.Sprintf("error: unknown tool %q", call.Name)
		res.IsError = true
		return res
	}

	input := call.Input
	if len(input) == 0 {
		input = json.RawMessage("{}")
	}

	timeout := r.dispatchTimeout
	if timeout <= 0 {
		timeout = defaultDispatchTimeout
	}
	ctx, cancel := context.WithTimeout(parent, timeout)
	defer cancel()

	type outcome struct {
		out string
		err error
	}
	done := make(chan outcome, 1) // buffered: an abandoned Run can still send and exit
	go func() {
		defer func() {
			if rec := recover(); rec != nil {
				log.Printf("tool %q panicked: %v", call.Name, rec)
				done <- outcome{err: fmt.Errorf("tool panicked: %v", rec)}
			}
		}()
		out, err := t.Run(ctx, input)
		done <- outcome{out: out, err: err}
	}()

	var out string
	var err error
	select {
	case o := <-done:
		out, err = o.out, o.err
	case <-ctx.Done():
		// The Run goroutine is abandoned (standard cost of a timeout shim);
		// its eventual send lands in the buffered channel and is dropped. The
		// abandoned Run may still mutate external state (write files, leave a
		// subprocess running) after we return — acceptable for the built-in
		// tools, which either finish fast or self-terminate on ctx (exec uses
		// CommandContext, web_fetch caps itself well under the ceiling).
		if parent.Err() != nil {
			res.Text = "error: " + parent.Err().Error()
		} else {
			res.Text = fmt.Sprintf("error: tool timed out after %s", timeout)
		}
		res.IsError = true
		return res
	}

	if err != nil {
		// Report a timeout only when the ceiling itself expired (the derived
		// context's deadline fired) and it was not an outer cancellation. A
		// tool's own internal deadline (e.g. http.Client.Timeout) also yields
		// a DeadlineExceeded error, but with the ceiling unfired it must pass
		// through as a plain tool error — not be relabeled as a dispatch
		// timeout with the wrong duration (spec §6).
		if ctx.Err() == context.DeadlineExceeded && parent.Err() == nil {
			res.Text = fmt.Sprintf("error: tool timed out after %s", timeout)
		} else if _, bad := err.(*invalidArgsError); bad {
			res.Text = "error: invalid arguments: " + err.Error()
		} else if isJSONError(err) {
			res.Text = "error: invalid arguments: " + err.Error()
		} else {
			res.Text = "error: " + err.Error()
		}
		res.IsError = true
		return res
	}

	res.Text = truncate(out)
	return res
}

// invalidArgsError marks a validation failure a tool raises after decoding;
// Dispatch renders it under the "invalid arguments" prefix.
type invalidArgsError struct{ msg string }

func (e *invalidArgsError) Error() string { return e.msg }

func badArgs(format string, a ...any) error {
	return &invalidArgsError{msg: fmt.Sprintf(format, a...)}
}

// isJSONError reports whether err originates from encoding/json decoding, so a
// tool's failed json.Unmarshal surfaces as an "invalid arguments" result.
func isJSONError(err error) bool {
	switch err.(type) {
	case *json.SyntaxError, *json.UnmarshalTypeError:
		return true
	}
	return false
}
