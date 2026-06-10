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

	"harness/internal/llm"
)

// Tool is one model-callable capability. Schema is hand-written JSON Schema for
// the input object; Run decodes input into its own typed struct and self-validates.
type Tool interface {
	Name() string
	Description() string     // model-facing, one line
	Schema() json.RawMessage // JSON Schema for the input object
	Run(ctx context.Context, input json.RawMessage) (string, error)
}

// Registry is an ordered set of tools. Order is preserved so Specs and the
// model-facing tool list are stable across runs.
type Registry struct {
	order []string
	tools map[string]Tool
}

// RegisterFileTools registers the built-in file tools (read_file, list_dir,
// grep, edit, write_file, apply_patch) on r, in that order. It is the only
// exported path to these tools; their types are unexported by design.
func RegisterFileTools(r *Registry) {
	r.Register(readFile{})
	r.Register(listDir{})
	r.Register(grep{})
	r.Register(edit{})
	r.Register(writeFile{})
	r.Register(applyPatch{})
}

// RegisterExecTools registers the exec tools (run_command, git, web_fetch) on
// r, in that order. It is the only exported path to these tools; their types
// are unexported by design.
func RegisterExecTools(r *Registry) {
	r.Register(runCommand{})
}

// Default returns a Registry preloaded with every built-in tool.
func Default() *Registry {
	r := &Registry{}
	RegisterFileTools(r)
	RegisterExecTools(r)
	return r
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

// Dispatch runs one tool call and always returns a result (design §8.2). It
// recovers panics, maps unknown tools and decode/run errors to is_error result
// strings, and applies the central output cap (design §8.3).
func (r *Registry) Dispatch(ctx context.Context, call llm.ToolCall) (res llm.ToolResult) {
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

	defer func() {
		if rec := recover(); rec != nil {
			log.Printf("tool %q panicked: %v", call.Name, rec)
			res.Text = fmt.Sprintf("error: tool panicked: %v", rec)
			res.IsError = true
		}
	}()

	out, err := t.Run(ctx, input)
	if err != nil {
		if _, bad := err.(*invalidArgsError); bad {
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
