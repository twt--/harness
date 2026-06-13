package tools

import (
	"context"
	"encoding/json"
)

const execSchema = `{
  "type": "object",
  "properties": {
    "argv": {
      "type": "array",
      "items": {"type": "string"},
      "minItems": 1,
      "description": "Program and its arguments, e.g. [\"grep\",\"-n\",\"foo bar\",\"file.txt\"]. argv[0] is resolved via PATH; remaining items are passed literally (no shell, no globbing, no $VAR expansion)."
    },
    "stdin": {"type": "string", "description": "Written to the program's standard input. Omit for no stdin."},
    "cwd": {"type": "string", "description": "Working directory (default: process cwd)."},
    "timeout_seconds": {"type": "integer", "description": "Kill the program after this many seconds (default 120, cap 600)."}
  },
  "required": ["argv"]
}`

// execTool runs a program directly from an argv array, with no shell between
// the model and the process. It exists because shell quoting is the dominant
// failure mode when a model feeds generated content (commit messages, inline
// scripts, JSON) through run_command; argv elements arrive byte-for-byte, so
// there is nothing to escape (design §9.8). Process-group kill, timeout, and
// output formatting are shared with run_command via runProcess.
type execTool struct{}

func (execTool) Name() string { return "exec" }

func (execTool) Description() string {
	return `Run a program directly. Provide a JSON object with an argv array, e.g. {"argv":["grep","-n","foo bar","file.txt"]}. No shell/globbing/pipes/$VAR; use run_command for shell features.`
}

func (execTool) Schema() json.RawMessage { return json.RawMessage(execSchema) }

func (execTool) ReadOnly() bool { return false }

func (execTool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	args, err := decodeExecArgs(input)
	if err != nil {
		return "", err
	}
	if len(args.Args) == 0 {
		return "", badArgs("argv is required and must be a non-empty array")
	}
	program := args.Args[0]
	args.Args = args.Args[1:]
	out, err := runProgram(ctx, program, args, program, false)
	if err != nil {
		// Start failures (typically a missing binary) are normal tool errors
		// naming the program so the model can correct its call.
		return "", err
	}
	return out, nil
}

func decodeExecArgs(input json.RawMessage) (programArgs, error) {
	return decodeProgramArgs(input, "argv")
}
