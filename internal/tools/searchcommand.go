package tools

import (
	"context"
	"encoding/json"
)

const searchCommandSchema = `{
  "type": "object",
  "properties": {
    "args": {
      "type": "array",
      "items": {"type": "string"},
      "minItems": 1,
      "description": "Arguments passed after the program name, e.g. [\"-R\",\"-n\",\"TODO\",\".\"]. Each item is passed literally; no shell, glob expansion, pipes, or $VAR expansion."
    },
    "stdin": {"type": "string", "description": "Written to the program's standard input. Omit for no stdin."},
    "cwd": {"type": "string", "description": "Working directory (default: process cwd)."},
    "timeout_seconds": {"type": "integer", "description": "Kill the program after this many seconds (default 120, cap 600)."}
  },
  "required": ["args"]
}`

func runSearchCommand(ctx context.Context, input json.RawMessage, displayName, program string) (string, error) {
	args, err := decodeSearchCommandArgs(input)
	if err != nil {
		return "", err
	}
	return runProgram(ctx, program, args, displayName, true)
}

func decodeSearchCommandArgs(input json.RawMessage) (programArgs, error) {
	return decodeProgramArgs(input, "args")
}
