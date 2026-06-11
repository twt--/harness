package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
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

type searchCommandArgs struct {
	Args           []string `json:"args"`
	Stdin          string   `json:"stdin"`
	Cwd            string   `json:"cwd"`
	TimeoutSeconds int      `json:"timeout_seconds"`
}

func runSearchCommand(ctx context.Context, input json.RawMessage, displayName, program string) (string, error) {
	var args searchCommandArgs
	if err := json.Unmarshal(input, &args); err != nil {
		return "", err
	}
	if len(args.Args) == 0 {
		return "", badArgs("args is required and must be a non-empty array")
	}
	if args.TimeoutSeconds < 0 {
		return "", badArgs("timeout_seconds must be >= 0")
	}
	if args.Cwd != "" {
		info, err := os.Stat(args.Cwd)
		if err != nil {
			return "", err
		}
		if !info.IsDir() {
			return "", fmt.Errorf("cwd %s is not a directory", args.Cwd)
		}
	}

	// These tools intentionally invoke the host search binaries directly; argv
	// elements are user/model supplied and are passed without a shell.
	cmd := exec.Command(program, args.Args...) // nosemgrep: dangerous-exec-command
	cmd.Dir = args.Cwd
	if args.Stdin != "" {
		cmd.Stdin = strings.NewReader(args.Stdin)
	}

	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf

	out, err := runProcess(ctx, cmd, &buf, args.TimeoutSeconds)
	if err != nil {
		return "", fmt.Errorf("%s: %w", displayName, err)
	}
	return out, nil
}
