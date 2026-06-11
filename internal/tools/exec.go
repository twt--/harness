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
	return "Run a program directly with an argv array (no shell). Use when arguments contain quotes, spaces, or newlines; no globbing/pipes/$VAR — use run_command for those. Returns combined stdout+stderr and the exit code."
}

func (execTool) Schema() json.RawMessage { return json.RawMessage(execSchema) }

func (execTool) ReadOnly() bool { return false }

func (execTool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var args struct {
		Argv           []string `json:"argv"`
		Stdin          string   `json:"stdin"`
		Cwd            string   `json:"cwd"`
		TimeoutSeconds int      `json:"timeout_seconds"`
	}
	if err := json.Unmarshal(input, &args); err != nil {
		return "", err
	}
	if len(args.Argv) == 0 {
		return "", badArgs("argv is required and must be a non-empty array")
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

	// Running an arbitrary user-supplied program is exec's documented purpose
	// (design §2 no-sandbox stance, §9.8) — hence the nosemgrep annotation.
	cmd := exec.Command(args.Argv[0], args.Argv[1:]...) // nosemgrep: dangerous-exec-command
	cmd.Dir = args.Cwd
	if args.Stdin != "" {
		cmd.Stdin = strings.NewReader(args.Stdin)
	}

	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf

	out, err := runProcess(ctx, cmd, &buf, args.TimeoutSeconds)
	if err != nil {
		// Start failures (typically a missing binary) are normal tool errors
		// naming the program so the model can correct its call.
		return "", fmt.Errorf("%s: %w", args.Argv[0], err)
	}
	return out, nil
}
