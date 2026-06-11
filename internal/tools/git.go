package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
)

const gitSchema = `{
  "type": "object",
  "properties": {
    "args": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Arguments after \"git\", e.g. [\"status\",\"--porcelain\"]."
    }
  },
  "required": ["args"]
}`

type gitTool struct{}

func (gitTool) Name() string { return "git" }

func (gitTool) Description() string {
	return `Run a git command. Pass arguments as an array, e.g. ["status","--porcelain"]. No shell; no pager.`
}

func (gitTool) Schema() json.RawMessage { return json.RawMessage(gitSchema) }

func (gitTool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var args struct {
		Args []string `json:"args"`
	}
	if err := json.Unmarshal(input, &args); err != nil {
		return "", err
	}
	if len(args.Args) == 0 {
		return "", badArgs("args is required and must be a non-empty array")
	}

	cmd := buildGitCommand(ctx, args.Args)

	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf

	runErr := cmd.Run()
	// Failing to locate or start git is infrastructure failure; a non-zero git
	// exit is a normal result the model should see (design §9.9).
	if runErr != nil {
		if _, ok := runErr.(*exec.ExitError); !ok {
			return "", fmt.Errorf("failed to run git: %w", runErr)
		}
	}

	out := buf.String()
	if len(out) > 0 && out[len(out)-1] != '\n' {
		out += "\n"
	}
	return out + fmt.Sprintf("[exit code: %d]", exitCode(runErr)), nil
}

// buildGitCommand assembles the git invocation without running it: --no-pager
// is injected as the first argument (no interactive pager), and
// GIT_TERMINAL_PROMPT=0 is added to the inherited environment so credential
// prompts fail fast instead of hanging on a missing TTY (design §9.9). Exposing
// the *exec.Cmd is the env-inspection seam tests rely on.
func buildGitCommand(ctx context.Context, userArgs []string) *exec.Cmd {
	argv := append([]string{"--no-pager"}, userArgs...)
	cmd := exec.CommandContext(ctx, "git", argv...)
	cmd.Env = append(os.Environ(), "GIT_TERMINAL_PROMPT=0")
	return cmd
}
