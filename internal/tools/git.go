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
      "minItems": 1,
      "description": "Arguments after \"git\", e.g. [\"status\",\"--porcelain\"]."
    }
  },
  "required": ["args"]
}`

type gitTool struct {
	program string
}

func gitProgram() (string, bool) {
	program, err := exec.LookPath("git")
	if err != nil {
		return "", false
	}
	return program, true
}

func newGitTool() (gitTool, bool) {
	program, ok := gitProgram()
	if !ok {
		return gitTool{}, false
	}
	return gitTool{program: program}, true
}

// GitAvailable reports whether the optional git-backed tools can be registered
// from the current PATH.
func GitAvailable() bool {
	_, ok := gitProgram()
	return ok
}

func (gitTool) Name() string { return "git" }

func (gitTool) Description() string {
	return `Run a git command. Provide a JSON object with an args array, e.g. {"args":["status","--porcelain"]}. No shell; no pager.`
}

func (gitTool) Schema() json.RawMessage { return json.RawMessage(gitSchema) }

func (gitTool) ReadOnly() bool { return false }

func (g gitTool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	args, err := decodeGitArgs(input)
	if err != nil {
		return "", err
	}
	if len(args) == 0 {
		return "", badArgs("args is required and must be a non-empty array")
	}

	return runGitArgs(ctx, g.program, args)
}

func decodeGitArgs(input json.RawMessage) ([]string, error) {
	var bare []string
	if err := json.Unmarshal(input, &bare); err == nil && bare != nil {
		return bare, nil
	}

	var args struct {
		Args []string `json:"args"`
	}
	if err := json.Unmarshal(input, &args); err != nil {
		return nil, err
	}
	return args.Args, nil
}

// runGitArgs executes git with userArgs and formats the combined output plus
// the exit-code marker; shared by git and git_readonly.
func runGitArgs(ctx context.Context, program string, userArgs []string) (string, error) {
	cmd := buildGitCommand(ctx, program, userArgs)

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
func buildGitCommand(ctx context.Context, program string, userArgs []string) *exec.Cmd {
	if program == "" {
		program = "git"
	}
	argv := append([]string{"--no-pager"}, userArgs...)
	cmd := exec.CommandContext(ctx, program, argv...)
	cmd.Env = append(os.Environ(), "GIT_TERMINAL_PROMPT=0")
	return cmd
}
