package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"syscall"
	"time"
)

const (
	runCommandDefaultTimeout = 120
	runCommandMaxTimeout     = 600
)

const runCommandSchema = `{
  "type": "object",
  "properties": {
    "command": {"type": "string", "description": "Shell command line to execute."},
    "cwd": {"type": "string", "description": "Working directory (default: process cwd)."},
    "timeout_seconds": {"type": "integer", "description": "Kill the command after this many seconds (default 120, cap 600)."}
  },
  "required": ["command"]
}`

type runCommand struct{}

func (runCommand) Name() string { return "run_command" }

func (runCommand) Description() string {
	return "Run a shell command. Returns combined stdout+stderr and the exit code."
}

func (runCommand) Schema() json.RawMessage { return json.RawMessage(runCommandSchema) }

func (runCommand) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var args struct {
		Command        string `json:"command"`
		Cwd            string `json:"cwd"`
		TimeoutSeconds int    `json:"timeout_seconds"`
	}
	if err := json.Unmarshal(input, &args); err != nil {
		return "", err
	}
	if args.Command == "" {
		return "", badArgs("command is required")
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

	timeout := args.TimeoutSeconds
	if timeout == 0 {
		timeout = runCommandDefaultTimeout
	}
	if timeout > runCommandMaxTimeout {
		timeout = runCommandMaxTimeout
	}

	runCtx, cancel := context.WithTimeout(ctx, time.Duration(timeout)*time.Second)
	defer cancel()

	cmd := shellCommand(args.Command)
	cmd.Dir = args.Cwd
	// Own process group so a timeout or cancel kills the whole tree, not just
	// the shell (design §9.7).
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf

	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("failed to start shell: %w", err)
	}

	// Kill the entire process group on timeout or cancellation. A goroutine
	// watches the run context; killing the group (negative pid) reaps children.
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-runCtx.Done():
			killGroup(cmd.Process.Pid)
		case <-done:
		}
	}()

	waitErr := cmd.Wait()

	out := buf.String()
	if len(out) > 0 && out[len(out)-1] != '\n' {
		out += "\n"
	}

	if ctxErr := runCtx.Err(); errors.Is(ctxErr, context.DeadlineExceeded) {
		return out + fmt.Sprintf("[timed out after %ds; process group killed]\n[exit code: -1]", timeout), nil
	} else if errors.Is(ctxErr, context.Canceled) {
		return out + "[cancelled; process group killed]\n[exit code: -1]", nil
	}

	return out + fmt.Sprintf("[exit code: %d]", exitCode(waitErr)), nil
}

// shellCommand builds the *exec.Cmd that runs line under the user's shell.
// Running an arbitrary shell command is run_command's documented purpose
// (design §2 no-sandbox stance, §9.7); the harness is assumed to be launched
// inside an already-sandboxed environment, so there is no command allowlist.
// The shell program name is a static literal in each branch; only the command
// line itself is user-supplied, which is intrinsic to this tool.
func shellCommand(line string) *exec.Cmd {
	if _, err := exec.LookPath("bash"); err == nil {
		// -l makes the login shell pick up the user's PATH/toolchain.
		return exec.Command("bash", "-lc", line)
	}
	return exec.Command("sh", "-c", line)
}

// killGroup sends SIGKILL to the entire process group led by pid. Setpgid made
// the child a group leader, so its pgid equals its pid; the negative target
// signals every process in the group.
func killGroup(pid int) {
	_ = syscall.Kill(-pid, syscall.SIGKILL)
}

// exitCode extracts a process exit code from cmd.Wait's error: 0 on success, the
// process's own code on a normal non-zero exit, or -1 when it was signalled or
// failed for another reason.
func exitCode(err error) int {
	if err == nil {
		return 0
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		return ee.ExitCode()
	}
	return -1
}
