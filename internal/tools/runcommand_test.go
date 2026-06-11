package tools

import (
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func runRunCommand(t *testing.T, args map[string]any) (string, error) {
	return runTool(t, runCommand{}, args)
}

func TestRunCommandEchoExitZero(t *testing.T) {
	out, err := runRunCommand(t, map[string]any{"command": "echo hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "hello") {
		t.Errorf("missing echoed output: %q", out)
	}
	if !strings.Contains(out, "[exit code: 0]") {
		t.Errorf("missing exit code marker: %q", out)
	}
}

func TestRunCommandNonZeroExitNotError(t *testing.T) {
	out, err := runRunCommand(t, map[string]any{"command": "exit 1"})
	if err != nil {
		t.Fatalf("non-zero exit must not be a tool error: %v", err)
	}
	if !strings.Contains(out, "[exit code: 1]") {
		t.Errorf("missing exit code 1 marker: %q", out)
	}
}

func TestRunCommandCombinedStdoutStderr(t *testing.T) {
	// Interleaved writes to both streams must appear in one buffer.
	out, err := runRunCommand(t, map[string]any{"command": "echo out; echo err 1>&2"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "out") || !strings.Contains(out, "err") {
		t.Errorf("combined output must contain both streams: %q", out)
	}
}

func TestRunCommandCwdHonored(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "marker.txt"), "x\n")
	out, err := runRunCommand(t, map[string]any{"command": "ls", "cwd": dir})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "marker.txt") {
		t.Errorf("command did not run in cwd %q: %q", dir, out)
	}
}

func TestRunCommandMissingCwd(t *testing.T) {
	_, err := runRunCommand(t, map[string]any{"command": "echo hi", "cwd": filepath.Join(t.TempDir(), "does-not-exist")})
	if err == nil {
		t.Fatal("expected error for missing cwd")
	}
}

func TestRunCommandStdinWired(t *testing.T) {
	out, err := runRunCommand(t, map[string]any{"command": "cat", "stdin": "hello stdin\n"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "hello stdin") {
		t.Errorf("stdin not wired to command: %q", out)
	}
	if !strings.Contains(out, "[exit code: 0]") {
		t.Errorf("missing exit code marker: %q", out)
	}
}

func TestRunCommandMissingCommand(t *testing.T) {
	_, err := runRunCommand(t, map[string]any{})
	if err == nil {
		t.Fatal("expected error for missing command")
	}
}

// The timeout test exercises a real subprocess kill (sanctioned exception).
// A sleeping child in its own process group must be killed when the timeout
// fires, and the partial output captured before the kill must be reported.
func TestRunCommandTimeoutKillsGroup(t *testing.T) {
	start := time.Now()
	out, err := runRunCommand(t, map[string]any{
		"command":         "echo started; sleep 30",
		"timeout_seconds": 1,
	})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("timeout must report a result, not a tool error: %v", err)
	}
	// Must not have waited anywhere near the 30s sleep.
	if elapsed > 10*time.Second {
		t.Errorf("command was not killed promptly: took %v", elapsed)
	}
	if !strings.Contains(out, "started") {
		t.Errorf("partial output before kill not reported: %q", out)
	}
	if !strings.Contains(out, "timed out") {
		t.Errorf("timeout should be noted in output: %q", out)
	}
}
