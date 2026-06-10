package tools

import (
	"context"
	"encoding/json"
	"os/exec"
	"strings"
	"testing"
)

// gitAvailable reports whether a git binary is on PATH; tests skip without it.
func gitAvailable(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
}

func runGit(t *testing.T, dir string, args ...string) (string, error) {
	t.Helper()
	in := map[string]any{"args": args}
	if dir != "" {
		// The git tool has no cwd param; drive directory via "-C <dir>".
		in["args"] = append([]string{"-C", dir}, args...)
	}
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	return gitTool{}.Run(context.Background(), b)
}

// scratchRepo initializes a fresh git repo in a temp dir with identity set.
func scratchRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	for _, argv := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "test@example.com"},
		{"config", "user.name", "Test"},
	} {
		cmd := exec.Command("git", append([]string{"-C", dir}, argv...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", argv, err, out)
		}
	}
	return dir
}

func TestGitStatusAddCommitLogRoundTrip(t *testing.T) {
	gitAvailable(t)
	dir := scratchRepo(t)
	mustWrite(t, dir+"/hello.txt", "hi\n")

	status, err := runGit(t, dir, "status", "--porcelain")
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if !strings.Contains(status, "hello.txt") {
		t.Errorf("status should show the untracked file: %q", status)
	}
	if !strings.Contains(status, "[exit code: 0]") {
		t.Errorf("status missing exit code marker: %q", status)
	}

	if _, err := runGit(t, dir, "add", "hello.txt"); err != nil {
		t.Fatalf("add: %v", err)
	}
	if _, err := runGit(t, dir, "commit", "-m", "add hello"); err != nil {
		t.Fatalf("commit: %v", err)
	}

	logOut, err := runGit(t, dir, "log", "--oneline")
	if err != nil {
		t.Fatalf("log: %v", err)
	}
	if !strings.Contains(logOut, "add hello") {
		t.Errorf("log should show the commit subject: %q", logOut)
	}
}

func TestGitNonZeroExitNotError(t *testing.T) {
	gitAvailable(t)
	dir := t.TempDir() // not a repo
	out, err := runGit(t, dir, "status")
	if err != nil {
		t.Fatalf("git's own failure must surface as a result, not a tool error: %v", err)
	}
	if !strings.Contains(strings.ToLower(out), "not a git repository") {
		t.Errorf("git's error message should be surfaced: %q", out)
	}
	if !strings.Contains(out, "[exit code:") {
		t.Errorf("missing exit code marker: %q", out)
	}
}

func TestGitMissingArgs(t *testing.T) {
	_, err := gitTool{}.Run(context.Background(), json.RawMessage(`{}`))
	if err == nil {
		t.Fatal("expected error for missing args")
	}
}

func TestGitEmptyArgs(t *testing.T) {
	_, err := gitTool{}.Run(context.Background(), json.RawMessage(`{"args":[]}`))
	if err == nil {
		t.Fatal("expected error for empty args array")
	}
}

// Env-inspection seam: the command builder must inject --no-pager as the first
// arg and GIT_TERMINAL_PROMPT=0 into the environment, without running git.
func TestGitCommandSeam(t *testing.T) {
	cmd := buildGitCommand(context.Background(), []string{"status", "--porcelain"})

	// --no-pager injected immediately after the program name.
	if len(cmd.Args) < 2 || cmd.Args[1] != "--no-pager" {
		t.Errorf("--no-pager not injected as first arg: %v", cmd.Args)
	}
	if cmd.Args[len(cmd.Args)-2] != "status" || cmd.Args[len(cmd.Args)-1] != "--porcelain" {
		t.Errorf("user args not preserved in order: %v", cmd.Args)
	}

	found := false
	for _, kv := range cmd.Env {
		if kv == "GIT_TERMINAL_PROMPT=0" {
			found = true
		}
	}
	if !found {
		t.Errorf("GIT_TERMINAL_PROMPT=0 not set in env: %v", cmd.Env)
	}
}
