package tools

import (
	"context"
	"encoding/json"
	"os/exec"
	"strings"
	"testing"
)

func runGitReadonly(t *testing.T, args ...string) (string, error) {
	t.Helper()
	b, err := json.Marshal(map[string]any{"args": args})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	return gitReadonly{}.Run(context.Background(), b)
}

// committedRepo builds a scratch repo with one commit and chdirs into it; the
// tool has no -C escape hatch, so tests drive the target repo via the cwd.
func committedRepo(t *testing.T) string {
	t.Helper()
	dir := scratchRepo(t)
	mustWrite(t, dir+"/hello.txt", "hi\n")
	for _, argv := range [][]string{{"add", "hello.txt"}, {"commit", "-m", "add hello"}} {
		cmd := exec.Command("git", append([]string{"-C", dir}, argv...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", argv, err, out)
		}
	}
	t.Chdir(dir)
	return dir
}

func TestGitReadonlyAllowsReadSubcommands(t *testing.T) {
	gitAvailable(t)
	committedRepo(t)

	status, err := runGitReadonly(t, "status", "--porcelain")
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if !strings.Contains(status, "[exit code: 0]") {
		t.Errorf("status missing exit code marker: %q", status)
	}

	logOut, err := runGitReadonly(t, "log", "--oneline")
	if err != nil {
		t.Fatalf("log: %v", err)
	}
	if !strings.Contains(logOut, "add hello") {
		t.Errorf("log should show the commit subject: %q", logOut)
	}

	// Subcommand-local flags pass through unchanged.
	patch, err := runGitReadonly(t, "log", "-p")
	if err != nil {
		t.Fatalf("log -p: %v", err)
	}
	if !strings.Contains(patch, "+hi") {
		t.Errorf("log -p should include the diff: %q", patch)
	}
}

func TestGitReadonlyRejectsWriteSubcommands(t *testing.T) {
	for _, args := range [][]string{
		{"commit", "-m", "x"},
		{"push"},
		{"checkout", "main"},
		{"add", "."},
		{"reset", "--hard"},
	} {
		out, err := runGitReadonly(t, args...)
		if err == nil {
			t.Errorf("git_readonly %v should be rejected, got %q", args, out)
		}
	}
}

// Global git options precede the subcommand (-c, -C, --exec-path, --paginate,
// --git-dir, ...) and could change behavior or escape the allowlist; the first
// argument must be a bare allowlisted subcommand, so all of these fail.
func TestGitReadonlyRejectsGlobalFlagInjection(t *testing.T) {
	for _, args := range [][]string{
		{"-c", "core.pager=cat", "log"},
		{"--exec-path=/tmp", "status"},
		{"-C", "/tmp", "log"},
		{"-p", "log"},
		{"--paginate", "log"},
		{"--git-dir=/tmp", "status"},
	} {
		out, err := runGitReadonly(t, args...)
		if err == nil {
			t.Errorf("git_readonly %v should be rejected, got %q", args, out)
		}
	}
}

// Some allowlisted subcommands carry flags that break the read-only boundary:
// diff/log/show --output writes a file, grep -O/--open-files-in-pager executes
// a command, and bisect run executes arbitrary commands per revision.
func TestGitReadonlyRejectsWriteAndExecCapableFlags(t *testing.T) {
	for _, args := range [][]string{
		{"diff", "--output=/tmp/pwn"},
		{"log", "--output", "/tmp/pwn"},
		{"show", "--output=/tmp/pwn"},
		{"grep", "-Ovim", "x"},
		{"grep", "-O", "x"},
		{"grep", "--open-files-in-pager=vim", "x"},
		{"grep", "--open-files-in-pager", "x"},
		{"bisect", "run", "sh", "-c", "true"},
	} {
		out, err := runGitReadonly(t, args...)
		if err == nil {
			t.Errorf("git_readonly %v should be rejected, got %q", args, out)
		}
	}
}

func TestGitReadonlyRejectionListsAllowedSubcommands(t *testing.T) {
	_, err := runGitReadonly(t, "commit", "-m", "x")
	if err == nil {
		t.Fatal("expected error")
	}
	for _, sub := range []string{"status", "log", "diff", "show", "grep", "blame", "bisect"} {
		if !strings.Contains(err.Error(), sub) {
			t.Errorf("error should list allowed subcommand %q: %v", sub, err)
		}
	}
}

func TestGitReadonlyMissingOrEmptyArgs(t *testing.T) {
	for _, in := range []string{`{}`, `{"args":[]}`} {
		if _, err := (gitReadonly{}).Run(context.Background(), json.RawMessage(in)); err == nil {
			t.Errorf("input %s: expected error", in)
		}
	}
}
