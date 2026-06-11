package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func runGrep(t *testing.T, args map[string]any) (string, error) {
	t.Helper()
	b, err := json.Marshal(args)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	return grep{}.Run(context.Background(), b)
}

func TestGrepRunsHostGrep(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "a.txt")
	mustWrite(t, p, "hello\nneedle here\n")

	out, err := runGrep(t, map[string]any{"args": []string{"-n", "needle", p}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "2:needle here") {
		t.Errorf("grep output missing match line: %q", out)
	}
	if !strings.Contains(out, "[exit code: 0]") {
		t.Errorf("grep output missing exit code: %q", out)
	}
}

func TestGrepArgsPassedLiterally(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "a.txt")
	mustWrite(t, p, "$HOME\na*b\n")

	out, err := runGrep(t, map[string]any{"args": []string{"-F", "-n", "$HOME", p}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "1:$HOME") {
		t.Errorf("argument should reach grep without shell expansion: %q", out)
	}
	if strings.Contains(out, os.Getenv("HOME")) && os.Getenv("HOME") != "$HOME" {
		t.Errorf("argument appears to have expanded through a shell: %q", out)
	}
}

func TestGrepCwdAndStdin(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "a.txt"), "needle in file\n")

	out, err := runGrep(t, map[string]any{
		"args": []string{"-n", "needle", "a.txt"},
		"cwd":  dir,
	})
	if err != nil {
		t.Fatalf("unexpected cwd error: %v", err)
	}
	if !strings.Contains(out, "1:needle in file") {
		t.Errorf("grep did not run in cwd: %q", out)
	}

	out, err = runGrep(t, map[string]any{
		"args":  []string{"-n", "stdin"},
		"stdin": "stdin match\n",
	})
	if err != nil {
		t.Fatalf("unexpected stdin error: %v", err)
	}
	if !strings.Contains(out, "1:stdin match") {
		t.Errorf("stdin was not passed to grep: %q", out)
	}
}

func TestGrepNonZeroExitNotToolError(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "a.txt")
	mustWrite(t, p, "nothing here\n")

	out, err := runGrep(t, map[string]any{"args": []string{"absent", p}})
	if err != nil {
		t.Fatalf("grep exit 1 must not be a tool error: %v", err)
	}
	if !strings.Contains(out, "[exit code: 1]") {
		t.Errorf("grep no-match should report exit 1: %q", out)
	}
}

func TestGrepValidatesArgs(t *testing.T) {
	if _, err := runGrep(t, map[string]any{}); err == nil {
		t.Fatal("expected error for missing args")
	}
	if _, err := runGrep(t, map[string]any{"args": []string{}}); err == nil {
		t.Fatal("expected error for empty args")
	}
	if _, err := runGrep(t, map[string]any{"args": []string{"x"}, "timeout_seconds": -1}); err == nil {
		t.Fatal("expected error for negative timeout")
	}
}

func TestRipgrepNotRegisteredWhenMissing(t *testing.T) {
	t.Setenv("PATH", t.TempDir())

	if RipgrepAvailable() {
		t.Fatal("rg should not be available from an empty PATH")
	}
	r := &Registry{}
	RegisterFileTools(r)
	if slices.Contains(r.Names(), "rg") {
		t.Errorf("RegisterFileTools registered rg even though it is missing: %v", r.Names())
	}
}

func TestRipgrepRegisteredAndRunsWhenPresent(t *testing.T) {
	dir := t.TempDir()
	makeExecutable(t, filepath.Join(dir, "rg"), `#!/bin/sh
printf 'fake rg:'
for arg in "$@"; do
  printf ' <%s>' "$arg"
done
printf '\n'
`)
	t.Setenv("PATH", dir)

	rg, ok := newRipgrep()
	if !ok {
		t.Fatal("expected fake rg to be found on PATH")
	}
	out, err := rg.Run(context.Background(), json.RawMessage(`{"args":["--json","needle with space"]}`))
	if err != nil {
		t.Fatalf("rg wrapper returned error: %v", err)
	}
	if !strings.Contains(out, "fake rg: <--json> <needle with space>") {
		t.Errorf("rg args not passed literally: %q", out)
	}

	r := &Registry{}
	RegisterFileTools(r)
	names := r.Names()
	grepIndex := slices.Index(names, "grep")
	rgIndex := slices.Index(names, "rg")
	editIndex := slices.Index(names, "edit")
	if rgIndex < 0 {
		t.Fatalf("RegisterFileTools did not include rg: %v", names)
	}
	if !(grepIndex < rgIndex && rgIndex < editIndex) {
		t.Errorf("rg should be registered between grep and edit: %v", names)
	}
}

func makeExecutable(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o755); err != nil {
		t.Fatalf("write executable %s: %v", path, err)
	}
}
