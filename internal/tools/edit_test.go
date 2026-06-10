package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func runEdit(t *testing.T, args map[string]any) (string, error) {
	t.Helper()
	b, err := json.Marshal(args)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	return edit{}.Run(context.Background(), b)
}

func TestEditSingleReplacement(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	mustWrite(t, p, "alpha beta gamma\n")
	out, err := runEdit(t, map[string]any{"path": p, "old_string": "beta", "new_string": "BETA"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "1 replacement") {
		t.Errorf("success message should report 1 replacement: %q", out)
	}
	got, _ := os.ReadFile(p)
	if string(got) != "alpha BETA gamma\n" {
		t.Errorf("file content wrong: %q", got)
	}
}

func TestEditZeroMatches(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	mustWrite(t, p, "alpha\n")
	_, err := runEdit(t, map[string]any{"path": p, "old_string": "absent", "new_string": "x"})
	if err == nil {
		t.Fatal("expected error for zero matches")
	}
	if !strings.Contains(err.Error(), "not found in") {
		t.Errorf("error text wrong: %v", err)
	}
}

func TestEditMultipleWithoutReplaceAll(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	mustWrite(t, p, "x x x\n")
	_, err := runEdit(t, map[string]any{"path": p, "old_string": "x", "new_string": "y"})
	if err == nil {
		t.Fatal("expected error for multiple matches without replace_all")
	}
	if !strings.Contains(err.Error(), "appears 3 times") {
		t.Errorf("error should report count: %v", err)
	}
	if !strings.Contains(err.Error(), "replace_all") {
		t.Errorf("error should mention replace_all: %v", err)
	}
	// File must be untouched.
	got, _ := os.ReadFile(p)
	if string(got) != "x x x\n" {
		t.Errorf("file should be unchanged: %q", got)
	}
}

func TestEditReplaceAll(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	mustWrite(t, p, "x x x\n")
	out, err := runEdit(t, map[string]any{"path": p, "old_string": "x", "new_string": "y", "replace_all": true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "3 replacement") {
		t.Errorf("should report 3 replacements: %q", out)
	}
	got, _ := os.ReadFile(p)
	if string(got) != "y y y\n" {
		t.Errorf("file content wrong: %q", got)
	}
}

func TestEditEmptyOldString(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	mustWrite(t, p, "alpha\n")
	_, err := runEdit(t, map[string]any{"path": p, "old_string": "", "new_string": "x"})
	if err == nil {
		t.Fatal("expected error for empty old_string")
	}
}

func TestEditOldEqualsNew(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	mustWrite(t, p, "alpha\n")
	_, err := runEdit(t, map[string]any{"path": p, "old_string": "alpha", "new_string": "alpha"})
	if err == nil {
		t.Fatal("expected error when old_string == new_string")
	}
}

func TestEditMissingFile(t *testing.T) {
	dir := t.TempDir()
	_, err := runEdit(t, map[string]any{"path": filepath.Join(dir, "nope.txt"), "old_string": "a", "new_string": "b"})
	if err == nil {
		t.Fatal("expected error for missing file")
	}
	if !strings.Contains(err.Error(), "write_file") {
		t.Errorf("missing file should direct to write_file: %v", err)
	}
}

func TestEditMissingRequiredArgs(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	mustWrite(t, p, "alpha\n")
	// new_string missing is allowed (deletion); path and old_string required.
	if _, err := runEdit(t, map[string]any{"old_string": "a", "new_string": "b"}); err == nil {
		t.Error("expected error for missing path")
	}
}
