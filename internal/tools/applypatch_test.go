package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func runApplyPatch(t *testing.T, patch string) (string, error) {
	t.Helper()
	b, err := json.Marshal(map[string]any{"patch": patch})
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	return applyPatch{}.Run(context.Background(), b)
}

func TestApplyPatchMultiFileOneBad(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "good.txt")
	bad := filepath.Join(dir, "bad.txt")
	mustWrite(t, good, "alpha\nbeta\n")
	mustWrite(t, bad, "actual\n")

	patch := "--- a/" + good + "\n+++ b/" + good + "\n@@ -1,2 +1,2 @@\n alpha\n-beta\n+BETA\n" +
		"--- a/" + bad + "\n+++ b/" + bad + "\n@@ -1,1 +1,1 @@\n-NOT_PRESENT\n+x\n"

	out, err := runApplyPatch(t, patch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "applied:") {
		t.Errorf("report should list applied files: %q", out)
	}
	if !strings.Contains(out, good) {
		t.Errorf("good file should be applied: %q", out)
	}
	if !strings.Contains(out, "rejected:") || !strings.Contains(out, bad) {
		t.Errorf("bad file should be rejected: %q", out)
	}
	if !strings.Contains(out, "hunk 1 of 1 did not match") {
		t.Errorf("rejection should name the failing hunk: %q", out)
	}
	// Good file applied.
	if got, _ := os.ReadFile(good); string(got) != "alpha\nBETA\n" {
		t.Errorf("good file not applied: %q", got)
	}
	// Bad file untouched.
	if got, _ := os.ReadFile(bad); string(got) != "actual\n" {
		t.Errorf("bad file should be untouched: %q", got)
	}
}

func TestApplyPatchCreateWhereExistsRejected(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "exists.txt")
	mustWrite(t, p, "here\n")
	patch := "--- /dev/null\n+++ b/" + p + "\n@@ -0,0 +1,1 @@\n+new\n"

	out, err := runApplyPatch(t, patch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "rejected:") || !strings.Contains(out, p) {
		t.Errorf("create-where-exists should be rejected: %q", out)
	}
	if got, _ := os.ReadFile(p); string(got) != "here\n" {
		t.Errorf("existing file should be untouched: %q", got)
	}
}

func TestApplyPatchAllApplied(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	mustWrite(t, p, "one\ntwo\n")
	patch := "--- a/" + p + "\n+++ b/" + p + "\n@@ -1,2 +1,2 @@\n one\n-two\n+TWO\n"

	out, err := runApplyPatch(t, patch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "applied:") {
		t.Errorf("report should list applied: %q", out)
	}
	if strings.Contains(out, "rejected:") {
		t.Errorf("report should have no rejections: %q", out)
	}
	if got, _ := os.ReadFile(p); string(got) != "one\nTWO\n" {
		t.Errorf("content wrong: %q", got)
	}
}

func TestApplyPatchUnparseableIsError(t *testing.T) {
	// A patch with no file headers cannot be parsed: this is a tool error, not
	// a per-file rejection (design §8.2/§9.6).
	_, err := runApplyPatch(t, "this is not a diff at all\n")
	if err == nil {
		t.Fatal("expected an error for unparseable patch text")
	}
}

func TestApplyPatchMissingArg(t *testing.T) {
	b, _ := json.Marshal(map[string]any{})
	_, err := applyPatch{}.Run(context.Background(), b)
	if err == nil {
		t.Fatal("expected error for missing patch arg")
	}
}

func TestApplyPatchMalformedHunkIsError(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	mustWrite(t, p, "x\n")
	patch := "--- a/" + p + "\n+++ b/" + p + "\n@@ garbage @@\n x\n"
	_, err := runApplyPatch(t, patch)
	if err == nil {
		t.Fatal("expected error for malformed hunk header")
	}
}
