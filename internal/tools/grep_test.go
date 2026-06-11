package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
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

func TestGrepBasicMatch(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "a.txt"), "hello\nfunc main\nworld\n")
	out, err := runGrep(t, map[string]any{"pattern": "func main", "path": dir})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// path:line:text
	if !strings.Contains(out, "a.txt:2:func main") {
		t.Errorf("expected path:line:text format: %q", out)
	}
}

func TestGrepIgnoreCase(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "a.txt"), "Hello World\n")
	out, err := runGrep(t, map[string]any{"pattern": "hello", "path": dir, "ignore_case": true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "a.txt:1:Hello World") {
		t.Errorf("ignore_case match missing: %q", out)
	}
	// Without ignore_case, lowercase pattern should not match.
	out2, _ := runGrep(t, map[string]any{"pattern": "hello", "path": dir})
	if out2 != "(no matches)" {
		t.Errorf("case-sensitive should not match: %q", out2)
	}
}

func TestGrepGlob(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "a.go"), "target\n")
	mustWrite(t, filepath.Join(dir, "b.txt"), "target\n")
	out, err := runGrep(t, map[string]any{"pattern": "target", "path": dir, "glob": "*.go"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Contains(out, "b.txt") {
		t.Errorf("glob should exclude b.txt: %q", out)
	}
	if !strings.Contains(out, "a.go") {
		t.Errorf("glob should include a.go: %q", out)
	}
}

func TestGrepMaxMatches(t *testing.T) {
	dir := t.TempDir()
	var b strings.Builder
	for i := 0; i < 50; i++ {
		fmt.Fprintf(&b, "match %d\n", i)
	}
	mustWrite(t, filepath.Join(dir, "a.txt"), b.String())
	out, err := runGrep(t, map[string]any{"pattern": "match", "path": dir, "max_matches": 10})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "[truncated at 10 matches]") {
		t.Errorf("expected max-matches marker: %q", out)
	}
	matchLines := 0
	for _, ln := range strings.Split(out, "\n") {
		if strings.HasPrefix(ln, "[") {
			continue
		}
		matchLines++
	}
	if matchLines != 10 {
		t.Errorf("want 10 match lines, got %d", matchLines)
	}
}

func TestGrepDenylistDirsSkipped(t *testing.T) {
	dir := t.TempDir()
	for _, d := range []string{".git", "node_modules"} {
		sub := filepath.Join(dir, d)
		if err := os.MkdirAll(sub, 0755); err != nil {
			t.Fatal(err)
		}
		mustWrite(t, filepath.Join(sub, "x.txt"), "needle\n")
	}
	mustWrite(t, filepath.Join(dir, "keep.txt"), "needle\n")
	out, err := runGrep(t, map[string]any{"pattern": "needle", "path": dir})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Contains(out, ".git") || strings.Contains(out, "node_modules") {
		t.Errorf("denylisted dirs should be skipped: %q", out)
	}
	if !strings.Contains(out, "keep.txt") {
		t.Errorf("non-denylisted file should match: %q", out)
	}
}

func TestGrepBinarySkipped(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "bin"), "needle\x00needle\n")
	mustWrite(t, filepath.Join(dir, "text.txt"), "needle\n")
	out, err := runGrep(t, map[string]any{"pattern": "needle", "path": dir})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Contains(out, "bin:") {
		t.Errorf("binary file should be skipped: %q", out)
	}
	if !strings.Contains(out, "text.txt") {
		t.Errorf("text file should match: %q", out)
	}
}

// Regression: the NUL sniff must scan the full 8KB head (design §9.1), not just
// the first 4KB. A binary file whose first NUL is at ~byte 6300 lies past
// bufio.Reader's default 4096-byte buffer, so Peek(8192) returns only 4096 bytes
// and the file would be scanned/emitted as text (review issue: grep.go grepFile).
func TestGrepBinaryDeepNULSkipped(t *testing.T) {
	dir := t.TempDir()
	deep := strings.Repeat("a", 6300) + "needle\x00needle" + strings.Repeat("a", 100)
	mustWrite(t, filepath.Join(dir, "deep.bin"), deep+"\n")
	mustWrite(t, filepath.Join(dir, "text.txt"), "needle\n")
	out, err := runGrep(t, map[string]any{"pattern": "needle", "path": dir})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Contains(out, "deep.bin:") {
		t.Errorf("binary file with deep NUL should be skipped: %q", out)
	}
	if !strings.Contains(out, "text.txt") {
		t.Errorf("text file should match: %q", out)
	}
}

func TestGrepSingleFile(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "only.txt")
	mustWrite(t, p, "one\ntwo needle\nthree\n")
	out, err := runGrep(t, map[string]any{"pattern": "needle", "path": p})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "only.txt:2:two needle") {
		t.Errorf("single-file path should work: %q", out)
	}
}

func TestGrepInvalidPattern(t *testing.T) {
	dir := t.TempDir()
	_, err := runGrep(t, map[string]any{"pattern": "(unclosed", "path": dir})
	if err == nil {
		t.Fatal("expected invalid pattern error")
	}
	if !strings.Contains(err.Error(), "invalid pattern:") {
		t.Errorf("error text wrong: %v", err)
	}
}

func TestGrepLongLineCap(t *testing.T) {
	dir := t.TempDir()
	long := "needle" + strings.Repeat("y", 400)
	mustWrite(t, filepath.Join(dir, "a.txt"), long+"\n")
	out, err := runGrep(t, map[string]any{"pattern": "needle", "path": dir})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// The line text after "path:line:" should be capped to 300 chars.
	idx := strings.Index(out, ":1:")
	if idx < 0 {
		t.Fatalf("no match line: %q", out)
	}
	text := strings.TrimRight(out[idx+3:], "\n")
	if len(text) > 300 {
		t.Errorf("line text not capped to 300: got %d chars", len(text))
	}
}

func TestGrepNoMatches(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "a.txt"), "nothing here\n")
	out, err := runGrep(t, map[string]any{"pattern": "absent", "path": dir})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "(no matches)" {
		t.Errorf("want (no matches), got %q", out)
	}
}

func TestGrepMissingPatternArg(t *testing.T) {
	_, err := runGrep(t, map[string]any{"path": "."})
	if err == nil {
		t.Fatal("expected error for missing pattern")
	}
}

// Regression: a file under the 5MB cap with a single line longer than the
// scanner's 1MB token limit must still be matched, not silently dropped when
// bufio.Scanner stops with ErrTooLong (review issue: grep.go grepFile).
func TestGrepLongSingleLineNotDropped(t *testing.T) {
	dir := t.TempDir()
	// 2MB single line containing the pattern, well over the 1MB scan token.
	long := strings.Repeat("a", 2*1024*1024) + "needle"
	mustWrite(t, filepath.Join(dir, "big.txt"), long+"\n")
	out, err := runGrep(t, map[string]any{"pattern": "needle", "path": dir})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out == "(no matches)" {
		t.Fatalf("long line silently dropped: got %q", out)
	}
	if !strings.Contains(out, "big.txt:1:") {
		t.Errorf("expected match on big.txt line 1: %q", out)
	}
}

// initGitRepo turns dir into a git repo; skips the test when git is missing.
func initGitRepo(t *testing.T, dir string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "t@example.com"},
		{"config", "user.name", "t"},
	} {
		cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
}

// mkWrite writes content to path, creating parent directories first. It wraps
// mustWrite (which does not mkdir) so the git tests can populate subdirs.
func mkWrite(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	mustWrite(t, path, content)
}

func TestGrepRespectsGitignore(t *testing.T) {
	dir := t.TempDir()
	initGitRepo(t, dir)
	mkWrite(t, filepath.Join(dir, ".gitignore"), "secret.txt\nbuildout/\n")
	mkWrite(t, filepath.Join(dir, "kept.txt"), "needle here\n")
	mkWrite(t, filepath.Join(dir, "secret.txt"), "needle here\n")
	mkWrite(t, filepath.Join(dir, "buildout", "gen.txt"), "needle here\n")
	mkWrite(t, filepath.Join(dir, "sub", ".gitignore"), "local.txt\n")
	mkWrite(t, filepath.Join(dir, "sub", "local.txt"), "needle here\n")
	mkWrite(t, filepath.Join(dir, "sub", "kept.txt"), "needle here\n")

	out, err := grep{}.Run(context.Background(),
		json.RawMessage(`{"pattern":"needle","path":"`+dir+`"}`))
	if err != nil {
		t.Fatalf("grep: %v", err)
	}
	for _, want := range []string{"kept.txt:1:needle here", "sub/kept.txt:1:"} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
	for _, banned := range []string{"secret.txt", "buildout", "local.txt"} {
		if strings.Contains(out, banned) {
			t.Errorf("ignored file %q leaked into output:\n%s", banned, out)
		}
	}
}

func TestGrepNoIgnoreSearchesEverything(t *testing.T) {
	dir := t.TempDir()
	initGitRepo(t, dir)
	mkWrite(t, filepath.Join(dir, ".gitignore"), "secret.txt\n")
	mkWrite(t, filepath.Join(dir, "secret.txt"), "needle here\n")

	out, err := grep{}.Run(context.Background(),
		json.RawMessage(`{"pattern":"needle","path":"`+dir+`","no_ignore":true}`))
	if err != nil {
		t.Fatalf("grep: %v", err)
	}
	if !strings.Contains(out, "secret.txt:1:needle here") {
		t.Errorf("no_ignore should search ignored files:\n%s", out)
	}
}

func TestGrepNonRepoFallsBackToDenylist(t *testing.T) {
	dir := t.TempDir() // no git repo here
	mkWrite(t, filepath.Join(dir, "a.txt"), "needle here\n")
	mkWrite(t, filepath.Join(dir, "node_modules", "dep.js"), "needle here\n")

	out, err := grep{}.Run(context.Background(),
		json.RawMessage(`{"pattern":"needle","path":"`+dir+`"}`))
	if err != nil {
		t.Fatalf("grep: %v", err)
	}
	if !strings.Contains(out, "a.txt:1:needle here") {
		t.Errorf("missing a.txt match:\n%s", out)
	}
	if strings.Contains(out, "node_modules") {
		t.Errorf("denylist not applied in fallback:\n%s", out)
	}
}

func TestGrepGitListingHonorsGlob(t *testing.T) {
	dir := t.TempDir()
	initGitRepo(t, dir)
	mkWrite(t, filepath.Join(dir, "a.go"), "needle\n")
	mkWrite(t, filepath.Join(dir, "a.txt"), "needle\n")

	out, err := grep{}.Run(context.Background(),
		json.RawMessage(`{"pattern":"needle","path":"`+dir+`","glob":"*.go"}`))
	if err != nil {
		t.Fatalf("grep: %v", err)
	}
	if !strings.Contains(out, "a.go:1:needle") || strings.Contains(out, "a.txt") {
		t.Errorf("glob filter wrong:\n%s", out)
	}
}
