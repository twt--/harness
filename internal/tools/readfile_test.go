package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func runReadFile(t *testing.T, args map[string]any) (string, error) {
	t.Helper()
	b, err := json.Marshal(args)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	return readFile{}.Run(context.Background(), b)
}

func TestReadFileNumbering(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	if err := os.WriteFile(p, []byte("alpha\nbeta\ngamma\n"), 0644); err != nil {
		t.Fatal(err)
	}
	out, err := runReadFile(t, map[string]any{"path": p})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// cat -n style: right-aligned number, tab, line.
	want := "     1\talpha\n     2\tbeta\n     3\tgamma"
	if out != want {
		t.Errorf("got:\n%q\nwant:\n%q", out, want)
	}
}

func TestReadFileOffsetLimit(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	var b strings.Builder
	for i := 1; i <= 10; i++ {
		fmt.Fprintf(&b, "L%d\n", i)
	}
	if err := os.WriteFile(p, []byte(b.String()), 0644); err != nil {
		t.Fatal(err)
	}
	out, err := runReadFile(t, map[string]any{"path": p, "offset": 3, "limit": 2})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Line numbers reflect the true line position, not the window position.
	want := "     3\tL3\n     4\tL4"
	if out != want {
		t.Errorf("got:\n%q\nwant:\n%q", out, want)
	}
}

func TestReadFileOffsetPastEOF(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	if err := os.WriteFile(p, []byte("a\nb\nc\n"), 0644); err != nil {
		t.Fatal(err)
	}
	_, err := runReadFile(t, map[string]any{"path": p, "offset": 99})
	if err == nil {
		t.Fatal("expected error for offset past EOF")
	}
	if !strings.Contains(err.Error(), "3") {
		t.Errorf("error should state the file's line count (3): %v", err)
	}
}

func TestReadFileMissing(t *testing.T) {
	dir := t.TempDir()
	_, err := runReadFile(t, map[string]any{"path": filepath.Join(dir, "nope.txt")})
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestReadFileDirectory(t *testing.T) {
	dir := t.TempDir()
	_, err := runReadFile(t, map[string]any{"path": dir})
	if err == nil {
		t.Fatal("expected error for directory")
	}
	if !strings.Contains(err.Error(), "use list_dir") {
		t.Errorf("directory error should direct to list_dir: %v", err)
	}
}

func TestReadFileBinary(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "bin")
	if err := os.WriteFile(p, []byte("text\x00more"), 0644); err != nil {
		t.Fatal(err)
	}
	_, err := runReadFile(t, map[string]any{"path": p})
	if err == nil {
		t.Fatal("expected binary rejection")
	}
	if !strings.Contains(err.Error(), "appears to be binary") {
		t.Errorf("binary error text wrong: %v", err)
	}
}

func TestReadFileEmpty(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "empty.txt")
	if err := os.WriteFile(p, nil, 0644); err != nil {
		t.Fatal(err)
	}
	out, err := runReadFile(t, map[string]any{"path": p})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "(empty file)" {
		t.Errorf("empty file marker wrong: %q", out)
	}
}

func TestReadFileMissingPathArg(t *testing.T) {
	_, err := runReadFile(t, map[string]any{})
	if err == nil {
		t.Fatal("expected error for missing path arg")
	}
}

func TestReadFileDefaultLineCap(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "big.txt")
	var b strings.Builder
	for i := 1; i <= 1500; i++ {
		fmt.Fprintf(&b, "line %d\n", i)
	}
	if err := os.WriteFile(p, []byte(b.String()), 0644); err != nil {
		t.Fatal(err)
	}
	out, err := runReadFile(t, map[string]any{"path": p})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	lines := strings.Split(out, "\n")
	if len(lines) != 1000 {
		t.Errorf("default cap should yield 1000 lines, got %d", len(lines))
	}
	if !strings.HasPrefix(lines[0], "     1\tline 1") {
		t.Errorf("first line wrong: %q", lines[0])
	}
}

func TestReadFileUnknownArgsTolerated(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	if err := os.WriteFile(p, []byte("x\n"), 0644); err != nil {
		t.Fatal(err)
	}
	out, err := runReadFile(t, map[string]any{"path": p, "bogus": 1})
	if err != nil {
		t.Fatalf("unknown key should be tolerated: %v", err)
	}
	if out != "     1\tx" {
		t.Errorf("got %q", out)
	}
}
