package tools

import (
	"encoding/json"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

func runExec(t *testing.T, args map[string]any) (string, error) {
	return runTool(t, execTool{}, args)
}

func TestExecEchoExitZero(t *testing.T) {
	out, err := runExec(t, map[string]any{"argv": []string{"echo", "hello"}})
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

func TestExecNonZeroExitNotError(t *testing.T) {
	out, err := runExec(t, map[string]any{"argv": []string{"false"}})
	if err != nil {
		t.Fatalf("non-zero exit must not be a tool error: %v", err)
	}
	if !strings.Contains(out, "[exit code: 1]") {
		t.Errorf("missing exit code 1 marker: %q", out)
	}
}

// The tool's whole point: arguments reach the program byte-for-byte, with no
// shell to expand variables or glob patterns.
func TestExecArgsPassedLiterally(t *testing.T) {
	out, err := runExec(t, map[string]any{"argv": []string{"echo", "$HOME", "a*b", "two words"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, want := range []string{"$HOME", "a*b", "two words"} {
		if !strings.Contains(out, want) {
			t.Errorf("argument not passed literally, missing %q: %q", want, out)
		}
	}
}

func TestExecStdinWired(t *testing.T) {
	out, err := runExec(t, map[string]any{"argv": []string{"cat"}, "stdin": "piped input\n"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "piped input") {
		t.Errorf("stdin not wired to program: %q", out)
	}
}

// Without stdin the program must see immediate EOF (the nil-Stdin /dev/null
// default), not hang waiting for input.
func TestExecNoStdinReadsEOF(t *testing.T) {
	out, err := runExec(t, map[string]any{"argv": []string{"cat"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "[exit code: 0]") {
		t.Errorf("cat without stdin should exit 0 on EOF: %q", out)
	}
}

func TestExecCombinedStdoutStderr(t *testing.T) {
	// exec adds no shell of its own; here the program happens to be sh.
	out, err := runExec(t, map[string]any{"argv": []string{"sh", "-c", "echo out; echo err 1>&2"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "out") || !strings.Contains(out, "err") {
		t.Errorf("combined output must contain both streams: %q", out)
	}
}

func TestExecCwdHonored(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "marker.txt"), "x\n")
	out, err := runExec(t, map[string]any{"argv": []string{"ls"}, "cwd": dir})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "marker.txt") {
		t.Errorf("program did not run in cwd %q: %q", dir, out)
	}
}

func TestExecMissingCwd(t *testing.T) {
	_, err := runExec(t, map[string]any{"argv": []string{"echo", "hi"}, "cwd": filepath.Join(t.TempDir(), "does-not-exist")})
	if err == nil {
		t.Fatal("expected error for missing cwd")
	}
}

func TestExecMissingArgv(t *testing.T) {
	_, err := runExec(t, map[string]any{})
	if err == nil {
		t.Fatal("expected error for missing argv")
	}
}

func TestExecEmptyArgv(t *testing.T) {
	_, err := runExec(t, map[string]any{"argv": []string{}})
	if err == nil {
		t.Fatal("expected error for empty argv")
	}
}

func TestDecodeExecArgsAcceptsObjectAndBareArray(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  execArgs
	}{
		{
			name:  "object",
			input: `{"argv":["echo","hello"],"cwd":"/tmp","timeout_seconds":5}`,
			want:  execArgs{Argv: []string{"echo", "hello"}, Cwd: "/tmp", TimeoutSeconds: 5},
		},
		{
			name:  "bare array",
			input: `["echo","hello"]`,
			want:  execArgs{Argv: []string{"echo", "hello"}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := decodeExecArgs(json.RawMessage(tt.input))
			if err != nil {
				t.Fatalf("decodeExecArgs: %v", err)
			}
			if !slices.Equal(got.Argv, tt.want.Argv) || got.Cwd != tt.want.Cwd || got.TimeoutSeconds != tt.want.TimeoutSeconds {
				t.Errorf("decodeExecArgs() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestExecDescriptionSteersToObjectArgv(t *testing.T) {
	desc := execTool{}.Description()
	if !strings.Contains(desc, "JSON object") || !strings.Contains(desc, `{"argv":[`) {
		t.Errorf("description should show object-shaped argv, got %q", desc)
	}
	if strings.Contains(desc, "Run a program directly with an argv array") {
		t.Errorf("description still encourages bare array argv: %q", desc)
	}
}

// A missing binary is a normal, model-correctable tool error naming the
// program — never a panic.
func TestExecProgramNotFound(t *testing.T) {
	_, err := runExec(t, map[string]any{"argv": []string{"definitely-not-a-real-binary-harness-xyz"}})
	if err == nil {
		t.Fatal("expected error for unknown program")
	}
	if !strings.Contains(err.Error(), "definitely-not-a-real-binary-harness-xyz") {
		t.Errorf("error should name the missing program: %v", err)
	}
}

// The timeout test exercises a real subprocess kill (sanctioned exception),
// mirroring TestRunCommandTimeoutKillsGroup.
func TestExecTimeoutKillsGroup(t *testing.T) {
	start := time.Now()
	out, err := runExec(t, map[string]any{
		"argv":            []string{"sh", "-c", "echo started; sleep 30"},
		"timeout_seconds": 1,
	})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("timeout must report a result, not a tool error: %v", err)
	}
	if elapsed > 10*time.Second {
		t.Errorf("program was not killed promptly: took %v", elapsed)
	}
	if !strings.Contains(out, "started") {
		t.Errorf("partial output before kill not reported: %q", out)
	}
	if !strings.Contains(out, "timed out") {
		t.Errorf("timeout should be noted in output: %q", out)
	}
}
