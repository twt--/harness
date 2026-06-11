package sysprompt

import (
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func writeFileForTest(path, content string) error {
	return os.WriteFile(path, []byte(content), 0o644)
}

func gitAvailable(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
}

// scratchRepo initializes a fresh git repo on a known branch with identity set.
func scratchRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	for _, argv := range [][]string{
		{"init", "-q", "-b", "main"},
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

func git(t *testing.T, dir string, argv ...string) {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", dir}, argv...)...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v\n%s", argv, err, out)
	}
}

func write(t *testing.T, path, content string) {
	t.Helper()
	if err := writeFileForTest(path, content); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

var fixedDate = time.Date(2026, 6, 9, 0, 0, 0, 0, time.UTC)

func TestEnvBlockShape(t *testing.T) {
	dir := t.TempDir() // not a git repo
	env := EnvContext(EnvOptions{Dir: dir, Now: func() time.Time { return fixedDate }})

	if !strings.HasPrefix(env, "<env>\n") || !strings.HasSuffix(env, "\n</env>") {
		t.Fatalf("env block not wrapped in <env>...</env>:\n%s", env)
	}
	if !strings.Contains(env, "cwd: "+dir) {
		t.Errorf("env missing cwd line:\n%s", env)
	}
	if !strings.Contains(env, "os: ") {
		t.Errorf("env missing os line:\n%s", env)
	}
	if !strings.Contains(env, "date: 2026-06-09") {
		t.Errorf("env missing date line:\n%s", env)
	}
}

func TestEnvNonRepo(t *testing.T) {
	dir := t.TempDir()
	env := EnvContext(EnvOptions{Dir: dir, Now: func() time.Time { return fixedDate }})
	if !strings.Contains(env, "git: (not a git repository)") {
		t.Errorf("non-repo dir should report not-a-repo:\n%s", env)
	}
}

func TestEnvGitSummary(t *testing.T) {
	gitAvailable(t)
	dir := scratchRepo(t)
	// One committed-then-modified file, one untracked file.
	write(t, dir+"/tracked.txt", "v1\n")
	git(t, dir, "add", "tracked.txt")
	git(t, dir, "commit", "-q", "-m", "init")
	write(t, dir+"/tracked.txt", "v2\n")
	write(t, dir+"/new.txt", "hello\n")

	env := EnvContext(EnvOptions{Dir: dir, Now: func() time.Time { return fixedDate }})

	if !strings.Contains(env, "branch=main") {
		t.Errorf("git line should name the branch:\n%s", env)
	}
	if !strings.Contains(env, "1 modified") {
		t.Errorf("git line should count modified files:\n%s", env)
	}
	if !strings.Contains(env, "1 untracked") {
		t.Errorf("git line should count untracked files:\n%s", env)
	}
}

func TestBuildAppendsByDefault(t *testing.T) {
	dir := t.TempDir()
	out := Build(Options{
		Append: "project note",
		Env:    EnvOptions{Dir: dir, Now: func() time.Time { return fixedDate }},
	})
	if !strings.Contains(out, builtinInstructions) {
		t.Errorf("append mode should keep builtin instructions")
	}
	if !strings.Contains(out, "project note") {
		t.Errorf("append text missing:\n%s", out)
	}
	if !strings.Contains(out, "<env>") {
		t.Errorf("env block should be present by default")
	}
	// builtin then env then append, with the env block intact.
	if !strings.Contains(out, builtinInstructions+"\n\n<env>") {
		t.Errorf("builtin should be followed by the env block:\n%s", out)
	}
}

func TestBuildOverrideReplacesBuiltin(t *testing.T) {
	dir := t.TempDir()
	out := Build(Options{
		Override: "ONLY THESE RULES",
		Env:      EnvOptions{Dir: dir, Now: func() time.Time { return fixedDate }},
	})
	if strings.Contains(out, builtinInstructions) {
		t.Errorf("override should drop builtin instructions:\n%s", out)
	}
	if !strings.Contains(out, "ONLY THESE RULES") {
		t.Errorf("override text missing:\n%s", out)
	}
	if !strings.Contains(out, "<env>") {
		t.Errorf("env block should still be present under override")
	}
}

func TestBuildIncludesAgentsMD(t *testing.T) {
	dir := t.TempDir()
	out := Build(Options{
		AgentsMD: "# Project rules\nAlways write tests.",
		Env:      EnvOptions{Dir: dir, Now: func() time.Time { return fixedDate }},
	})
	if !strings.Contains(out, "# Project rules\nAlways write tests.") {
		t.Errorf("AGENTS.md content should appear in system prompt:\n%s", out)
	}
	// Order: builtin -> env -> agents.md (no append)
	envIdx := strings.Index(out, "<env>")
	agentsIdx := strings.Index(out, "# Project rules")
	if envIdx < 0 || agentsIdx < 0 || envIdx >= agentsIdx {
		t.Errorf("AGENTS.md should come after the env block:\n%s", out)
	}
}

func TestBuildAgentsMDBeforeAppend(t *testing.T) {
	dir := t.TempDir()
	out := Build(Options{
		AgentsMD: "from agents.md",
		Append:   "from -system flag",
		Env:      EnvOptions{Dir: dir, Now: func() time.Time { return fixedDate }},
	})
	agentsIdx := strings.Index(out, "from agents.md")
	appendIdx := strings.Index(out, "from -system flag")
	if agentsIdx < 0 || appendIdx < 0 || agentsIdx >= appendIdx {
		t.Errorf("AGENTS.md should come before -system append:\n%s", out)
	}
}

func TestBuildEmptyAgentsMDOmitted(t *testing.T) {
	dir := t.TempDir()
	out := Build(Options{
		AgentsMD: "",
		Append:   "project note",
		Env:      EnvOptions{Dir: dir, Now: func() time.Time { return fixedDate }},
	})
	// No double blank-line gap beyond the normal separators.
	if strings.Contains(out, "\n\n\n\n") {
		t.Errorf("empty AGENTS.md should not leave extra blank lines:\n%q", out)
	}
	if !strings.Contains(out, "project note") {
		t.Errorf("append text should still be present:\n%s", out)
	}
}

// The mode prompt is the final section: after builtin instructions, env,
// AGENTS.md, and -system append, so a mode layers on top of everything else.
func TestBuildModePromptAppendedLast(t *testing.T) {
	out := Build(Options{
		Append:     "project note",
		AgentsMD:   "agents rules",
		ModePrompt: "mode section",
		NoEnv:      true,
	})
	if !strings.HasSuffix(out, "project note\n\nmode section") {
		t.Errorf("mode prompt should be the final section after append:\n%s", out)
	}
	if !strings.Contains(out, builtinInstructions) {
		t.Errorf("builtin instructions must be kept")
	}
}

func TestBuildEmptyModePromptOmitted(t *testing.T) {
	out := Build(Options{NoEnv: true})
	if out != builtinInstructions {
		t.Errorf("no options should yield just the builtin instructions:\n%s", out)
	}
}

func TestBuildNoEnvDropsEnvBlock(t *testing.T) {
	dir := t.TempDir()
	out := Build(Options{
		NoEnv: true,
		Env:   EnvOptions{Dir: dir, Now: func() time.Time { return fixedDate }},
	})
	if strings.Contains(out, "<env>") {
		t.Errorf("-no-env should drop the env block:\n%s", out)
	}
	if !strings.Contains(out, builtinInstructions) {
		t.Errorf("builtin should remain with -no-env")
	}
	if strings.HasSuffix(out, "\n\n") {
		t.Errorf("no trailing separator should remain when env is dropped: %q", out)
	}
}
