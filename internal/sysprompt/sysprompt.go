// Package sysprompt builds the system prompt: the builtin agentic-coding
// instructions plus an environment-context block (cwd, os, date, git summary),
// with composition options for appending, overriding, or dropping the env block
// (design §8.5).
package sysprompt

import (
	"context"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

// builtinInstructions is the default agentic-coding guidance (design §8.5).
const builtinInstructions = `You are a coding agent operating in a real working directory through a set of tools.

Work directly with the tools rather than guessing at file contents or command output:
- Read a file with read_file before editing it; never assume what it contains.
- Prefer edit with enough surrounding context to make old_string unique; use write_file only to create new files or fully replace one.
- Use grep and list_dir to locate code instead of speculating about paths.
- Run builds, tests, and linters with run_command, and use git for version-control operations; read the actual output before deciding what to do next.
- Make the smallest change that satisfies the request, then verify it.

Stop and report once the task is done. Do not keep calling tools after the work is complete.`

// Options controls system-prompt composition (design §8.5 flags). Append and
// Override are mutually relevant: Override replaces the builtin instructions,
// Append adds project notes after them. NoEnv drops the env block entirely.
// AgentsMD carries the contents of an AGENTS.md file discovered in the working
// directory; when non-empty it is appended after the env block (and before
// Append), giving the model project-specific instructions without the user
// having to pass -system explicitly. SkillsCatalog is an optional section
// listing available agent skills for progressive disclosure.
type Options struct {
	Append         string // appended after the builtin (or override) instructions
	Override       string // replaces the builtin instructions when non-empty
	NoEnv          bool   // drop the env-context block
	AgentsMD       string // contents of AGENTS.md from the working directory (optional)
	SkillsCatalog  string // available skills catalog (optional, from skills discovery)
	Env            EnvOptions
}

// Build composes the full system prompt per design §8.5: instructions, then a
// blank-line separator and the env block (unless NoEnv), then any appended text.
func Build(opts Options) string {
	instructions := builtinInstructions
	if opts.Override != "" {
		instructions = opts.Override
	}

	parts := []string{instructions}
	if !opts.NoEnv {
		parts = append(parts, EnvContext(opts.Env))
	}
	if opts.AgentsMD != "" {
		parts = append(parts, opts.AgentsMD)
	}
	if opts.SkillsCatalog != "" {
		parts = append(parts, opts.SkillsCatalog)
	}
	if opts.Append != "" {
		parts = append(parts, opts.Append)
	}
	return strings.Join(parts, "\n\n")
}

// EnvOptions parameterizes the env block for testability: Dir is the working
// directory whose git status is summarized (default process cwd via ""), Now
// supplies the date (default time.Now).
type EnvOptions struct {
	Dir string
	Now func() time.Time
}

// EnvContext renders the environment-context block (design §8.5):
//
//	<env>
//	cwd: /Users/twt/project
//	os: darwin/arm64
//	date: 2026-06-09
//	git: branch=main, 2 modified, 1 untracked
//	</env>
func EnvContext(opts EnvOptions) string {
	dir := opts.Dir
	if dir == "" {
		dir = "."
	}
	now := time.Now
	if opts.Now != nil {
		now = opts.Now
	}

	var b strings.Builder
	b.WriteString("<env>\n")
	fmt.Fprintf(&b, "cwd: %s\n", dir)
	fmt.Fprintf(&b, "os: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Fprintf(&b, "date: %s\n", now().Format("2006-01-02"))
	fmt.Fprintf(&b, "git: %s\n", gitSummary(dir))
	b.WriteString("</env>")
	return b.String()
}

// gitSummary returns the branch and modified/untracked counts, or
// "(not a git repository)" when dir is not in a work tree (design §8.5).
func gitSummary(dir string) string {
	branch, ok := gitBranch(dir)
	if !ok {
		return "(not a git repository)"
	}

	modified, untracked := gitStatusCounts(dir)
	if branch == "" {
		branch = "(detached)"
	}
	return fmt.Sprintf("branch=%s, %d modified, %d untracked", branch, modified, untracked)
}

// gitBranch runs `git branch --show-current`; ok is false when the command
// fails (no git, or not a repository).
func gitBranch(dir string) (string, bool) {
	out, err := runGit(dir, "branch", "--show-current")
	if err != nil {
		return "", false
	}
	return strings.TrimSpace(out), true
}

// gitStatusCounts parses `git status --porcelain`: untracked lines start with
// "??"; everything else is a tracked file with staged and/or unstaged changes,
// counted as modified.
func gitStatusCounts(dir string) (modified, untracked int) {
	out, err := runGit(dir, "status", "--porcelain")
	if err != nil {
		return 0, 0
	}
	for _, line := range strings.Split(out, "\n") {
		if len(line) < 2 {
			continue
		}
		if strings.HasPrefix(line, "??") {
			untracked++
		} else {
			modified++
		}
	}
	return modified, untracked
}

func runGit(dir string, args ...string) (string, error) {
	cmd := exec.CommandContext(context.Background(), "git", append([]string{"-C", dir, "--no-pager"}, args...)...)
	out, err := cmd.Output()
	return string(out), err
}
