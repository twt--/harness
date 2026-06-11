package tools

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
)

const gitReadonlySchema = `{
  "type": "object",
  "properties": {
    "args": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Arguments after \"git\"; the first must be a read-only subcommand, e.g. [\"log\",\"--oneline\"]."
    }
  },
  "required": ["args"]
}`

// gitReadonlySubcommands is the git_readonly allowlist. bisect is included
// even though it checks out commits (moves HEAD): it is repository
// exploration, accepted by design.
var gitReadonlySubcommands = []string{"bisect", "blame", "diff", "grep", "log", "show", "status"}

type gitReadonly struct{}

func (gitReadonly) Name() string { return "git_readonly" }

func (gitReadonly) Description() string {
	return `Run a read-only git command: status, log, diff, show, grep, blame, or bisect (bisect checks out commits). Pass arguments as an array starting with the subcommand, e.g. ["log","--oneline"]. No shell; no pager.`
}

func (gitReadonly) Schema() json.RawMessage { return json.RawMessage(gitReadonlySchema) }

func (gitReadonly) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var args struct {
		Args []string `json:"args"`
	}
	if err := json.Unmarshal(input, &args); err != nil {
		return "", err
	}
	if len(args.Args) == 0 {
		return "", badArgs("args is required and must be a non-empty array")
	}
	// The first argument must be a bare allowlisted subcommand. Global git
	// options (-c, -C, --git-dir, --exec-path, --paginate, ...) precede the
	// subcommand, so requiring a non-flag first argument blocks every global
	// option injection.
	sub := args.Args[0]
	if strings.HasPrefix(sub, "-") || !slices.Contains(gitReadonlySubcommands, sub) {
		return "", badArgs("first argument must be one of: %s", strings.Join(gitReadonlySubcommands, ", "))
	}
	// "bisect run <cmd>" executes an arbitrary command per revision, escaping
	// the read-only boundary; the other bisect operations only move HEAD.
	if sub == "bisect" && len(args.Args) > 1 && args.Args[1] == "run" {
		return "", badArgs(`git_readonly does not allow "bisect run" (it executes commands)`)
	}
	// A few subcommand-local flags still write files or launch programs even on
	// read-only subcommands (diff/log/show --output, grep --open-files-in-pager).
	// Reject them so the boundary holds; ordinary flags pass through.
	for _, a := range args.Args[1:] {
		if disallowedReadonlyFlag(a) {
			return "", badArgs("flag %q is not allowed in git_readonly", a)
		}
	}
	return runGitArgs(ctx, args.Args)
}

// disallowedReadonlyFlag reports whether a subcommand-local flag can write a
// file or launch a program, which would break the read-only boundary:
// --output/--output-directory (diff, log, show write to a file) and
// -O/--open-files-in-pager (grep opens matches in a pager/editor).
func disallowedReadonlyFlag(arg string) bool {
	switch {
	case arg == "--output" || strings.HasPrefix(arg, "--output="):
		return true
	case arg == "--output-directory" || strings.HasPrefix(arg, "--output-directory="):
		return true
	case arg == "--open-files-in-pager" || strings.HasPrefix(arg, "--open-files-in-pager="):
		return true
	case arg == "-o" || arg == "-O" || strings.HasPrefix(arg, "-O"):
		return true
	default:
		return false
	}
}
