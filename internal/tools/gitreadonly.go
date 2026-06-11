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
	// option injection; subcommand-local flags after args[0] pass through.
	if sub := args.Args[0]; strings.HasPrefix(sub, "-") || !slices.Contains(gitReadonlySubcommands, sub) {
		return "", badArgs("first argument must be one of: %s", strings.Join(gitReadonlySubcommands, ", "))
	}
	return runGitArgs(ctx, args.Args)
}
