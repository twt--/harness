package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"harness/internal/tools/patch"
)

const applyPatchSchema = `{
  "type": "object",
  "properties": {
    "patch": {"type": "string", "description": "Full unified-diff text. May span multiple files and supports create (--- /dev/null), delete (+++ /dev/null), and rename."}
  },
  "required": ["patch"]
}`

type applyPatch struct{}

func (applyPatch) Name() string { return "apply_patch" }

func (applyPatch) Description() string {
	return "Apply a unified-diff patch. May span multiple files; supports create, delete, and rename."
}

func (applyPatch) Schema() json.RawMessage { return json.RawMessage(applyPatchSchema) }

func (applyPatch) ReadOnly() bool { return false }

func (applyPatch) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var args struct {
		Patch string `json:"patch"`
	}
	if err := json.Unmarshal(input, &args); err != nil {
		return "", err
	}
	if args.Patch == "" {
		return "", badArgs("patch is required")
	}

	files, err := patch.Parse(args.Patch)
	if err != nil {
		return "", err
	}

	res := patch.Apply(files)
	return formatReport(res), nil
}

// formatReport renders the apply result as the tool's success string: one
// "applied:" line listing the written paths and one "rejected:" line per file
// left untouched, with the failure reason so the model can retry just those.
func formatReport(res patch.Result) string {
	var b strings.Builder
	if len(res.Applied) > 0 {
		fmt.Fprintf(&b, "applied: %s\n", strings.Join(res.Applied, ", "))
	}
	for _, r := range res.Rejected {
		fmt.Fprintf(&b, "rejected: %s (%s)\n", r.Path, r.Reason)
	}
	if b.Len() == 0 {
		return "no changes"
	}
	return strings.TrimRight(b.String(), "\n")
}
