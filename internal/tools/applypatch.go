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
    "patch": {"type": "string", "description": "Codex apply_patch text beginning with *** Begin Patch and ending with *** End Patch. Supports *** Add File, *** Delete File, *** Update File, and *** Move to."}
  },
  "required": ["patch"]
}`

type applyPatch struct{}

func (applyPatch) Name() string { return "apply_patch" }

func (applyPatch) Description() string {
	return "Apply a Codex-format patch. Supports add, delete, update, and move."
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

	files, err := patch.ParseCodex(args.Patch)
	if err != nil {
		return "", err
	}

	res := patch.ApplyCodex(files)
	report := formatReport(files, res)
	if len(res.Rejected) > 0 {
		return report, fmt.Errorf("%s", report)
	}
	return report, nil
}

func formatReport(files []patch.FilePatch, res patch.Result) string {
	if len(res.Rejected) > 0 {
		r := res.Rejected[0]
		return fmt.Sprintf("Failed to apply patch to %s: %s", r.Path, r.Reason)
	}
	if len(res.Applied) == 0 {
		return "no changes"
	}

	statusByPath := make(map[string]string, len(files))
	for _, f := range files {
		status := "M"
		switch {
		case f.IsCreate:
			status = "A"
		case f.IsDelete:
			status = "D"
		}
		statusByPath[f.Path()] = status
	}

	var b strings.Builder
	b.WriteString("Success. Updated the following files:\n")
	for _, path := range res.Applied {
		fmt.Fprintf(&b, "%s %s\n", statusByPath[path], path)
	}
	return strings.TrimRight(b.String(), "\n")
}
