package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"strings"
)

const editSchema = `{
  "type": "object",
  "properties": {
    "path": {"type": "string", "description": "File to edit; must already exist (use write_file to create)."},
    "old_string": {"type": "string", "description": "Exact text to replace, whitespace included."},
    "new_string": {"type": "string", "description": "Replacement text."},
    "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)."}
  },
  "required": ["path", "old_string", "new_string"]
}`

type edit struct{}

func (edit) Name() string { return "edit" }

func (edit) Description() string {
	return "Replace an exact string in a file. old_string must appear exactly once unless replace_all is set."
}

func (edit) Schema() json.RawMessage { return json.RawMessage(editSchema) }

func (edit) ReadOnly() bool { return false }

func (edit) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var args struct {
		Path       string `json:"path"`
		OldString  string `json:"old_string"`
		NewString  string `json:"new_string"`
		ReplaceAll bool   `json:"replace_all"`
	}
	if err := json.Unmarshal(input, &args); err != nil {
		return "", err
	}
	if args.Path == "" {
		return "", badArgs("path is required")
	}
	if args.OldString == "" {
		return "", badArgs("old_string must not be empty; use write_file to create or overwrite a file")
	}
	if args.OldString == args.NewString {
		return "", badArgs("old_string and new_string are identical; nothing to change")
	}

	data, err := os.ReadFile(args.Path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return "", fmt.Errorf("%s does not exist; use write_file to create it", args.Path)
		}
		return "", err
	}

	content := string(data)
	count := strings.Count(content, args.OldString)
	switch {
	case count == 0:
		return "", fmt.Errorf("old_string not found in %s", args.Path)
	case count > 1 && !args.ReplaceAll:
		return "", fmt.Errorf("old_string appears %d times; add surrounding context to make it unique, or set replace_all", count)
	}

	var updated string
	if args.ReplaceAll {
		updated = strings.ReplaceAll(content, args.OldString, args.NewString)
	} else {
		updated = strings.Replace(content, args.OldString, args.NewString, 1)
		count = 1
	}

	info, err := os.Stat(args.Path)
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(args.Path, []byte(updated), info.Mode().Perm()); err != nil {
		return "", err
	}
	return fmt.Sprintf("edited %s (%d replacement(s))", args.Path, count), nil
}
