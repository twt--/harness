package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

// binarySniffBytes is how many leading bytes are scanned for NUL to classify a
// file as binary (design §9.1).
const binarySniffBytes = 8 * 1024

// readFileMaxBytes bounds how much of a very large file is loaded when no
// explicit window is requested (design §9.1: >10MB reads only the first window).
const readFileMaxBytes = 10 * 1024 * 1024

// readFileDefaultLimit is the default number of lines returned (design §9.1).
const readFileDefaultLimit = 1000

const readFileSchema = `{
  "type": "object",
  "properties": {
    "path": {"type": "string", "description": "File path to read."},
    "offset": {"type": "integer", "description": "1-based starting line."},
    "limit": {"type": "integer", "description": "Maximum number of lines (default 1000)."}
  },
  "required": ["path"]
}`

type readFile struct{}

func (readFile) Name() string { return "read_file" }

func (readFile) Description() string {
	return "Read a file from disk. Returns line-numbered content; supports offset/limit for large files."
}

func (readFile) Schema() json.RawMessage { return json.RawMessage(readFileSchema) }

func (readFile) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var args struct {
		Path   string `json:"path"`
		Offset int    `json:"offset"`
		Limit  int    `json:"limit"`
	}
	if err := json.Unmarshal(input, &args); err != nil {
		return "", err
	}
	if args.Path == "" {
		return "", badArgs("path is required")
	}
	if args.Offset < 0 {
		return "", badArgs("offset must be >= 1")
	}
	if args.Limit < 0 {
		return "", badArgs("limit must be >= 0")
	}

	info, err := os.Stat(args.Path)
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		return "", fmt.Errorf("%s is a directory; use list_dir", args.Path)
	}

	windowed := args.Offset > 0 || args.Limit > 0

	f, err := os.Open(args.Path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	// Cap the read for very large files unless an explicit window was given.
	var reader io.Reader = f
	if !windowed && info.Size() > readFileMaxBytes {
		reader = io.LimitReader(f, readFileMaxBytes)
	}
	data, err := io.ReadAll(reader)
	if err != nil {
		return "", err
	}

	sniff := data
	if len(sniff) > binarySniffBytes {
		sniff = sniff[:binarySniffBytes]
	}
	if bytes.IndexByte(sniff, 0) >= 0 {
		return "", fmt.Errorf("%s appears to be binary", args.Path)
	}

	if len(data) == 0 {
		return "(empty file)", nil
	}

	lines := splitLines(data)
	total := len(lines)

	offset := args.Offset
	if offset == 0 {
		offset = 1
	}
	if offset > total {
		return "", fmt.Errorf("offset %d is past end of file (%s has %d lines)", offset, args.Path, total)
	}
	limit := args.Limit
	if limit == 0 {
		limit = readFileDefaultLimit
	}
	end := offset - 1 + limit
	if end > total {
		end = total
	}

	var b strings.Builder
	for i := offset - 1; i < end; i++ {
		if i > offset-1 {
			b.WriteByte('\n')
		}
		fmt.Fprintf(&b, "%6d\t%s", i+1, lines[i])
	}
	return b.String(), nil
}

// splitLines splits file content into logical lines, dropping a single trailing
// newline so a file ending in "\n" does not yield a spurious empty final line.
func splitLines(data []byte) []string {
	s := string(data)
	s = strings.TrimSuffix(s, "\n")
	if s == "" {
		// Content was a single trailing newline only.
		return []string{""}
	}
	return strings.Split(s, "\n")
}
