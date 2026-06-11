package tools

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
)

const (
	grepDefaultMaxMatches = 200
	grepMaxFileBytes      = 5 * 1024 * 1024
	grepMaxLineChars      = 300
)

// grepDenylist names directories skipped during a recursive walk (design §9.3).
var grepDenylist = map[string]bool{
	".git": true, "node_modules": true, "vendor": true, "dist": true,
	"build": true, "target": true, ".venv": true, "__pycache__": true,
	".svn": true, ".hg": true,
}

const grepSchema = `{
  "type": "object",
  "properties": {
    "pattern": {"type": "string", "description": "Go (RE2) regular expression."},
    "path": {"type": "string", "description": "File or directory to search (default \".\")."},
    "glob": {"type": "string", "description": "Optional path.Match glob filtering base names."},
    "ignore_case": {"type": "boolean", "description": "Case-insensitive match (prepends (?i))."},
    "max_matches": {"type": "integer", "description": "Maximum matches to return (default 200)."},
    "no_ignore": {"type": "boolean", "description": "Search ignored files too (gitignore filtering is the default inside git repos)."}
  },
  "required": ["pattern"]
}`

type grep struct{}

func (grep) Name() string { return "grep" }

func (grep) Description() string {
	return "Search file contents with a Go (RE2) regular expression. Recurses from a path; respects .gitignore inside git repos; prints path:line:text."
}

func (grep) Schema() json.RawMessage { return json.RawMessage(grepSchema) }

func (grep) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var args struct {
		Pattern    string `json:"pattern"`
		Path       string `json:"path"`
		Glob       string `json:"glob"`
		IgnoreCase bool   `json:"ignore_case"`
		MaxMatches int    `json:"max_matches"`
		NoIgnore   bool   `json:"no_ignore"`
	}
	if err := json.Unmarshal(input, &args); err != nil {
		return "", err
	}
	if args.Pattern == "" {
		return "", badArgs("pattern is required")
	}
	if args.MaxMatches < 0 {
		return "", badArgs("max_matches must be >= 0")
	}

	pat := args.Pattern
	if args.IgnoreCase {
		pat = "(?i)" + pat
	}
	re, err := regexp.Compile(pat)
	if err != nil {
		return "", fmt.Errorf("invalid pattern: %v", err)
	}

	root := args.Path
	if root == "" {
		root = "."
	}
	if args.Glob != "" {
		if _, err := path.Match(args.Glob, ""); err != nil {
			return "", badArgs("invalid glob: %v", err)
		}
	}
	maxMatches := args.MaxMatches
	if maxMatches == 0 {
		maxMatches = grepDefaultMaxMatches
	}

	info, err := os.Stat(root)
	if err != nil {
		return "", err
	}

	var matches []string
	truncated := false

	// emit records one match line, applying the line-length and max-match caps.
	// It returns false once the cap is reached so walking can stop.
	emit := func(relpath string, lineno int, text string) bool {
		if len(matches) >= maxMatches {
			truncated = true
			return false
		}
		if len(text) > grepMaxLineChars {
			text = text[:grepMaxLineChars]
		}
		matches = append(matches, fmt.Sprintf("%s:%d:%s", relpath, lineno, text))
		return true
	}

	if !info.IsDir() {
		if err := grepFile(ctx, root, root, re, emit); err != nil {
			return "", err
		}
	} else {
		listed := false
		if !args.NoIgnore {
			if files, ok := gitListFiles(ctx, root); ok {
				listed = true
				for _, rel := range files {
					if ctx.Err() != nil {
						return "", ctx.Err()
					}
					if args.Glob != "" {
						if ok, _ := path.Match(args.Glob, filepath.Base(rel)); !ok {
							continue
						}
					}
					display := filepath.Join(filepath.Base(root), rel)
					if err := grepFile(ctx, filepath.Join(root, rel), display, re, emit); err != nil {
						return "", err
					}
					if truncated {
						break
					}
				}
			}
		}
		if !listed {
			if err := walkGrep(ctx, root, args.Glob, re, emit, &truncated); err != nil {
				return "", err
			}
		}
	}

	if len(matches) == 0 {
		return "(no matches)", nil
	}
	out := strings.Join(matches, "\n")
	if truncated {
		out += fmt.Sprintf("\n[truncated at %d matches]", maxMatches)
	}
	return out, nil
}

// gitListFiles lists tracked plus untracked-but-not-ignored files under root
// (git grep --untracked semantics), paths relative to root, sorted. ok is
// false when root is not in a git work tree or git is unavailable; the caller
// falls back to the denylist walk. Ignore semantics — nesting, negation,
// global excludes — are git's own, which is the point (spec §9).
func gitListFiles(ctx context.Context, root string) ([]string, bool) {
	cmd := exec.CommandContext(ctx, "git", "-C", root,
		"ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", ".")
	out, err := cmd.Output()
	if err != nil {
		return nil, false
	}
	var files []string
	for _, f := range bytes.Split(out, []byte{0}) {
		if len(f) > 0 {
			files = append(files, string(f))
		}
	}
	slices.Sort(files)
	return files, true
}

// walkGrep is the denylist walk used outside git repos or with no_ignore.
func walkGrep(ctx context.Context, root, glob string, re *regexp.Regexp, emit func(string, int, string) bool, truncated *bool) error {
	walkErr := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if d != nil && d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if d.IsDir() {
			if p != root && grepDenylist[d.Name()] {
				return fs.SkipDir
			}
			return nil
		}
		if !d.Type().IsRegular() {
			return nil
		}
		if glob != "" {
			ok, _ := path.Match(glob, d.Name())
			if !ok {
				return nil
			}
		}
		rel, rerr := filepath.Rel(root, p)
		if rerr != nil {
			rel = p
		}
		rel = filepath.Join(filepath.Base(root), rel)
		if cerr := grepFile(ctx, p, rel, re, emit); cerr != nil {
			return cerr
		}
		if *truncated {
			return errStopWalk
		}
		return nil
	})
	if walkErr != nil && walkErr != errStopWalk {
		return walkErr
	}
	return nil
}

// errStopWalk halts a WalkDir once the match cap is hit; it is filtered out by
// the caller rather than surfaced as a real error.
var errStopWalk = fmt.Errorf("grep: match cap reached")

// grepFile scans one file, skipping files over the size cap and binary files
// (NUL in the first sniff window). displayPath is what appears in output.
func grepFile(ctx context.Context, fsPath, displayPath string, re *regexp.Regexp, emit func(string, int, string) bool) error {
	fi, err := os.Stat(fsPath)
	if err != nil || fi.Size() > grepMaxFileBytes {
		return nil
	}
	f, err := os.Open(fsPath)
	if err != nil {
		return nil
	}
	defer f.Close()

	// Buffer must hold the full sniff window; bufio.NewReader's default 4096-byte
	// buffer would make Peek(binarySniffBytes) return only 4096 bytes and miss a
	// NUL deeper in the head.
	br := bufio.NewReaderSize(f, binarySniffBytes)
	head, _ := br.Peek(binarySniffBytes)
	if bytes.IndexByte(head, 0) >= 0 {
		return nil // binary
	}

	// ReadString tolerates lines longer than any fixed buffer (minified bundles
	// commonly exceed 1MB); bufio.Scanner would stop with ErrTooLong and the
	// file would be silently dropped. Match text is capped per match in emit.
	lineno := 0
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		line, err := br.ReadString('\n')
		if len(line) > 0 || err == nil {
			lineno++
			trimmed := strings.TrimRight(line, "\n")
			trimmed = strings.TrimRight(trimmed, "\r")
			if re.MatchString(trimmed) {
				if !emit(displayPath, lineno, trimmed) {
					return nil
				}
			}
		}
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return nil // read error mid-file: stop scanning this file, keep walking
		}
	}
}
