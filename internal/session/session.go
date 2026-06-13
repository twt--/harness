// Package session persists resumable state plus append-only replay/archive
// records. A session path is a directory:
//
//	state.json       compact state used for resume
//	raw.ndjson       user-facing replay events
//	compactions/     raw messages removed from active context
//	artifacts/       full tool outputs omitted from active context
package session

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"harness/internal/llm"
)

// Version is the on-disk schema version.
const Version = 2

const (
	stateFile = "state.json"
	eventLog  = "raw.ndjson"
)

// Session is the compact, resumable conversation state.
type Session struct {
	Version  int           `json:"version"`
	Provider string        `json:"provider"`
	Model    string        `json:"model"`
	Created  time.Time     `json:"created"`
	Updated  time.Time     `json:"updated"`
	System   string        `json:"system"`
	Mode     string        `json:"mode,omitempty"`
	Turn     int           `json:"turn,omitempty"`
	Messages []llm.Message `json:"messages"`
	Usage    UsageTotals   `json:"usage"`
}

// UsageTotals is the cumulative token accounting plus dollar cost for a session.
// CostUSD is 0 when the model has no price entry in the registry.
type UsageTotals struct {
	llm.Usage
	CostUSD float64 `json:"cost_usd"`
}

// Save writes state.json atomically under dir. Parent directories are created,
// and the session directory itself is the stable path printed to the user.
func (s Session) Save(dir string) error {
	if dir == "" {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("session: create dir: %w", err)
	}
	s.Version = Version

	data, err := json.Marshal(s)
	if err != nil {
		return fmt.Errorf("session: marshal: %w", err)
	}

	target := filepath.Join(dir, stateFile)
	tmp := target + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("session: write temp: %w", err)
	}
	if err := os.Rename(tmp, target); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("session: rename: %w", err)
	}
	return nil
}

// Load reads dir/state.json and repairs a dangling trailing tool_use, yielding a
// transcript that can be sent to either provider dialect.
func Load(dir string) (Session, error) {
	data, err := os.ReadFile(filepath.Join(dir, stateFile))
	if err != nil {
		return Session{}, err
	}
	var s Session
	if err := json.Unmarshal(data, &s); err != nil {
		return Session{}, fmt.Errorf("session: decode %s: %w", filepath.Join(dir, stateFile), err)
	}
	s.Messages = repair(s.Messages)
	return s, nil
}

// Event is one append-only replay record. Display carries the exact user-facing
// line for events that the renderer shows as dim one-liners.
type Event struct {
	Time       time.Time       `json:"time,omitempty"`
	Type       string          `json:"type"`
	Turn       int             `json:"turn,omitempty"`
	Text       string          `json:"text,omitempty"`
	Display    string          `json:"display,omitempty"`
	ToolID     string          `json:"tool_id,omitempty"`
	Tool       string          `json:"tool,omitempty"`
	Input      json.RawMessage `json:"input,omitempty"`
	Usage      *llm.Usage      `json:"usage,omitempty"`
	ModelTurns int             `json:"model_turns,omitempty"`
}

const (
	EventUser           = "user"
	EventAssistantDelta = "assistant_delta"
	EventToolStart      = "tool_start"
	EventToolResult     = "tool_result"
	EventNotice         = "notice"
	EventTurnUsage      = "turn_usage"
)

// AppendEvent appends ev as one JSON line to raw.ndjson under dir.
func AppendEvent(dir string, ev Event) error {
	if dir == "" {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("session: create dir: %w", err)
	}
	f, err := os.OpenFile(filepath.Join(dir, eventLog), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("session: open event log: %w", err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	if err := enc.Encode(ev); err != nil {
		return fmt.Errorf("session: append event: %w", err)
	}
	return nil
}

// ReplayOptions controls the plain-text replay renderer.
type ReplayOptions struct {
	IncludeToolOutput bool
}

// Replay prints a user-facing reconstruction of raw.ndjson.
func Replay(dir string, w io.Writer, opts ReplayOptions) error {
	f, err := os.Open(filepath.Join(dir, eventLog))
	if err != nil {
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	assistantLineOpen := false
	finishAssistant := func() {
		if assistantLineOpen {
			fmt.Fprintln(w)
			assistantLineOpen = false
		}
	}

	for sc.Scan() {
		var ev Event
		if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
			return fmt.Errorf("session: replay decode: %w", err)
		}
		switch ev.Type {
		case EventUser:
			finishAssistant()
			fmt.Fprintf(w, "> %s\n", ev.Text)
		case EventAssistantDelta:
			io.WriteString(w, ev.Text)
			assistantLineOpen = !strings.HasSuffix(ev.Text, "\n")
		case EventToolResult, EventNotice, EventTurnUsage:
			finishAssistant()
			if ev.Display != "" {
				fmt.Fprintln(w, ev.Display)
			}
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	finishAssistant()
	return nil
}

// LatestTurnOutput returns the user-visible output recorded for the latest turn,
// excluding the user's prompt. Missing replay logs are treated as empty output so
// callers can use it before the first completed turn.
func LatestTurnOutput(dir string) (string, error) {
	if dir == "" {
		return "", nil
	}
	f, err := os.Open(filepath.Join(dir, eventLog))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", nil
		}
		return "", err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)

	latestTurn := 0
	var b strings.Builder
	assistantLineOpen := false
	finishAssistant := func() {
		if assistantLineOpen {
			b.WriteByte('\n')
		}
		assistantLineOpen = false
	}
	resetForTurn := func(turn int) {
		latestTurn = turn
		b.Reset()
		assistantLineOpen = false
	}

	for sc.Scan() {
		var ev Event
		if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
			return "", fmt.Errorf("session: replay decode: %w", err)
		}
		if ev.Turn == 0 {
			continue
		}
		if ev.Turn > latestTurn || ev.Type == EventUser && ev.Turn == latestTurn {
			resetForTurn(ev.Turn)
		}
		if ev.Turn != latestTurn || ev.Type == EventUser {
			continue
		}
		switch ev.Type {
		case EventAssistantDelta:
			b.WriteString(ev.Text)
			assistantLineOpen = !strings.HasSuffix(ev.Text, "\n")
		case EventToolResult, EventNotice, EventTurnUsage:
			finishAssistant()
			if ev.Display != "" {
				b.WriteString(ev.Display)
				b.WriteByte('\n')
			}
		}
	}
	if err := sc.Err(); err != nil {
		return "", err
	}
	finishAssistant()
	return strings.TrimRight(b.String(), "\n"), nil
}

// Compaction stores the raw messages removed from active context and the summary
// that replaced them.
type Compaction struct {
	Time     time.Time     `json:"time"`
	Summary  string        `json:"summary"`
	Usage    llm.Usage     `json:"usage"`
	Messages []llm.Message `json:"messages"`
}

// SaveCompaction writes one numbered compaction archive and returns the relative
// path to its input JSON file.
func SaveCompaction(dir string, c Compaction) (string, error) {
	if dir == "" {
		return "", nil
	}
	base := filepath.Join(dir, "compactions")
	if err := os.MkdirAll(base, 0o755); err != nil {
		return "", fmt.Errorf("session: create compactions dir: %w", err)
	}
	idx, err := nextIndex(base, ".input.json")
	if err != nil {
		return "", err
	}
	prefix := fmt.Sprintf("%04d", idx)

	inputRel := filepath.Join("compactions", prefix+".input.json")
	inputPath := filepath.Join(dir, inputRel)
	if err := writeJSONAtomic(inputPath, c.Messages); err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(base, prefix+".summary.md"), []byte(c.Summary), 0o644); err != nil {
		return "", fmt.Errorf("session: write compaction summary: %w", err)
	}
	meta := struct {
		Time         time.Time `json:"time"`
		Usage        llm.Usage `json:"usage"`
		MessageCount int       `json:"message_count"`
		Input        string    `json:"input"`
		Summary      string    `json:"summary"`
	}{
		Time:         c.Time,
		Usage:        c.Usage,
		MessageCount: len(c.Messages),
		Input:        inputRel,
		Summary:      filepath.Join("compactions", prefix+".summary.md"),
	}
	if err := writeJSONAtomic(filepath.Join(base, prefix+".meta.json"), meta); err != nil {
		return "", err
	}
	return inputRel, nil
}

// SaveToolResultArtifact writes full output omitted from active context.
func SaveToolResultArtifact(dir string, turn int, result llm.ToolResult) (string, error) {
	if dir == "" || !result.Truncated || result.OriginalText == "" {
		return "", nil
	}
	rel := filepath.Join("artifacts", "tool-results", fmt.Sprintf("%04d-%s.txt", turn, safeName(result.ForID)))
	path := filepath.Join(dir, rel)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", fmt.Errorf("session: create artifact dir: %w", err)
	}
	if err := os.WriteFile(path, []byte(result.OriginalText), 0o644); err != nil {
		return "", fmt.Errorf("session: write tool artifact: %w", err)
	}
	return rel, nil
}

func writeJSONAtomic(path string, v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("session: marshal %s: %w", path, err)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("session: write temp %s: %w", path, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("session: rename %s: %w", path, err)
	}
	return nil
}

func nextIndex(dir, suffix string) (int, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, err
	}
	var nums []int
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, suffix) {
			continue
		}
		var n int
		if _, err := fmt.Sscanf(strings.TrimSuffix(name, suffix), "%d", &n); err == nil {
			nums = append(nums, n)
		}
	}
	sort.Ints(nums)
	if len(nums) == 0 {
		return 1, nil
	}
	return nums[len(nums)-1] + 1, nil
}

func safeName(s string) string {
	if s == "" {
		return "result"
	}
	var b strings.Builder
	for _, r := range s {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	return b.String()
}

// repair applies the dangling-tool_use rule. It is a no-op for a complete
// transcript.
func repair(msgs []llm.Message) []llm.Message {
	if len(msgs) == 0 {
		return msgs
	}
	last := msgs[len(msgs)-1]
	if last.Role != llm.RoleAssistant {
		return msgs
	}

	var results []llm.ContentBlock
	for _, b := range last.Content {
		if b.Kind == llm.BlockToolUse {
			results = append(results, llm.ContentBlock{
				Kind:        llm.BlockToolResult,
				ResultForID: b.ToolUseID,
				ResultText:  "interrupted",
				ResultError: true,
			})
		}
	}
	if len(results) == 0 {
		return msgs
	}
	return append(msgs, llm.Message{Role: llm.RoleUser, Content: results})
}

// DefaultPath returns <stateDir>/harness/sessions/<timestamp>/.
func DefaultPath(stateDir string, at time.Time) string {
	name := at.UTC().Format("20060102T150405Z")
	return filepath.Join(stateDir, "harness", "sessions", name)
}
