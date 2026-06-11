// Package session persists a conversation transcript to disk and restores it.
// Sessions are saved after every turn, atomically (write <path>.tmp then
// os.Rename), and resumed by Load, which applies the §4 repair rule for a
// transcript saved mid-turn (a dangling tool_use gets a synthesized interrupted
// tool_result). Transcripts are provider-neutral: the JSON uses the internal
// kind/tool_use_id/... tags, never provider wire fields, so a session started
// against one provider resumes against another (design §11).
package session

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"harness/internal/llm"
)

// Version is the on-disk schema version (design §11).
const Version = 1

// Session is the persisted conversation state (design §11).
type Session struct {
	Version  int           `json:"version"`
	Provider string        `json:"provider"`
	Model    string        `json:"model"`
	Created  time.Time     `json:"created"`
	Updated  time.Time     `json:"updated"`
	System   string        `json:"system"`
	Mode     string        `json:"mode,omitempty"`
	Messages []llm.Message `json:"messages"`
	Usage    UsageTotals   `json:"usage"`
}

// UsageTotals is the cumulative token accounting plus dollar cost for a session.
// CostUSD is 0 when the model has no price entry in the registry (design §11).
type UsageTotals struct {
	llm.Usage
	CostUSD float64 `json:"cost_usd"`
}

// Save writes the session to path atomically: it marshals to <path>.tmp, fsyncs
// nothing extra (the rename is the barrier), then os.Renames over path. Parent
// directories are created. On any failure the temp file is removed so no
// half-written .tmp is left behind (design §11).
func (s Session) Save(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("session: create dir: %w", err)
	}

	// Compact (not indented): json.MarshalIndent rewrites embedded
	// json.RawMessage (ToolInput) with injected whitespace, which would not
	// round-trip byte-for-byte. Compact encoding keeps raw tool inputs exact.
	data, err := json.Marshal(s)
	if err != nil {
		return fmt.Errorf("session: marshal: %w", err)
	}

	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("session: write temp: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("session: rename: %w", err)
	}
	return nil
}

// Load reads and decodes a session, then repairs any dangling tool_use left by a
// mid-turn save (design §4): a trailing assistant message whose tool_use blocks
// have no matching results gets a following user message of synthesized
// interrupted tool_result blocks, one per call, in call order. The returned
// transcript satisfies llm.ValidateTranscript.
func Load(path string) (Session, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Session{}, err
	}
	var s Session
	if err := json.Unmarshal(data, &s); err != nil {
		return Session{}, fmt.Errorf("session: decode %s: %w", path, err)
	}
	s.Messages = repair(s.Messages)
	return s, nil
}

// repair applies the §4 dangling-tool_use rule. It is a no-op for a complete
// transcript. Only a trailing assistant message can dangle: a mid-turn save
// stops right after the assistant's tool_use blocks, before any results are
// appended.
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

// DefaultPath returns the auto-save path under the given state dir (design §11):
// <stateDir>/harness/sessions/<timestamp>.json. The state dir is a parameter so
// tests inject a temp dir; main supplies $XDG_STATE_HOME or ~/.local/state. The
// timestamp is second-resolution so two sessions a second apart never collide.
func DefaultPath(stateDir string, at time.Time) string {
	name := at.UTC().Format("20060102T150405Z") + ".json"
	return filepath.Join(stateDir, "harness", "sessions", name)
}
