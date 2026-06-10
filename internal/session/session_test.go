package session

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"harness/internal/llm"
)

// sampleSession builds a valid session whose transcript contains a complete
// tool_use/tool_result pair, so ValidateTranscript passes before any mutation.
func sampleSession() Session {
	created := time.Date(2026, 6, 9, 10, 0, 0, 0, time.UTC)
	return Session{
		Version:  1,
		Provider: "anthropic",
		Model:    "claude-opus-4-8",
		Created:  created,
		Updated:  created.Add(2 * time.Minute),
		System:   "be terse",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{
				{Kind: llm.BlockText, Text: "list the dir"},
			}},
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				{Kind: llm.BlockText, Text: "sure"},
				{Kind: llm.BlockToolUse, ToolUseID: "call_1", ToolName: "list_dir", ToolInput: json.RawMessage(`{"path":"."}`)},
			}},
			{Role: llm.RoleUser, Content: []llm.ContentBlock{
				{Kind: llm.BlockToolResult, ResultForID: "call_1", ResultText: "main.go"},
			}},
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				{Kind: llm.BlockText, Text: "done"},
			}},
		},
		Usage: UsageTotals{
			Usage:   llm.Usage{InputTokens: 1200, OutputTokens: 340, CacheReadTokens: 800, CacheWriteTokens: 0},
			CostUSD: 0.0123,
		},
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	s := sampleSession()
	if err := llm.ValidateTranscript(s.Messages); err != nil {
		t.Fatalf("sample transcript invalid: %v", err)
	}

	path := filepath.Join(t.TempDir(), "session.json")
	if err := s.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	got, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if err := llm.ValidateTranscript(got.Messages); err != nil {
		t.Fatalf("loaded transcript invalid: %v", err)
	}
	if !reflect.DeepEqual(s, got) {
		t.Fatalf("round-trip mismatch:\n want %+v\n  got %+v", s, got)
	}
}

// A second save over the same path (the after-every-turn case) round-trips too.
func TestSaveLoadSaveRoundTrip(t *testing.T) {
	s := sampleSession()
	path := filepath.Join(t.TempDir(), "session.json")
	if err := s.Save(path); err != nil {
		t.Fatalf("first Save: %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	loaded.Updated = loaded.Updated.Add(time.Minute)
	if err := loaded.Save(path); err != nil {
		t.Fatalf("second Save: %v", err)
	}
	again, err := Load(path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if !reflect.DeepEqual(loaded, again) {
		t.Fatalf("save->load->save mismatch:\n want %+v\n  got %+v", loaded, again)
	}
}

func TestSaveLeavesNoTmpFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "session.json")
	if err := sampleSession().Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tmp") {
			t.Fatalf("temp file left behind: %s", e.Name())
		}
	}
	if len(entries) != 1 {
		t.Fatalf("expected exactly one file after save, got %d: %v", len(entries), entries)
	}
}

// Save creates parent directories so DefaultPath's nested sessions dir works.
func TestSaveCreatesParentDirs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "a", "b", "session.json")
	if err := sampleSession().Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("session not written: %v", err)
	}
}

// A transcript saved mid-turn ends with an assistant tool_use that has no
// matching result. Loading must repair it by synthesizing an interrupted result,
// yielding a transcript that passes ValidateTranscript.
func TestLoadRepairsDanglingToolUse(t *testing.T) {
	dangling := Session{
		Version:  1,
		Provider: "anthropic",
		Model:    "claude-opus-4-8",
		Created:  time.Date(2026, 6, 9, 10, 0, 0, 0, time.UTC),
		Updated:  time.Date(2026, 6, 9, 10, 1, 0, 0, time.UTC),
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{
				{Kind: llm.BlockText, Text: "edit the file"},
			}},
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				{Kind: llm.BlockToolUse, ToolUseID: "call_x", ToolName: "edit", ToolInput: json.RawMessage(`{}`)},
				{Kind: llm.BlockToolUse, ToolUseID: "call_y", ToolName: "edit", ToolInput: json.RawMessage(`{}`)},
			}},
		},
	}
	// Validate the pre-repair transcript IS dangling (the bug we are fixing).
	if err := llm.ValidateTranscript(dangling.Messages); err == nil {
		t.Fatalf("expected dangling transcript to be invalid before repair")
	}

	path := filepath.Join(t.TempDir(), "session.json")
	if err := dangling.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	got, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if err := llm.ValidateTranscript(got.Messages); err != nil {
		t.Fatalf("repaired transcript invalid: %v", err)
	}

	// The repair appends one user message carrying interrupted results, in call
	// order, for every dangling tool_use.
	last := got.Messages[len(got.Messages)-1]
	if last.Role != llm.RoleUser {
		t.Fatalf("repair message role %q, want user", last.Role)
	}
	if len(last.Content) != 2 {
		t.Fatalf("repair carried %d results, want 2", len(last.Content))
	}
	for i, want := range []string{"call_x", "call_y"} {
		b := last.Content[i]
		if b.Kind != llm.BlockToolResult {
			t.Fatalf("block %d kind %q, want tool_result", i, b.Kind)
		}
		if b.ResultForID != want {
			t.Fatalf("block %d result_for_id %q, want %q", i, b.ResultForID, want)
		}
		if !b.ResultError {
			t.Fatalf("block %d result_error false, want true", i)
		}
		if b.ResultText != "interrupted" {
			t.Fatalf("block %d result_text %q, want \"interrupted\"", i, b.ResultText)
		}
	}
}

// A complete transcript is loaded unchanged (no spurious repair message).
func TestLoadDoesNotRepairCompleteTranscript(t *testing.T) {
	s := sampleSession()
	path := filepath.Join(t.TempDir(), "session.json")
	if err := s.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	got, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(got.Messages) != len(s.Messages) {
		t.Fatalf("message count changed: %d -> %d (spurious repair?)", len(s.Messages), len(got.Messages))
	}
}

// Saved files are provider-neutral: the internal JSON tags (kind, tool_use_id,
// ...) must appear, and no OpenAI wire strings (function, tool_calls) may leak.
func TestSavedFileIsProviderNeutral(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session.json")
	if err := sampleSession().Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	body := string(data)
	for _, forbidden := range []string{"function", "tool_calls", "tool_call_id", "arguments"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("saved session leaked provider wire string %q:\n%s", forbidden, body)
		}
	}
	for _, want := range []string{"tool_use_id", "result_for_id"} {
		if !strings.Contains(body, want) {
			t.Fatalf("saved session missing provider-neutral tag %q", want)
		}
	}
}

// Cross-provider resume: a session saved under anthropic loads cleanly and its
// transcript is re-sendable; the caller (Phase 10) overrides provider/model from
// flags. Here we assert the loaded transcript is valid and provider field is
// preserved as recorded.
func TestCrossProviderResume(t *testing.T) {
	s := sampleSession()
	path := filepath.Join(t.TempDir(), "session.json")
	if err := s.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	got, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got.Provider != "anthropic" {
		t.Fatalf("provider %q not preserved", got.Provider)
	}
	if err := llm.ValidateTranscript(got.Messages); err != nil {
		t.Fatalf("transcript not re-sendable under a different provider: %v", err)
	}
}

func TestLoadMissingFileIsError(t *testing.T) {
	_, err := Load(filepath.Join(t.TempDir(), "nope.json"))
	if err == nil {
		t.Fatalf("expected error loading missing session file")
	}
}

func TestLoadMalformedFileIsError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	_, err := Load(path)
	if err == nil {
		t.Fatalf("expected error loading malformed session file")
	}
}

// DefaultPath builds a timestamped path under an injectable state dir, ending in
// .json under sessions/.
func TestDefaultPath(t *testing.T) {
	stateDir := t.TempDir()
	at := time.Date(2026, 6, 9, 14, 30, 15, 0, time.UTC)
	p := DefaultPath(stateDir, at)
	if filepath.Dir(p) != filepath.Join(stateDir, "harness", "sessions") {
		t.Fatalf("DefaultPath dir %q unexpected", filepath.Dir(p))
	}
	if !strings.HasSuffix(p, ".json") {
		t.Fatalf("DefaultPath %q does not end in .json", p)
	}
	// The timestamp must round to a path that does not collide minute-to-minute.
	p2 := DefaultPath(stateDir, at.Add(time.Second))
	if p == p2 {
		t.Fatalf("DefaultPath collides one second apart: %q", p)
	}
}
