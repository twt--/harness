package ui

import (
	"bytes"
	"strings"
	"testing"
)

func readEditedInput(t *testing.T, input string) (replInput, bool, error) {
	t.Helper()
	var out bytes.Buffer
	return newPromptLineEditor(strings.NewReader(input), &out).read("> ")
}

func TestPromptLineEditorInsertsAtCursor(t *testing.T) {
	input, ok, err := readEditedInput(t, "abc\x1b[D\x1b[DX\n")
	if err != nil {
		t.Fatalf("read = %v", err)
	}
	if !ok {
		t.Fatal("read returned ok=false")
	}
	if input.text != "aXbc" {
		t.Fatalf("input text = %q, want aXbc", input.text)
	}
}

func TestPromptLineEditorBackspaceDeletesBeforeCursor(t *testing.T) {
	input, ok, err := readEditedInput(t, "abc\x1b[D\x7f\n")
	if err != nil {
		t.Fatalf("read = %v", err)
	}
	if !ok {
		t.Fatal("read returned ok=false")
	}
	if input.text != "ac" {
		t.Fatalf("input text = %q, want ac", input.text)
	}
}

func TestPromptLineEditorDeleteDeletesAtCursor(t *testing.T) {
	input, ok, err := readEditedInput(t, "abc\x1b[D\x1b[3~\n")
	if err != nil {
		t.Fatalf("read = %v", err)
	}
	if !ok {
		t.Fatal("read returned ok=false")
	}
	if input.text != "ab" {
		t.Fatalf("input text = %q, want ab", input.text)
	}
}

func TestPromptLineEditorCursorBoundariesAreNoops(t *testing.T) {
	input, ok, err := readEditedInput(t, "\x1b[Da\x1b[C\x1b[CX\n")
	if err != nil {
		t.Fatalf("read = %v", err)
	}
	if !ok {
		t.Fatal("read returned ok=false")
	}
	if input.text != "aX" {
		t.Fatalf("input text = %q, want aX", input.text)
	}
}

func TestPromptLineEditorIsRuneAware(t *testing.T) {
	input, ok, err := readEditedInput(t, "aé\x1b[DX\n")
	if err != nil {
		t.Fatalf("read = %v", err)
	}
	if !ok {
		t.Fatal("read returned ok=false")
	}
	if input.text != "aXé" {
		t.Fatalf("input text = %q, want aXé", input.text)
	}
}

func TestPromptLineEditorCtrlGReturnsEditInputWithDraft(t *testing.T) {
	input, ok, err := readEditedInput(t, "draft\a")
	if err != nil {
		t.Fatalf("read = %v", err)
	}
	if !ok {
		t.Fatal("read returned ok=false")
	}
	if !input.edit || input.text != "draft" {
		t.Fatalf("input = %+v, want edit draft", input)
	}
}

func TestPromptLineEditorCtrlDOnEmptyReturnsEOF(t *testing.T) {
	_, ok, err := readEditedInput(t, "\x04")
	if err != nil {
		t.Fatalf("read = %v", err)
	}
	if ok {
		t.Fatal("Ctrl-D on empty input should return ok=false")
	}
}

func TestPromptLineEditorCtrlDWithTextIsIgnored(t *testing.T) {
	input, ok, err := readEditedInput(t, "a\x04b\n")
	if err != nil {
		t.Fatalf("read = %v", err)
	}
	if !ok {
		t.Fatal("read returned ok=false")
	}
	if input.text != "ab" {
		t.Fatalf("input text = %q, want ab", input.text)
	}
}

func TestPromptLineEditorBracketedPasteSubmitsEmptyPrompt(t *testing.T) {
	pasted := "/exit is text\nsecond line"
	input, ok, err := readEditedInput(t, bracketedPasteStart+pasted+bracketedPasteEnd)
	if err != nil {
		t.Fatalf("read = %v", err)
	}
	if !ok {
		t.Fatal("read returned ok=false")
	}
	if !input.pasted || input.text != pasted {
		t.Fatalf("input = %+v, want pasted %q", input, pasted)
	}
}

func TestPromptLineEditorBracketedPasteInsertsAtCursor(t *testing.T) {
	input, ok, err := readEditedInput(t, "ab\x1b[D"+bracketedPasteStart+"X"+bracketedPasteEnd+"\n")
	if err != nil {
		t.Fatalf("read = %v", err)
	}
	if !ok {
		t.Fatal("read returned ok=false")
	}
	if input.pasted {
		t.Fatal("paste into non-empty prompt should not force literal-paste submission")
	}
	if input.text != "aXb" {
		t.Fatalf("input text = %q, want aXb", input.text)
	}
}
