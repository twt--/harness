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

func readEditedInputs(t *testing.T, input string, count int) []replInput {
	t.Helper()
	var out bytes.Buffer
	editor := newPromptLineEditor(strings.NewReader(input), &out)
	inputs := make([]replInput, 0, count)
	for range count {
		input, ok, err := editor.read("> ")
		if err != nil {
			t.Fatalf("read = %v", err)
		}
		if !ok {
			t.Fatal("read returned ok=false")
		}
		inputs = append(inputs, input)
	}
	return inputs
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

func TestPromptLineEditorArrowUpRecallsHistory(t *testing.T) {
	inputs := readEditedInputs(t, "first\nsecond\n\x1b[A\n", 3)

	if inputs[2].text != "second" {
		t.Fatalf("history recall = %q, want second", inputs[2].text)
	}
}

func TestPromptLineEditorArrowUpDownRestoresDraft(t *testing.T) {
	inputs := readEditedInputs(t, "first\nsecond\ndra\x1b[A\x1b[A\x1b[B\x1b[Bft\n", 3)

	if inputs[2].text != "draft" {
		t.Fatalf("draft after history navigation = %q, want draft", inputs[2].text)
	}
}

func TestPromptLineEditorRecalledHistoryCanBeEdited(t *testing.T) {
	inputs := readEditedInputs(t, "hello\n\x1b[A!\n", 2)

	if inputs[1].text != "hello!" {
		t.Fatalf("edited history recall = %q, want hello!", inputs[1].text)
	}
}

func TestPromptLineEditorSS3ArrowUpRecallsHistory(t *testing.T) {
	inputs := readEditedInputs(t, "first\n\x1bOA\n", 2)

	if inputs[1].text != "first" {
		t.Fatalf("SS3 history recall = %q, want first", inputs[1].text)
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

func TestPromptLineEditorMultilinePasteIsNotAddedToHistory(t *testing.T) {
	pasted := "first line\nsecond line"
	inputs := readEditedInputs(t, bracketedPasteStart+pasted+bracketedPasteEnd+"\x1b[A\n", 2)

	if !inputs[0].pasted || inputs[0].text != pasted {
		t.Fatalf("first input = %+v, want pasted %q", inputs[0], pasted)
	}
	if inputs[1].text != "" {
		t.Fatalf("history after multiline paste = %q, want empty", inputs[1].text)
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

func TestPromptLineEditorRedrawClearsWrappedRows(t *testing.T) {
	var out bytes.Buffer
	editor := newPromptLineEditor(strings.NewReader("abcde\n"), &out)
	editor.columns = func() int { return 6 }

	input, ok, err := editor.read("> ")
	if err != nil {
		t.Fatalf("read = %v", err)
	}
	if !ok {
		t.Fatal("read returned ok=false")
	}
	if input.text != "abcde" {
		t.Fatalf("input text = %q, want abcde", input.text)
	}

	got := out.String()
	want := "\x1b8\r\x1b[2K\x1b[B\r\x1b[2K\x1b8> abcde"
	if !strings.Contains(got, want) {
		t.Fatalf("wrapped redraw should clear both prompt rows before repainting\nwant fragment: %q\ngot: %q", want, got)
	}
}
