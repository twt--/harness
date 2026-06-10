package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"harness/internal/llm"
)

// fakeTool is a configurable Tool for exercising Dispatch.
type fakeTool struct {
	name   string
	desc   string
	schema string
	run    func(ctx context.Context, input json.RawMessage) (string, error)
}

func (f fakeTool) Name() string            { return f.name }
func (f fakeTool) Description() string     { return f.desc }
func (f fakeTool) Schema() json.RawMessage { return json.RawMessage(f.schema) }
func (f fakeTool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	return f.run(ctx, input)
}

func newOK(name, out string) fakeTool {
	return fakeTool{
		name:   name,
		desc:   "ok tool",
		schema: `{"type":"object"}`,
		run: func(ctx context.Context, input json.RawMessage) (string, error) {
			return out, nil
		},
	}
}

func TestRegistrySpecsOrdered(t *testing.T) {
	r := &Registry{}
	r.Register(newOK("alpha", "a"))
	r.Register(newOK("beta", "b"))
	r.Register(newOK("gamma", "c"))

	specs := r.Specs()
	if len(specs) != 3 {
		t.Fatalf("want 3 specs, got %d", len(specs))
	}
	want := []string{"alpha", "beta", "gamma"}
	for i, s := range specs {
		if s.Name != want[i] {
			t.Errorf("specs[%d].Name = %q, want %q", i, s.Name, want[i])
		}
	}
	// Parameters must be passed through unchanged from the tool's Schema().
	if string(specs[0].Parameters) != `{"type":"object"}` {
		t.Errorf("Parameters not passed through: %q", specs[0].Parameters)
	}
	if specs[0].Description != "ok tool" {
		t.Errorf("Description not passed through: %q", specs[0].Description)
	}
}

func TestDispatch(t *testing.T) {
	panicTool := fakeTool{
		name:   "boom",
		desc:   "panics",
		schema: `{"type":"object"}`,
		run: func(ctx context.Context, input json.RawMessage) (string, error) {
			panic("kaboom")
		},
	}
	errTool := fakeTool{
		name:   "err",
		desc:   "errors",
		schema: `{"type":"object"}`,
		run: func(ctx context.Context, input json.RawMessage) (string, error) {
			return "", fmt.Errorf("something broke")
		},
	}
	argTool := fakeTool{
		name:   "needsarg",
		desc:   "validates args",
		schema: `{"type":"object"}`,
		run: func(ctx context.Context, input json.RawMessage) (string, error) {
			var v struct {
				X int `json:"x"`
			}
			if err := json.Unmarshal(input, &v); err != nil {
				return "", err
			}
			return fmt.Sprintf("x=%d", v.X), nil
		},
	}

	r := &Registry{}
	r.Register(newOK("ok", "all good"))
	r.Register(panicTool)
	r.Register(errTool)
	r.Register(argTool)

	tests := []struct {
		name        string
		call        llm.ToolCall
		wantText    string
		wantErr     bool
		wantContain bool // wantText is a substring rather than the whole text
	}{
		{
			name:     "success passes through",
			call:     llm.ToolCall{ID: "1", Name: "ok", Input: json.RawMessage(`{}`)},
			wantText: "all good",
			wantErr:  false,
		},
		{
			name:        "unknown tool",
			call:        llm.ToolCall{ID: "2", Name: "nope", Input: json.RawMessage(`{}`)},
			wantText:    `error: unknown tool "nope"`,
			wantErr:     true,
			wantContain: true,
		},
		{
			name:        "invalid json args",
			call:        llm.ToolCall{ID: "3", Name: "needsarg", Input: json.RawMessage(`{not json`)},
			wantText:    "error: invalid arguments:",
			wantErr:     true,
			wantContain: true,
		},
		{
			name:        "tool returns error",
			call:        llm.ToolCall{ID: "4", Name: "err", Input: json.RawMessage(`{}`)},
			wantText:    "error: something broke",
			wantErr:     true,
			wantContain: true,
		},
		{
			name:        "tool panics",
			call:        llm.ToolCall{ID: "5", Name: "boom", Input: json.RawMessage(`{}`)},
			wantText:    "error: tool panicked: kaboom",
			wantErr:     true,
			wantContain: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			res := r.Dispatch(context.Background(), tc.call)
			if res.ForID != tc.call.ID {
				t.Errorf("ForID = %q, want %q", res.ForID, tc.call.ID)
			}
			if res.IsError != tc.wantErr {
				t.Errorf("IsError = %v, want %v (text=%q)", res.IsError, tc.wantErr, res.Text)
			}
			if tc.wantContain {
				if !strings.Contains(res.Text, tc.wantText) {
					t.Errorf("Text = %q, want substring %q", res.Text, tc.wantText)
				}
			} else if res.Text != tc.wantText {
				t.Errorf("Text = %q, want %q", res.Text, tc.wantText)
			}
		})
	}
}

func TestDispatchEmptyInputTreatedAsObject(t *testing.T) {
	// Models sometimes omit the input entirely for zero-arg tools. Dispatch
	// must not reject an empty/nil Input as invalid JSON; the tool sees "{}".
	r := &Registry{}
	r.Register(fakeTool{
		name:   "z",
		desc:   "zero arg",
		schema: `{"type":"object"}`,
		run: func(ctx context.Context, input json.RawMessage) (string, error) {
			var v map[string]any
			if err := json.Unmarshal(input, &v); err != nil {
				return "", err
			}
			return "ok", nil
		},
	})
	for _, in := range []json.RawMessage{nil, json.RawMessage(""), json.RawMessage("{}")} {
		res := r.Dispatch(context.Background(), llm.ToolCall{ID: "1", Name: "z", Input: in})
		if res.IsError {
			t.Errorf("input %q: unexpected error %q", in, res.Text)
		}
	}
}

func TestDispatchTruncateByBytes(t *testing.T) {
	big := strings.Repeat("x", 70*1024) // > 64KB, single line
	r := &Registry{}
	r.Register(newOK("big", big))

	res := r.Dispatch(context.Background(), llm.ToolCall{ID: "1", Name: "big", Input: json.RawMessage(`{}`)})
	if res.IsError {
		t.Fatalf("unexpected error: %q", res.Text)
	}
	if len(res.Text) > len(big) {
		t.Errorf("truncated output longer than input: %d > %d", len(res.Text), len(big))
	}
	if !strings.Contains(res.Text, "[truncated:") {
		t.Errorf("missing truncation marker: %q", res.Text[max(0, len(res.Text)-200):])
	}
	// Marker reports the original size in bytes/KB.
	if !strings.Contains(res.Text, "use read_file offset/limit or grep to narrow") {
		t.Errorf("marker missing narrowing advice: %q", res.Text)
	}
}

func TestDispatchTruncateByLines(t *testing.T) {
	var b strings.Builder
	for i := 0; i < 4213; i++ {
		fmt.Fprintf(&b, "line %d\n", i)
	}
	r := &Registry{}
	r.Register(newOK("many", b.String()))

	res := r.Dispatch(context.Background(), llm.ToolCall{ID: "1", Name: "many", Input: json.RawMessage(`{}`)})
	if res.IsError {
		t.Fatalf("unexpected error: %q", res.Text)
	}
	if !strings.Contains(res.Text, "[truncated: showing first 1000 of 4213 lines") {
		t.Errorf("missing line-truncation marker with counts: tail=%q", res.Text[max(0, len(res.Text)-200):])
	}
	if !strings.Contains(res.Text, "use read_file offset/limit or grep to narrow") {
		t.Errorf("marker missing narrowing advice")
	}
	// Only the first 1000 lines should remain (plus the marker line).
	lines := strings.Split(strings.TrimRight(res.Text, "\n"), "\n")
	if len(lines) > 1002 {
		t.Errorf("expected ~1001 lines after truncation, got %d", len(lines))
	}
	if !strings.HasPrefix(res.Text, "line 0\n") {
		t.Errorf("first line not preserved: %q", res.Text[:20])
	}
}

func TestDispatchNoTruncateWithinCaps(t *testing.T) {
	out := "small output\nwith two lines"
	r := &Registry{}
	r.Register(newOK("small", out))
	res := r.Dispatch(context.Background(), llm.ToolCall{ID: "1", Name: "small", Input: json.RawMessage(`{}`)})
	if res.Text != out {
		t.Errorf("output mutated: %q", res.Text)
	}
}
