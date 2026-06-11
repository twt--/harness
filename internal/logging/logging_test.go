package logging

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
)

func TestPlainHandlerRendersLevelCategoryAndMessage(t *testing.T) {
	var b bytes.Buffer
	logger := slog.New(NewPlainHandler(&b, HandlerOptions{Level: slog.LevelInfo}))

	logger.Warn(`Tool "rg" is disabled. Reason: "rg" binary not found.`, Category("cli_tools"))

	got := b.String()
	want := "[warn] [cli_tools] Tool \"rg\" is disabled. Reason: \"rg\" binary not found.\n"
	if got != want {
		t.Fatalf("log output = %q, want %q", got, want)
	}
}

func TestPlainHandlerFiltersByLevel(t *testing.T) {
	var b bytes.Buffer
	logger := slog.New(NewPlainHandler(&b, HandlerOptions{Level: slog.LevelWarn}))

	logger.Info("hidden")
	logger.Warn("shown")

	got := b.String()
	if got != "[warn] shown\n" {
		t.Fatalf("log output = %q", got)
	}
}

func TestPlainHandlerQuietSuppressesNonErrorRecords(t *testing.T) {
	var b bytes.Buffer
	logger := slog.New(NewPlainHandler(&b, HandlerOptions{
		Level: slog.LevelDebug,
		Quiet: true,
	}))

	logger.Warn("hidden")
	logger.Error("shown")

	if got := b.String(); got != "[error] shown\n" {
		t.Fatalf("quiet log output = %q", got)
	}
}

func TestPlainHandlerRendersExtraAttrsAsPlainText(t *testing.T) {
	var b bytes.Buffer
	logger := slog.New(NewPlainHandler(&b, HandlerOptions{Level: slog.LevelInfo})).
		With("tool", "git").
		WithGroup("detail")

	logger.Info("disabled", slog.String("reason", "missing"))

	got := b.String()
	for _, want := range []string{"[info] disabled", `tool="git"`, `detail.reason="missing"`} {
		if !strings.Contains(got, want) {
			t.Fatalf("log output %q missing %q", got, want)
		}
	}
}

func TestParseLevel(t *testing.T) {
	tests := map[string]slog.Level{
		"":        slog.LevelInfo,
		"DEBUG":   slog.LevelDebug,
		"info":    slog.LevelInfo,
		"warn":    slog.LevelWarn,
		"warning": slog.LevelWarn,
		"error":   slog.LevelError,
	}
	for input, want := range tests {
		got, err := ParseLevel(input)
		if err != nil {
			t.Fatalf("ParseLevel(%q): %v", input, err)
		}
		if got != want {
			t.Fatalf("ParseLevel(%q) = %v, want %v", input, got, want)
		}
	}
	if _, err := ParseLevel("verbose"); err == nil {
		t.Fatal("ParseLevel(\"verbose\") should fail")
	}
}
