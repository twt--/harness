package ui

import (
	"bytes"
	"testing"
	"time"
)

func TestTimestampedWriterPrefixesEachLineShort(t *testing.T) {
	var b bytes.Buffer
	now := func() time.Time { return time.Date(2026, 6, 13, 7, 8, 9, 0, time.Local) }
	w := NewTimestampedWriter(&b, now, TimestampShortLayout)

	if _, err := w.Write([]byte("hello\nworld")); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if got, want := b.String(), "07:08:09 hello\n07:08:09 world"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestTimestampedWriterSupportsFullLayout(t *testing.T) {
	var b bytes.Buffer
	now := func() time.Time { return time.Date(2026, 6, 13, 7, 8, 9, 0, time.Local) }
	w := NewTimestampedWriter(&b, now, TimestampFullLayout)

	if _, err := w.Write([]byte("hello\n")); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if got, want := b.String(), "2026-06-13 07:08:09 hello\n"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestTimestampedWriterReprefixesAfterClearLine(t *testing.T) {
	var b bytes.Buffer
	now := func() time.Time { return time.Date(2026, 6, 13, 7, 8, 9, 0, time.Local) }
	w := NewTimestampedWriter(&b, now, TimestampShortLayout)

	if _, err := w.Write([]byte("draft")); err != nil {
		t.Fatalf("Write draft: %v", err)
	}
	if _, err := w.Write([]byte("\r\x1b[2Kprompt")); err != nil {
		t.Fatalf("Write redraw: %v", err)
	}
	if got, want := b.String(), "07:08:09 draft\r\x1b[2K07:08:09 prompt"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}
