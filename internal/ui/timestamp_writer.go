package ui

import (
	"bytes"
	"fmt"
	"io"
	"time"
)

const (
	TimestampShortLayout = "15:04:05"
	TimestampFullLayout  = "2006-01-02 15:04:05"
)

var clearCurrentLine = []byte("\r\x1b[2K")

// NewTimestampedWriter prefixes each terminal line with the current local time.
func NewTimestampedWriter(w io.Writer, now func() time.Time, layout string) io.Writer {
	if now == nil {
		now = time.Now
	}
	if layout == "" {
		layout = TimestampShortLayout
	}
	return &timestampedWriter{w: w, now: now, layout: layout, atLineStart: true}
}

type timestampedWriter struct {
	w           io.Writer
	now         func() time.Time
	layout      string
	atLineStart bool
}

func (tw *timestampedWriter) Write(p []byte) (int, error) {
	origLen := len(p)
	for len(p) > 0 {
		if bytes.HasPrefix(p, clearCurrentLine) {
			if _, err := tw.w.Write(clearCurrentLine); err != nil {
				return 0, err
			}
			p = p[len(clearCurrentLine):]
			tw.atLineStart = true
			continue
		}
		if tw.atLineStart {
			if _, err := fmt.Fprintf(tw.w, "%s ", tw.now().Format(tw.layout)); err != nil {
				return 0, err
			}
			tw.atLineStart = false
		}
		i := bytes.IndexAny(p, "\r\n")
		if i < 0 {
			if _, err := tw.w.Write(p); err != nil {
				return 0, err
			}
			break
		}
		if _, err := tw.w.Write(p[:i+1]); err != nil {
			return 0, err
		}
		p = p[i+1:]
		tw.atLineStart = true
	}
	return origLen, nil
}
