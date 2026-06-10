package tools

import (
	"fmt"
	"strings"
)

// Central output caps, applied in Dispatch as a backstop for every tool
// (design §8.3): 64 KB or 1000 lines per result, whichever hits first.
const (
	maxResultBytes = 64 * 1024
	maxResultLines = 1000
)

// truncate caps s to the central limits, appending a teaching marker that
// reports the original size and advises how to narrow. The byte cap and line
// cap are independent; whichever triggers first wins. If neither triggers, s is
// returned unchanged.
func truncate(s string) string {
	totalLines := strings.Count(s, "\n")
	if !strings.HasSuffix(s, "\n") && len(s) > 0 {
		totalLines++
	}

	if totalLines > maxResultLines {
		// Keep the first maxResultLines lines.
		idx := 0
		count := 0
		for count < maxResultLines {
			nl := strings.IndexByte(s[idx:], '\n')
			if nl < 0 {
				idx = len(s)
				break
			}
			idx += nl + 1
			count++
		}
		kept := s[:idx]
		marker := fmt.Sprintf("[truncated: showing first %d of %d lines; use read_file offset/limit or grep to narrow]", maxResultLines, totalLines)
		out := kept
		if !strings.HasSuffix(out, "\n") {
			out += "\n"
		}
		return out + marker
	}

	if len(s) > maxResultBytes {
		marker := fmt.Sprintf("\n[truncated: showing first %s of %s; use read_file offset/limit or grep to narrow]", humanBytes(maxResultBytes), humanBytes(len(s)))
		return s[:maxResultBytes] + marker
	}

	return s
}

// humanBytes renders a byte count as a short human-readable size.
func humanBytes(n int) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%dB", n)
	}
	div, exp := int64(unit), 0
	for v := int64(n) / unit; v >= unit; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f%cB", float64(n)/float64(div), "KMGTPE"[exp])
}
