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
// reports the original size and advises how to narrow. Both caps always hold:
// the line cap applies first, then the byte cap is re-applied to whatever
// remains so that many long lines cannot bypass the payload-size backstop. If
// neither cap triggers, s is returned unchanged.
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
		if !strings.HasSuffix(kept, "\n") {
			kept += "\n"
		}
		marker := fmt.Sprintf("[truncated: showing first %d of %d lines; use read_file offset/limit or grep to narrow]", maxResultLines, totalLines)
		// The byte cap is a payload-size backstop: re-apply it so that many
		// large lines under the line cap cannot bust the 64KB limit.
		return capBytes(kept+marker, len(s))
	}

	return capBytes(s, len(s))
}

// capBytes enforces maxResultBytes on s, appending a marker that reports the
// original byte size (origBytes) when it trims. If s already carries a
// line-truncation marker, capBytes trims the kept body, not the marker tail.
func capBytes(s string, origBytes int) string {
	if len(s) <= maxResultBytes {
		return s
	}
	marker := fmt.Sprintf("\n[truncated: showing first %s of %s; use read_file offset/limit or grep to narrow]", humanBytes(maxResultBytes), humanBytes(origBytes))
	keep := maxResultBytes - len(marker)
	if keep < 0 {
		keep = 0
	}
	return s[:keep] + marker
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
