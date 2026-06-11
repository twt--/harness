//go:build darwin || linux

package term

import (
	"errors"
	"fmt"
	"os"
	"syscall"
)

// softReset undoes the terminal-emulator modes a crashed full-screen program
// commonly leaves enabled. Unlike RIS (\033c) it does not clear the screen or
// scrollback. DECSTR alone does not reliably disable mouse/focus/paste
// reporting across emulators, hence the explicit DECRST list.
//
// Leaving the alternate screen comes first and is guarded by DECSC: DECRST
// 1049 performs a DECRC cursor-restore even when the alternate screen is not
// active, and with no position ever saved that restores home — jumping the
// cursor to the top of the screen. Saving the cursor immediately before makes
// the restore a no-op in the normal case, while after a crashed 1049h app the
// normal screen's slot still holds the position saved on entry. The pair must
// precede DECSTR, which resets the saved-cursor slot in some emulators.
const softReset = "\x1b7\x1b[?1049l" + // leave alternate screen (DECSC-guarded, see above)
	"\x1b[!p" + // DECSTR: SGR, autowrap, origin/insert mode, cursor visible
	"\x1b[?1003l\x1b[?1002l\x1b[?1000l" + // mouse tracking off (any-event, button-event, normal)
	"\x1b[?1006l\x1b[?1005l\x1b[?1015l" + // mouse coordinate encodings off (SGR, UTF-8, urxvt)
	"\x1b[?1004l" + // focus reporting off (the ESC[I / ESC[O junk on focus changes)
	"\x1b[?2004l" + // bracketed paste off
	"\x1b[?25h" + // show cursor (DECSTR covers it in xterm; explicit for partial emulators)
	"\x1b(B\x0f" + // G0 = ASCII, shift in (undo line-drawing charset)
	"\x1b[0m" // SGR reset (also in DECSTR; explicit for partial emulators)

// Reset restores the controlling terminal to a usable state: kernel termios
// to the platform's `stty sane` equivalent (echo, canonical mode, default
// control characters), then the emulator soft reset above. It targets
// /dev/tty so it works regardless of stdin/stderr redirection, and is a
// silent no-op when the process has no controlling terminal.
func Reset() error {
	f, err := os.OpenFile("/dev/tty", os.O_RDWR|syscall.O_NOCTTY, 0)
	if err != nil {
		return nil // no controlling terminal: nothing to fix
	}
	defer f.Close()

	tio, err := getTermios(f.Fd())
	if err != nil {
		if errors.Is(err, syscall.ENOTTY) {
			return nil
		}
		return fmt.Errorf("term: get termios: %w", err)
	}
	sane(&tio)
	if err := setTermios(f.Fd(), &tio); err != nil {
		return fmt.Errorf("term: set termios: %w", err)
	}
	if _, err := f.WriteString(softReset); err != nil {
		return fmt.Errorf("term: write soft reset: %w", err)
	}
	return nil
}
