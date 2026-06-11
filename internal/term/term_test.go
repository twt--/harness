//go:build darwin || linux

package term

import (
	"os"
	"strings"
	"syscall"
	"testing"
)

// softReset must disable every emulator mode a crashed TUI commonly leaves on
// (mouse tracking, focus reporting, bracketed paste, alternate screen) without
// clearing the screen or scrollback — that is the whole point of replacing RIS.
func TestSoftResetDisablesLeftoverModes(t *testing.T) {
	want := []string{
		"\x1b[!p",     // DECSTR soft reset
		"\x1b[?1000l", // mouse: normal tracking off
		"\x1b[?1002l", // mouse: button-event tracking off
		"\x1b[?1003l", // mouse: any-event tracking off
		"\x1b[?1005l", // mouse: UTF-8 coords off
		"\x1b[?1006l", // mouse: SGR coords off
		"\x1b[?1015l", // mouse: urxvt coords off
		"\x1b[?1004l", // focus in/out reporting off (ESC[I / ESC[O junk)
		"\x1b[?2004l", // bracketed paste off
		"\x1b[?1049l", // leave alternate screen
		"\x1b[?25h",   // show cursor
		"\x1b(B\x0f",  // G0 = ASCII, shift in (undo line-drawing charset)
		"\x1b[0m",     // SGR reset
	}
	for _, seq := range want {
		if !strings.Contains(softReset, seq) {
			t.Errorf("softReset missing %q", seq)
		}
	}
	for _, seq := range []string{"\x1bc", "[2J", "[3J"} {
		if strings.Contains(softReset, seq) {
			t.Errorf("softReset contains screen-destroying sequence %q", seq)
		}
	}
}

// Reset must be a silent no-op when the process has no controlling terminal
// (tests, pipes, CI). When one is present this test skips rather than mutate
// the developer's terminal; TestResetOnRealTTY covers that side.
func TestResetNoTTY(t *testing.T) {
	if f, err := os.OpenFile("/dev/tty", os.O_RDWR|syscall.O_NOCTTY, 0); err == nil {
		f.Close()
		t.Skip("controlling terminal present; no-op path not reachable")
	}
	if err := Reset(); err != nil {
		t.Fatalf("Reset() without a controlling terminal = %v, want nil", err)
	}
}

// TestResetOnRealTTY runs only when a controlling terminal is available (e.g.
// under script(1) or a developer's shell): it deliberately breaks the terminal
// (echo and canonical mode off), calls Reset, and verifies both are restored.
// The original termios is restored afterwards either way.
func TestResetOnRealTTY(t *testing.T) {
	f, err := os.OpenFile("/dev/tty", os.O_RDWR|syscall.O_NOCTTY, 0)
	if err != nil {
		t.Skip("no controlling terminal")
	}
	defer f.Close()

	orig, err := getTermios(f.Fd())
	if err != nil {
		t.Skipf("get termios on /dev/tty: %v", err)
	}
	defer setTermios(f.Fd(), &orig)

	broken := orig
	broken.Lflag &^= syscall.ECHO | syscall.ICANON
	if err := setTermios(f.Fd(), &broken); err != nil {
		t.Fatalf("breaking termios: %v", err)
	}

	if err := Reset(); err != nil {
		t.Fatalf("Reset() = %v", err)
	}

	got, err := getTermios(f.Fd())
	if err != nil {
		t.Fatalf("get termios after Reset: %v", err)
	}
	if got.Lflag&syscall.ECHO == 0 {
		t.Error("ECHO still off after Reset")
	}
	if got.Lflag&syscall.ICANON == 0 {
		t.Error("ICANON still off after Reset")
	}
}
