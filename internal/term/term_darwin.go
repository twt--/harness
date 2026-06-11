//go:build darwin

package term

import (
	"syscall"
	"unsafe"
)

const (
	reqGet = syscall.TIOCGETA
	reqSet = syscall.TIOCSETA

	// _POSIX_VDISABLE on darwin; disables a control character.
	vDisable = 0xFF

	// ALTWERASE from <sys/termios.h>; not exposed by stdlib syscall.
	altWERASE = 0x00000200
)

// sane applies the BSD `stty sane` flag semantics (f_sane in bin/stty/key.c):
// the four flag words are replaced wholesale with the TTYDEF_* defaults from
// <sys/ttydefaults.h>, preserving the current CLOCAL bit and the LKEEP set of
// lflag bits. Unlike /bin/stty sane — which omits this by accident of its
// implementation — the control characters are also reset to the ttydefchars
// defaults, as BSD libc's cfmakesane() does, so corrupted ^C/erase bindings
// are repaired too. Speeds are left alone (forcing TTYDEF_SPEED's 9600 baud
// would be wrong on a pty).
func sane(t *syscall.Termios) {
	keepCLOCAL := t.Cflag & syscall.CLOCAL
	keepL := t.Lflag & (syscall.ECHOKE | syscall.ECHOE | syscall.ECHOK |
		syscall.ECHOPRT | syscall.ECHOCTL | altWERASE |
		syscall.TOSTOP | syscall.NOFLSH) // f_sane's LKEEP

	// TTYDEF_IFLAG
	t.Iflag = syscall.BRKINT | syscall.ICRNL | syscall.IMAXBEL |
		syscall.IXON | syscall.IXANY
	// TTYDEF_OFLAG
	t.Oflag = syscall.OPOST | syscall.ONLCR
	// TTYDEF_CFLAG, preserving CLOCAL
	t.Cflag = syscall.CREAD | syscall.CS8 | syscall.HUPCL | keepCLOCAL
	// TTYDEF_LFLAG, OR'd with the preserved LKEEP bits
	t.Lflag = syscall.ECHO | syscall.ICANON | syscall.ISIG | syscall.IEXTEN |
		syscall.ECHOE | syscall.ECHOKE | syscall.ECHOCTL | keepL

	// ttydefchars: the spare slots are _POSIX_VDISABLE, so disable everything
	// first and then set the defined defaults.
	for i := range t.Cc {
		t.Cc[i] = vDisable
	}
	t.Cc[syscall.VEOF] = 0x04     // ^D
	t.Cc[syscall.VERASE] = 0x7F   // DEL
	t.Cc[syscall.VWERASE] = 0x17  // ^W
	t.Cc[syscall.VKILL] = 0x15    // ^U
	t.Cc[syscall.VREPRINT] = 0x12 // ^R
	t.Cc[syscall.VINTR] = 0x03    // ^C
	t.Cc[syscall.VQUIT] = 0x1C    // ^\
	t.Cc[syscall.VSUSP] = 0x1A    // ^Z
	t.Cc[syscall.VDSUSP] = 0x19   // ^Y
	t.Cc[syscall.VSTART] = 0x11   // ^Q
	t.Cc[syscall.VSTOP] = 0x13    // ^S
	t.Cc[syscall.VLNEXT] = 0x16   // ^V
	t.Cc[syscall.VDISCARD] = 0x0F // ^O
	t.Cc[syscall.VMIN] = 1
	t.Cc[syscall.VTIME] = 0
	t.Cc[syscall.VSTATUS] = 0x14 // ^T
	// VEOL and VEOL2 stay disabled (vDisable), matching ttydefchars.
}

func getTermios(fd uintptr) (syscall.Termios, error) {
	var t syscall.Termios
	if _, _, errno := syscall.Syscall6(syscall.SYS_IOCTL, fd, reqGet,
		uintptr(unsafe.Pointer(&t)), 0, 0, 0); errno != 0 {
		return t, errno
	}
	return t, nil
}

func setTermios(fd uintptr, t *syscall.Termios) error {
	if _, _, errno := syscall.Syscall6(syscall.SYS_IOCTL, fd, reqSet,
		uintptr(unsafe.Pointer(t)), 0, 0, 0); errno != 0 {
		return errno
	}
	return nil
}
