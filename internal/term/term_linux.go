//go:build linux

package term

import (
	"syscall"
	"unsafe"
)

const (
	reqGet = syscall.TCGETS
	reqSet = syscall.TCSETS

	// EXTPROC from <asm-generic/termbits.h> (0x10000 on amd64 and arm64);
	// missing from stdlib syscall on linux/amd64.
	extPROC = 0x10000
)

// sane applies GNU coreutils' `stty sane` (the SANE_SET/SANE_UNSET entries in
// src/stty.c plus the control_info saneval defaults): specific bits are OR'd
// in or cleared from the existing termios — whole flag words are never
// replaced — the listed control characters are reset, and VMIN/VTIME, baud,
// CSIZE/parity, CLOCAL, IXON, and PENDIN are deliberately left untouched.
// The output delay masks (NLDLY etc.), which GNU sane zeroes, are omitted:
// the constants are not in stdlib syscall and zero delays have been
// universal for decades.
func sane(t *syscall.Termios) {
	t.Iflag &^= syscall.IGNBRK | syscall.INLCR | syscall.IGNCR |
		syscall.IXOFF | syscall.IUTF8 | syscall.IUCLC | syscall.IXANY
	t.Iflag |= syscall.BRKINT | syscall.ICRNL | syscall.IMAXBEL
	// IXON is left as-is, matching GNU.

	t.Oflag |= syscall.OPOST | syscall.ONLCR
	t.Oflag &^= syscall.OLCUC | syscall.OCRNL | syscall.ONOCR | syscall.ONLRET |
		syscall.OFILL | syscall.OFDEL

	t.Cflag |= syscall.CREAD

	t.Lflag |= syscall.ISIG | syscall.ICANON | syscall.IEXTEN |
		syscall.ECHO | syscall.ECHOE | syscall.ECHOK |
		syscall.ECHOCTL | syscall.ECHOKE
	t.Lflag &^= syscall.ECHONL | syscall.NOFLSH | syscall.XCASE |
		syscall.TOSTOP | syscall.ECHOPRT | syscall.FLUSHO | extPROC

	// control_info saneval defaults; _POSIX_VDISABLE is 0 on linux.
	t.Cc[syscall.VINTR] = 0x03    // ^C
	t.Cc[syscall.VQUIT] = 0x1C    // ^\
	t.Cc[syscall.VERASE] = 0x7F   // DEL
	t.Cc[syscall.VKILL] = 0x15    // ^U
	t.Cc[syscall.VEOF] = 0x04     // ^D
	t.Cc[syscall.VEOL] = 0x00     // disabled
	t.Cc[syscall.VEOL2] = 0x00    // disabled
	t.Cc[syscall.VSWTC] = 0x00    // disabled
	t.Cc[syscall.VSTART] = 0x11   // ^Q
	t.Cc[syscall.VSTOP] = 0x13    // ^S
	t.Cc[syscall.VSUSP] = 0x1A    // ^Z
	t.Cc[syscall.VREPRINT] = 0x12 // ^R
	t.Cc[syscall.VWERASE] = 0x17  // ^W
	t.Cc[syscall.VLNEXT] = 0x16   // ^V
	t.Cc[syscall.VDISCARD] = 0x0F // ^O
	// VMIN and VTIME are left untouched (GNU sane_mode breaks before them).
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
