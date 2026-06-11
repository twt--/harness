//go:build !darwin && !linux

package term

// Reset is a no-op on platforms without termios support.
func Reset() error {
	return nil
}

func Size() (rows, cols int, ok bool) {
	return 0, 0, false
}
