//go:build !darwin && !linux

package term

// Reset is a no-op on platforms without termios support.
func Reset() error {
	return nil
}
