package ui

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"strings"
	"unicode"
)

const (
	ctrlD = 0x04
	del   = 0x7f
)

type promptLineEditor struct {
	r *bufio.Reader
	w io.Writer
}

func newPromptLineEditor(in io.Reader, w io.Writer) *promptLineEditor {
	return &promptLineEditor{
		r: bufio.NewReader(in),
		w: w,
	}
}

func (e *promptLineEditor) read(prompt string) (replInput, bool, error) {
	state := lineEditState{prompt: prompt}
	if err := state.redraw(e.w); err != nil {
		return replInput{}, false, err
	}

	for {
		r, _, err := e.r.ReadRune()
		if err != nil {
			if errors.Is(err, io.EOF) && len(state.buf) > 0 {
				return replInput{text: string(state.buf)}, true, nil
			}
			if errors.Is(err, io.EOF) {
				return replInput{}, false, nil
			}
			return replInput{}, false, err
		}

		switch r {
		case '\r', '\n':
			if _, err := fmt.Fprint(e.w, "\n"); err != nil {
				return replInput{}, false, err
			}
			return replInput{text: string(state.buf)}, true, nil
		case rune(lineTermEdit):
			if _, err := fmt.Fprint(e.w, "\n"); err != nil {
				return replInput{}, false, err
			}
			return replInput{text: string(state.buf), edit: true}, true, nil
		case ctrlD:
			if len(state.buf) == 0 {
				return replInput{}, false, nil
			}
		case '\b', del:
			state.backspace()
			if err := state.redraw(e.w); err != nil {
				return replInput{}, false, err
			}
		case rune(lineTermEscape):
			action, text, err := e.readEscape()
			if err != nil {
				if errors.Is(err, io.EOF) {
					return replInput{}, false, nil
				}
				return replInput{}, false, err
			}
			switch action {
			case lineEditLeft:
				state.left()
			case lineEditRight:
				state.right()
			case lineEditDelete:
				state.delete()
			case lineEditPaste:
				if len(state.buf) == 0 {
					if _, err := fmt.Fprintf(e.w, "\r\x1b[2K%s%s\n", prompt, text); err != nil {
						return replInput{}, false, err
					}
					return replInput{text: text, pasted: true}, true, nil
				}
				state.insertString(text)
			}
			if err := state.redraw(e.w); err != nil {
				return replInput{}, false, err
			}
		default:
			if r == '\t' || unicode.IsPrint(r) {
				state.insert(r)
				if err := state.redraw(e.w); err != nil {
					return replInput{}, false, err
				}
			}
		}
	}
}

type lineEditAction int

const (
	lineEditIgnore lineEditAction = iota
	lineEditLeft
	lineEditRight
	lineEditDelete
	lineEditPaste
)

func (e *promptLineEditor) readEscape() (lineEditAction, string, error) {
	c, err := e.r.ReadByte()
	if err != nil {
		return lineEditIgnore, "", err
	}
	switch c {
	case '[':
		seq, err := e.readCSI()
		if err != nil {
			return lineEditIgnore, "", err
		}
		switch seq {
		case "C":
			return lineEditRight, "", nil
		case "D":
			return lineEditLeft, "", nil
		case "3~":
			return lineEditDelete, "", nil
		case "200~":
			text, err := e.readBracketedPaste()
			if err != nil {
				return lineEditIgnore, "", err
			}
			return lineEditPaste, text, nil
		default:
			return lineEditIgnore, "", nil
		}
	case 'O':
		c, err := e.r.ReadByte()
		if err != nil {
			return lineEditIgnore, "", err
		}
		switch c {
		case 'C':
			return lineEditRight, "", nil
		case 'D':
			return lineEditLeft, "", nil
		default:
			return lineEditIgnore, "", nil
		}
	default:
		return lineEditIgnore, "", nil
	}
}

func (e *promptLineEditor) readCSI() (string, error) {
	var b strings.Builder
	for {
		c, err := e.r.ReadByte()
		if err != nil {
			return b.String(), err
		}
		b.WriteByte(c)
		if c >= '@' && c <= '~' {
			return b.String(), nil
		}
	}
}

func (e *promptLineEditor) readBracketedPaste() (string, error) {
	var b strings.Builder
	for {
		c, err := e.r.ReadByte()
		if err != nil {
			return b.String(), err
		}
		b.WriteByte(c)
		text := b.String()
		if strings.HasSuffix(text, bracketedPasteEnd) {
			return strings.TrimSuffix(text, bracketedPasteEnd), nil
		}
	}
}

type lineEditState struct {
	prompt string
	buf    []rune
	cursor int
}

func (s *lineEditState) insert(r rune) {
	s.buf = append(s.buf, 0)
	copy(s.buf[s.cursor+1:], s.buf[s.cursor:])
	s.buf[s.cursor] = r
	s.cursor++
}

func (s *lineEditState) insertString(text string) {
	for _, r := range text {
		s.insert(r)
	}
}

func (s *lineEditState) left() {
	if s.cursor > 0 {
		s.cursor--
	}
}

func (s *lineEditState) right() {
	if s.cursor < len(s.buf) {
		s.cursor++
	}
}

func (s *lineEditState) backspace() {
	if s.cursor == 0 {
		return
	}
	copy(s.buf[s.cursor-1:], s.buf[s.cursor:])
	s.buf = s.buf[:len(s.buf)-1]
	s.cursor--
}

func (s *lineEditState) delete() {
	if s.cursor >= len(s.buf) {
		return
	}
	copy(s.buf[s.cursor:], s.buf[s.cursor+1:])
	s.buf = s.buf[:len(s.buf)-1]
}

func (s *lineEditState) redraw(w io.Writer) error {
	if _, err := fmt.Fprintf(w, "\r\x1b[2K%s%s", s.prompt, string(s.buf)); err != nil {
		return err
	}
	if back := len(s.buf) - s.cursor; back > 0 {
		if _, err := fmt.Fprintf(w, "\x1b[%dD", back); err != nil {
			return err
		}
	}
	return nil
}
