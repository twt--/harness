package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	webFetchDefaultMaxBytes = 1 << 20 // 1 MB
	webFetchMaxBytes        = 5 << 20 // 5 MB
	webFetchTimeout         = 30 * time.Second
	webFetchMaxRedirects    = 5
)

const webFetchSchema = `{
  "type": "object",
  "properties": {
    "url": {"type": "string", "description": "Absolute http or https URL to fetch."},
    "max_bytes": {"type": "integer", "description": "Maximum response bytes to read (default 1MB, cap 5MB)."}
  },
  "required": ["url"]
}`

type webFetch struct{}

func (webFetch) Name() string { return "web_fetch" }

func (webFetch) Description() string {
	return "Fetch a URL (http/https) and return its text content. HTML is reduced to readable text."
}

func (webFetch) Schema() json.RawMessage { return json.RawMessage(webFetchSchema) }

// web_fetch issues a GET and mutates no workspace state.
func (webFetch) ReadOnly() bool { return true }

func (webFetch) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var args struct {
		URL      string `json:"url"`
		MaxBytes int    `json:"max_bytes"`
	}
	if err := json.Unmarshal(input, &args); err != nil {
		return "", err
	}
	if args.URL == "" {
		return "", badArgs("url is required")
	}
	if args.MaxBytes < 0 {
		return "", badArgs("max_bytes must be >= 0")
	}
	if err := validateHTTPURL(args.URL); err != nil {
		return "", err
	}

	maxBytes := args.MaxBytes
	if maxBytes == 0 {
		maxBytes = webFetchDefaultMaxBytes
	}
	if maxBytes > webFetchMaxBytes {
		maxBytes = webFetchMaxBytes
	}

	client := &http.Client{
		Timeout: webFetchTimeout,
		// Re-validate every hop as http/https; cap redirect depth.
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= webFetchMaxRedirects {
				return fmt.Errorf("stopped after %d redirects", webFetchMaxRedirects)
			}
			return validateHTTPURL(req.URL.String())
		},
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, args.URL, nil)
	if err != nil {
		return "", err
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	contentType := resp.Header.Get("Content-Type")
	mediaType := mediaTypeOf(contentType)
	if !isTextual(mediaType) {
		return "", fmt.Errorf("unsupported content type %q (binary content is not fetched as text)", contentType)
	}

	// Read one extra byte so the cap can be reported without a Content-Length.
	raw, err := io.ReadAll(io.LimitReader(resp.Body, int64(maxBytes)))
	if err != nil {
		return "", err
	}

	body := string(raw)
	if mediaType == "text/html" {
		body = reduceHTML(body)
	}

	header := fmt.Sprintf("# %s (%s, %s)", resp.Request.URL.String(), resp.Status, contentType)
	return header + "\n" + body, nil
}

// validateHTTPURL rejects anything that is not an absolute http/https URL.
// Fetching arbitrary http/https URLs is web_fetch's documented purpose
// (design §2, §9.10); there is no private-IP/SSRF blocking by design.
func validateHTTPURL(raw string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("invalid url: %v", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("unsupported url scheme %q; only http and https are allowed", u.Scheme)
	}
	if u.Host == "" {
		return fmt.Errorf("url %q has no host", raw)
	}
	return nil
}

// mediaTypeOf extracts the lowercase media type from a Content-Type header,
// dropping any parameters (e.g. "; charset=utf-8").
func mediaTypeOf(contentType string) string {
	mt := contentType
	if i := strings.IndexByte(mt, ';'); i >= 0 {
		mt = mt[:i]
	}
	return strings.ToLower(strings.TrimSpace(mt))
}

// reduceHTML turns an HTML document into readable-ish text (design §9.10): it
// drops <script>/<style> element contents, strips all remaining tags,
// unescapes HTML entities, and collapses runs of whitespace. It is a heuristic
// reducer for docs and articles, not a renderer.
func reduceHTML(s string) string {
	s = stripElement(s, "script")
	s = stripElement(s, "style")
	s = stripTags(s)
	s = html.UnescapeString(s)
	return collapseWhitespace(s)
}

// stripElement removes every <name ...>...</name> block (contents included),
// case-insensitively. An unterminated opening tag drops the rest of the input.
func stripElement(s, name string) string {
	openTag := "<" + name
	closeTag := "</" + name
	lower := strings.ToLower(s)
	var b strings.Builder
	for {
		start := strings.Index(lower, openTag)
		if start < 0 {
			b.WriteString(s)
			break
		}
		b.WriteString(s[:start])
		endClose := strings.Index(lower[start:], closeTag)
		if endClose < 0 {
			break // unterminated: discard remainder
		}
		// Advance past the closing tag's '>'.
		rest := lower[start+endClose:]
		gt := strings.IndexByte(rest, '>')
		if gt < 0 {
			break
		}
		cut := start + endClose + gt + 1
		s = s[cut:]
		lower = lower[cut:]
	}
	return b.String()
}

// stripTags removes everything from '<' to the matching '>'. Text outside tags
// is preserved; a '<' with no '>' drops the remainder.
func stripTags(s string) string {
	var b strings.Builder
	for {
		lt := strings.IndexByte(s, '<')
		if lt < 0 {
			b.WriteString(s)
			break
		}
		b.WriteString(s[:lt])
		gt := strings.IndexByte(s[lt:], '>')
		if gt < 0 {
			break
		}
		s = s[lt+gt+1:]
	}
	return b.String()
}

// collapseWhitespace replaces every run of whitespace with a single space and
// trims the ends, so reduced HTML reads as flowing text.
func collapseWhitespace(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// isTextual reports whether a media type carries text the model can read:
// any text/*, application/json, application/xml, or +json/+xml suffixes. An
// absent type is treated as text (servers often omit it for plain responses).
func isTextual(mediaType string) bool {
	switch {
	case mediaType == "":
		return true
	case strings.HasPrefix(mediaType, "text/"):
		return true
	case mediaType == "application/json" || mediaType == "application/xml":
		return true
	case strings.HasSuffix(mediaType, "+json") || strings.HasSuffix(mediaType, "+xml"):
		return true
	}
	return false
}
