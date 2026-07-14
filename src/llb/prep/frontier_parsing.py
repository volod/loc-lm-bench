"""Focused frontier parsing implementation."""

import json
import re
from typing import Any

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)  # e.g. per-page `<!-- source_pdf: ... -->`

_LINE_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)  # inline table `<br>` -> whitespace

_DROP_CHARS = set("*_#|•")  # bold/italic/heading/table-pipe/bullet markers

_DASHES = "–—―"  # en/em/horizontal-bar dashes -> ascii hyphen

_SPACE_BEFORE_PUNCT = ".,;:)"  # PDF extraction often inserts a space before these

_QUOTE_FOLD = {**{c: '"' for c in "“”„"}, **{c: "'" for c in "‘’‚‛"}}


class _Normalizer:
    """Char-by-char state machine behind `_normalize` (see its docstring for the rules)."""

    def __init__(self, masked: str) -> None:
        self.masked = masked
        self.out: list[str] = []
        self.index: list[int] = []
        self.prev_space = False
        self.i = 0

    @staticmethod
    def _is_dash(ch: str) -> bool:
        return ch in _DASHES or ch == "-"

    def _append_boundary_space(self, original_index: int) -> None:
        if not self.prev_space and self.out:
            self.out.append(" ")
            self.index.append(original_index)
            self.prev_space = True

    def _consume_dashes(self) -> None:
        """A single dash is kept; a `--` run (table rule / separator) becomes a word boundary."""
        run = 1
        while self.i + run < len(self.masked) and self._is_dash(self.masked[self.i + run]):
            run += 1
        if run >= 2:
            self._append_boundary_space(self.i)
            self.i += run
            return
        self.out.append("-")
        self.index.append(self.i)
        self.prev_space = False
        self.i += 1

    def _consume_char(self, ch: str) -> None:
        if ch in _SPACE_BEFORE_PUNCT and self.prev_space:
            self.out.pop()  # drop the space PDF extraction wedged before this punctuation
            self.index.pop()
        self.out.append(_QUOTE_FOLD.get(ch, ch).casefold())
        self.index.append(self.i)
        self.prev_space = False
        self.i += 1

    def run(self) -> tuple[str, list[int]]:
        while self.i < len(self.masked):
            ch = self.masked[self.i]
            if self._is_dash(ch):
                self._consume_dashes()
            elif ch in _DROP_CHARS:
                self.i += 1
            elif ch.isspace():
                self.i += 1
                self._append_boundary_space(self.i - 1)
            else:
                self._consume_char(ch)
        while self.out and self.out[-1] == " ":  # no trailing boundary space (offsets stay exact)
            self.out.pop()
            self.index.pop()
        return "".join(self.out), self.index


def _normalize(text: str) -> tuple[str, list[int]]:
    """Casefold, collapse whitespace, and drop markdown / PDF-extraction decoration, returning the
    normalized string and, for each normalized char, its index in the ORIGINAL text (so a match
    maps back to EXACT offsets). Dropping decoration lets a clean drafted span ground against
    markdown-decorated corpus text: `**bold**`, `#### headings`, `| table |` pipes, `<br>`, list
    bullets, HTML comments (per-page `<!-- source_pdf ... -->` markers), curly quotes/dashes, table
    `----` rules, and the stray space PDF extraction inserts before `.,;:)` are all normalized away."""
    # Blank out multi-char decoration first (keeping length so offsets stay aligned to the original).
    masked = _HTML_COMMENT.sub(lambda m: " " * len(m.group()), text)
    masked = _LINE_BREAK.sub(lambda m: " " * len(m.group()), masked)
    return _Normalizer(masked).run()


def ground_span(doc_text: str, span_text: str) -> tuple[int, str] | None:
    """Locate `span_text` in `doc_text`. Exact substring first; then a casefold/whitespace-
    normalized match mapped back to EXACT original offsets. Returns (start, exact_doc_substring)
    or None when ungrounded (so a label can never point at text that is not there)."""
    span_text = span_text.strip()
    if not span_text:
        return None
    exact = doc_text.find(span_text)
    if exact >= 0:
        return exact, span_text
    norm_doc, index = _normalize(doc_text)
    norm_span, _ = _normalize(span_text)
    if not norm_span:
        return None
    pos = norm_doc.find(norm_span)
    if pos < 0:
        return None
    start = index[pos]
    end = index[pos + len(norm_span) - 1] + 1  # inclusive last char -> exclusive end
    return start, doc_text[start:end]  # exact original substring (offsets stay exact)


def parse_json_block(text: str) -> Any:
    """Parse JSON from a completion, tolerating a ```json ... ``` fence or surrounding prose."""
    fenced = _JSON_FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = min((i for i in (candidate.find("["), candidate.find("{")) if i >= 0), default=-1)
        end = max(candidate.rfind("]"), candidate.rfind("}"))
        if start >= 0 and end > start:
            return json.loads(candidate[start : end + 1])
        raise
