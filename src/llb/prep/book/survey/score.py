"""A SMOKE reading of one candidate's output against the synthetic page's transcription.

This is not the pilot's metric harness (`book-recovery-quality-metrics` owns that). It answers
only "did the tool read this page at all, and did it keep the tokens a misreading would corrupt":

- `word_recall`: the share of reference words found in the output, order-free, so a converter
  that reorders columns is not punished here for what the reading-order metric will measure;
- `char_agreement`: one minus the character edit distance over the reference length, in order;
- each dangerous token as `exact`, `hyphen_minus` (the value with an ASCII hyphen or a dash
  where the page prints a minus sign -- a distinct substitution class, not a pass), or `missing`.

Markup is stripped first (Markdown, HTML tags, the few LaTeX commands a converter uses for the
page's Greek letters), so a tool is read on its text, not on its output format. Inside LaTeX math
an ASCII `-` IS the minus sign, so it is read as one there; outside math it stays a hyphen.
"""

import re
import unicodedata
from dataclasses import dataclass

from llb.prep.book.survey.page_content import FixtureTruth

MINUS = "\u2212"
MINUS_LOOKALIKES = ("-", "\u2013", "\u2014", "\u2010", "\u2011")  # hyphen, en, em, two hyphens
EXACT = "exact"
HYPHEN_MINUS = "hyphen_minus"
MISSING = "missing"

_LATEX_SYMBOLS = {
    r"\alpha": "\u03b1",
    r"\beta": "\u03b2",
    r"\Delta": "\u0394",
    r"\mu": "\u03bc",
    r"\%": "%",
    r"\times": "\u00d7",
    r"\cdot": "\u00b7",
    r"\ln": "ln",
    r"\left": "",
    r"\right": "",
    r"\mathrm": "",
    r"\text": "",
    r"\quad": " ",
    r"\,": " ",
    "~": " ",
}
_MATH = re.compile(r"\$\$.+?\$\$|\$.+?\$|\\\(.+?\\\)|\\\[.+?\\\]", re.DOTALL)
_DELIMITERS = (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]"))
_EQ_TAG = re.compile(r"\\tag\s*\{([^}]*)\}")
_EQ_NUMBER = re.compile(r"\((\d+)" + MINUS + r"(\d+)\)")  # "(3-4)" inside math is a label
_SCRIPT = re.compile(r"\s*[_^]\s*|[{}]")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_TABLE_RULE = re.compile(r":?-{3,}:?")
_MARKUP = re.compile(r"[#*`|_^{}$\\]|!\[[^\]]*\]\([^)]*\)")
_SPACE = re.compile(r"\s+")
_WORD_EDGE = re.compile("^[^\\w%\u2212]+|[^\\w%\u2212]+$")


@dataclass(frozen=True)
class SmokeScore:
    word_recall: float
    char_agreement: float
    dangerous: dict[str, str]

    @property
    def dangerous_exact(self) -> int:
        return sum(1 for verdict in self.dangerous.values() if verdict == EXACT)


def _symbols(text: str) -> str:
    for command, replacement in _LATEX_SYMBOLS.items():
        # A command's terminating space is part of the command (`\Delta v` reads "Delta-v", not "Delta v").
        pattern = re.escape(command) + (r"(?![A-Za-z]) ?" if command[-1].isalpha() else "")
        text = re.sub(pattern, replacement, text)
    return text


def _math(span: "re.Match[str]") -> str:
    """A math span as it typesets: `-` is a minus, scripts join their base, labels stay."""
    inner = span.group(0).strip()
    stripped = True
    while stripped:  # a converter may nest them: `$$\[ ... \]$$`
        stripped = False
        for opener, closer in _DELIMITERS:
            if inner.startswith(opener) and inner.endswith(closer) and len(inner) > 2 * len(opener):
                inner, stripped = inner[len(opener) : -len(closer)].strip(), True
                break
    tags = _EQ_TAG.findall(inner)
    inner = _EQ_TAG.sub("", inner).replace("-", MINUS)
    inner = _SCRIPT.sub("", _symbols(inner))
    inner = _EQ_NUMBER.sub(r"(\1-\2)", inner)
    return inner + "".join(f" ({tag})" for tag in tags)


def plain_text(text: str) -> str:
    """The output's characters without markup, NFC-normalised, whitespace collapsed."""
    text = _COMMENT.sub(" ", text)
    text = _MATH.sub(_math, text)
    text = _symbols(text)
    text = _TAG.sub(" ", text)
    text = _TABLE_RULE.sub(" ", text)
    text = _MARKUP.sub(" ", text)
    text = unicodedata.normalize("NFC", text).replace("\u00b5", "\u03bc")  # micro sign -> Greek mu
    return _SPACE.sub(" ", text).strip()


def _words(text: str) -> list[str]:
    words = (_WORD_EDGE.sub("", word).lower() for word in text.split())
    return [word for word in words if word]


def _word_recall(reference: str, hypothesis: str) -> float:
    pool: dict[str, int] = {}
    for word in _words(hypothesis):
        pool[word] = pool.get(word, 0) + 1
    ref = _words(reference)
    found = 0
    for word in ref:
        if pool.get(word, 0) > 0:
            pool[word] -= 1
            found += 1
    return found / len(ref) if ref else 0.0


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, two-row dynamic programme."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            )
        previous = current
    return previous[-1]


def _token_verdict(token: str, hypothesis: str) -> str:
    if token in hypothesis:
        return EXACT
    if MINUS in token:
        for lookalike in MINUS_LOOKALIKES:
            if token.replace(MINUS, lookalike) in hypothesis:
                return HYPHEN_MINUS
    return MISSING


def score_reading(truth: FixtureTruth, output: str) -> SmokeScore:
    """Score one candidate's reading of the synthetic page."""
    reference = plain_text(truth.reading_text())
    hypothesis = plain_text(output)
    distance = edit_distance(reference, hypothesis)
    agreement = max(0.0, 1.0 - distance / len(reference)) if reference else 0.0
    return SmokeScore(
        word_recall=round(_word_recall(reference, hypothesis), 3),
        char_agreement=round(agreement, 3),
        dangerous={token: _token_verdict(token, hypothesis) for token in truth.dangerous_tokens},
    )
