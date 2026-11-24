"""Term-by-term formula fidelity.

Fidelity is the share of the reference's top-level terms (each side of ``=``, then each top-level
``+`` or binary ``-``) that the hypothesis still has. A term dropped from an equation that still
parses lowers that share. A string that does not parse is flagged and is not given a fidelity
number. Canonical spelling and the parse check live in the sibling modules.
"""

import re
from dataclasses import dataclass

from llb.prep.book.metrics.latex_norm import (
    LatexError,
    canonicalize,
    looks_like_latex,
    plain_form,
    script_tokens,
)

__all__ = [
    "FormulaRead",
    "LatexError",
    "canonicalize",
    "formula_fidelity",
    "latex_parses",
    "looks_like_latex",
    "plain_form",
    "script_tokens",
    "terms_of",
]


@dataclass(frozen=True)
class FormulaRead:
    """One formula after normalization. ``fidelity`` is None when the reading did not parse."""

    parse_valid: bool
    fidelity: float | None
    reference_terms: int
    matched_terms: int


def latex_parses(source: str) -> bool:
    """True when the canonical form is a complete expression."""
    from llb.prep.book.metrics.latex_parse import consume

    try:
        consume(canonicalize(source))
    except LatexError:
        return False
    return True


def terms_of(source: str) -> list[str]:
    """Top-level terms of a canonical formula. Empty when the string does not parse."""
    canonical = canonicalize(source)
    if not latex_parses(canonical):
        return []
    return [term for side in _split_top(canonical, "=") for term in _split_additive(side)]


def formula_fidelity(
    reference_latex: str, hypothesis: str, *, hypothesis_is_latex: bool
) -> FormulaRead:
    """Term overlap of one reference formula against one reading."""
    if not latex_parses(reference_latex):
        return FormulaRead(False, None, 0, 0)
    reference_terms = terms_of(reference_latex)
    if hypothesis_is_latex:
        if not latex_parses(hypothesis):
            return FormulaRead(False, None, len(reference_terms), 0)
        matched = _overlap(reference_terms, terms_of(hypothesis))
        return FormulaRead(
            True, _ratio(matched, len(reference_terms)), len(reference_terms), matched
        )
    haystack = plain_form(hypothesis).replace("\u2212", "-")
    matched = 0
    for needle in sorted(map(plain_form, reference_terms), key=len, reverse=True):
        if needle:
            haystack, hits = re.subn(r"(?<!-)" + re.escape(needle), "\0", haystack, count=1)
            matched += hits
    return FormulaRead(True, _ratio(matched, len(reference_terms)), len(reference_terms), matched)


def _ratio(matched: int, total: int) -> float | None:
    if total == 0:
        return None
    return matched / total


def _overlap(reference: list[str], hypothesis: list[str]) -> int:
    pool = list(hypothesis)
    found = 0
    for term in reference:
        if term in pool:
            pool.remove(term)
            found += 1
    return found


def _split_top(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "{(":
            depth += 1
        elif char in "})" and depth:
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return [part for part in parts if part]


def _split_additive(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "{(":
            depth += 1
        elif char in "})" and depth:
            depth -= 1
        elif depth == 0 and char in "+-" and index > start and _binary(text, index):
            parts.append(text[start:index])
            start = index if char == "-" else index + 1
    parts.append(text[start:])
    return [part for part in parts if part]


def _binary(text: str, index: int) -> bool:
    previous = text[index - 1]
    return previous not in "{(=+-^_"
