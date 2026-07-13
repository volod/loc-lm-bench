"""Typos step: deterministic corpus-vocabulary typo tolerance over a bounded edit distance.

A query token ABSENT from the indexed corpus vocabulary is corrected to its nearest in-vocabulary
token within Damerau-Levenshtein (OSA) distance 1 (2 for tokens over 8 chars); a token the corpus
already contains is NEVER altered.
"""

import logging
import re
from collections.abc import Iterable

from llb.rag.lexical import normalize_token, tokenize
from llb.rag.query_prep.base import STEP_TYPOS, KnownWordProbe, QueryEdit

_LOG = logging.getLogger(__name__)

# Typo-tolerance thresholds (Damerau-Levenshtein / OSA edit distance). A longer token tolerates a
# second edit because a single transposition + substitution is common in long Ukrainian inflections.
TYPO_MAX_DISTANCE_SHORT = 1
TYPO_LONG_TOKEN_CHARS = 8
TYPO_MAX_DISTANCE_LONG = 2


def damerau_levenshtein(a: str, b: str, max_distance: int | None = None) -> int:
    """Optimal string alignment (Damerau-Levenshtein with adjacent transpositions), bounded.

    Returns `max_distance + 1` early once the running minimum exceeds `max_distance`, so the
    per-token vocabulary scan stays cheap. Exact distance when `max_distance` is None.
    """
    la, lb = len(a), len(b)
    if max_distance is not None and abs(la - lb) > max_distance:
        return max_distance + 1
    prev2: list[int] = []
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_best = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            best = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                best = min(best, prev2[j - 2] + 1)
            cur[j] = best
            row_best = min(row_best, best)
        if max_distance is not None and row_best > max_distance:
            return max_distance + 1
        prev2, prev = prev, cur
    return prev[lb]


def _typo_budget(token: str) -> int:
    return TYPO_MAX_DISTANCE_LONG if len(token) > TYPO_LONG_TOKEN_CHARS else TYPO_MAX_DISTANCE_SHORT


def nearest_vocab_token(token: str, vocabulary: "frozenset[str]", max_distance: int) -> str | None:
    """The closest in-vocabulary token within `max_distance` edits, else None.

    Deterministic: ties break on smaller edit distance, then lexicographically smaller token, so
    the result is independent of the (unordered) vocabulary iteration order.
    """
    best: tuple[int, str] | None = None
    for candidate in vocabulary:
        if abs(len(candidate) - len(token)) > max_distance:
            continue
        distance = damerau_levenshtein(token, candidate, max_distance)
        if distance <= max_distance:
            key = (distance, candidate)
            if best is None or key < best:
                best = key
    return best[1] if best is not None else None


def apply_typos(
    query: str,
    vocabulary: "frozenset[str]",
    known_word: KnownWordProbe | None = None,
) -> tuple[str, list[QueryEdit]]:
    """Correct out-of-vocabulary word tokens to their nearest in-vocabulary neighbor.

    An in-vocabulary token is never altered. Purely numeric tokens are left untouched so an
    article/law number or code is never "corrected" into a different one. Every correction is
    recorded and logged.

    `known_word` is the opt-in morphology guard (morphology-aware-typo-guard): an OOV token the
    probe recognizes as a valid Ukrainian word form is a grammatical inflection, not a typo, so
    it is left for index+query lemmatization to match instead of being edit-distance "corrected"
    to a different corpus surface form. Genuine misspellings stay unknown to the probe and are
    still corrected.
    """
    from llb.rag.lexical import _TOKEN_RE

    edits: list[QueryEdit] = []

    def _replace(match: "re.Match[str]") -> str:
        raw = match.group(0)
        token = normalize_token(raw)
        if not token or token.isdigit() or token in vocabulary:
            return raw
        if known_word is not None and known_word(token):
            _LOG.debug("[query-prep] typo guard: %r is a known word form; left unchanged", token)
            return raw
        correction = nearest_vocab_token(token, vocabulary, _typo_budget(token))
        if correction is None or correction == token:
            return raw
        edits.append(QueryEdit(STEP_TYPOS, "typo", original=token, replacement=correction))
        _LOG.info("[query-prep] typo %r -> %r (corpus vocabulary)", token, correction)
        return correction

    processed = _TOKEN_RE.sub(_replace, query)
    return processed, edits


def build_vocabulary(texts: Iterable[str]) -> "frozenset[str]":
    """The set of normalized corpus tokens (the in-vocabulary set the typo step protects)."""
    vocab: set[str] = set()
    for text in texts:
        vocab.update(tokenize(text))
    return frozenset(vocab)
