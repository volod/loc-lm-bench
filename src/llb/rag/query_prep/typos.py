"""Typos step: deterministic corpus-vocabulary typo tolerance over a bounded edit distance.

A query token ABSENT from the indexed corpus vocabulary is corrected to its nearest in-vocabulary
token within Damerau-Levenshtein (OSA) distance 1 (2 for tokens over 8 chars); a token the corpus
already contains is NEVER altered. Which of the near candidates is chosen -- and whether choosing
at all is safe -- is decided by the restoration constraints in `restore`.
"""

import logging
import re
from collections.abc import Iterable, Mapping, Sequence

from llb.rag.lexical import normalize_token, tokenize
from llb.rag.query_prep.base import KIND_TYPO, STEP_TYPOS, KnownWordProbe, QueryEdit
from llb.rag.query_prep.distance import damerau_levenshtein
from llb.rag.query_prep.restore import (
    TokenProvenance,
    VocabularyContext,
    select_restoration,
)

_LOG = logging.getLogger(__name__)

# Typo-tolerance thresholds (Damerau-Levenshtein / OSA edit distance). A longer token tolerates a
# second edit because a single transposition + substitution is common in long Ukrainian inflections.
TYPO_MAX_DISTANCE_SHORT = 1
TYPO_LONG_TOKEN_CHARS = 8
TYPO_MAX_DISTANCE_LONG = 2
TYPO_MIN_TOKEN_CHARS = 3


def _typo_budget(token: str) -> int:
    return TYPO_MAX_DISTANCE_LONG if len(token) > TYPO_LONG_TOKEN_CHARS else TYPO_MAX_DISTANCE_SHORT


def vocab_candidates(
    token: str, vocabulary: "frozenset[str]", max_distance: int
) -> list[tuple[int, str]]:
    """Every in-vocabulary token within `max_distance` edits, as `(distance, candidate)`.

    Deterministic and independent of the (unordered) vocabulary iteration order: the list is
    sorted by distance, then lexicographically. Candidates never cross the alphabetic/numeric
    kind boundary, so an article number is never a candidate for a word (or the reverse).
    """
    alphabetic = token.replace("'", "").isalpha()
    found: list[tuple[int, str]] = []
    for candidate in vocabulary:
        if candidate.replace("'", "").isalpha() != alphabetic:
            continue
        if abs(len(candidate) - len(token)) > max_distance:
            continue
        distance = damerau_levenshtein(token, candidate, max_distance)
        if distance <= max_distance:
            found.append((distance, candidate))
    return sorted(found)


def nearest_vocab_token(token: str, vocabulary: "frozenset[str]", max_distance: int) -> str | None:
    """The closest in-vocabulary token within `max_distance` edits, else None.

    The UNCONSTRAINED nearest neighbor: ties break on smaller edit distance, then on the
    lexicographically smaller token. `apply_typos` ranks the same candidate set through the
    restoration constraints instead of taking this first entry blindly.
    """
    candidates = vocab_candidates(token, vocabulary, max_distance)
    return candidates[0][1] if candidates else None


def apply_typos(
    query: str,
    vocabulary: "frozenset[str]",
    known_word: KnownWordProbe | None = None,
    provenance: Mapping[str, TokenProvenance] | None = None,
    context: VocabularyContext | None = None,
) -> tuple[str, list[QueryEdit]]:
    """Correct out-of-vocabulary word tokens to their nearest SAFE in-vocabulary neighbor.

    An in-vocabulary token is never altered. Purely numeric tokens are left untouched so an
    article/law number or code is never "corrected" into a different one. Every correction is
    recorded and logged.

    `known_word` is the opt-in morphology probe (morphology-aware-typo-guard). As a GUARD, an OOV
    token it recognizes as a valid Ukrainian word form is a grammatical inflection, not a typo, so
    it is left for index+query lemmatization to match instead of being edit-distance "corrected"
    to a different corpus surface form; genuine misspellings stay unknown to the probe and are
    still corrected. As a RANKING signal it additionally prefers candidates that are themselves
    real word forms.

    `provenance` carries each normalized token back to the noisy form the user actually typed
    (`restore.normalization_provenance` over the normalize step's edits) and `context` is the
    corpus co-occurrence index; both constrain WHICH candidate may be chosen, and either can veto
    the correction entirely. See `restore` for the constraint order.
    """
    from llb.rag.lexical import _TOKEN_RE

    edits: list[QueryEdit] = []
    origins = provenance or {}
    anchors: Sequence["frozenset[int]"] = context.anchors(query) if context is not None else ()

    def _replace(match: "re.Match[str]") -> str:
        raw = match.group(0)
        token = normalize_token(raw)
        if (
            not token
            or token.isdigit()
            or len(token.replace("'", "")) < TYPO_MIN_TOKEN_CHARS
            or token in vocabulary
        ):
            return raw
        if known_word is not None and known_word(token):
            _LOG.debug("[query-prep] typo guard: %r is a known word form; left unchanged", token)
            return raw
        correction = select_restoration(
            token,
            vocab_candidates(token, vocabulary, _typo_budget(token)),
            provenance=origins.get(token),
            known_word=known_word,
            context=context,
            anchors=anchors,
        )
        if correction is None or correction == token:
            return raw
        edits.append(QueryEdit(STEP_TYPOS, KIND_TYPO, original=token, replacement=correction))
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
