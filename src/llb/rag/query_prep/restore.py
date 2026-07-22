"""Ambiguity-aware restoration constraints for the typos step.

Normalization is lossy: Latin-typed Ukrainian drops the soft sign and apostrophes (`sut` is both
`сут` and `суть`), and homoglyph repair rewrites look-alike characters. Whatever normalization
could not invert exactly reaches the typos step as an out-of-vocabulary token, where a bare
nearest-neighbor scan is free to pick a corpus surface that the user could not have typed -- a
different inflection, or a different short function word one edit away.

This module supplies the three signals that constrain that choice:

- **Surface compatibility** (hard filter). The token carries the noisy form it came from plus the
  normalization edit `kind`; a candidate survives only when re-applying that same lossy transform
  to it reproduces the typed form within `SURFACE_MAX_DISTANCE`. Romanizing `суть` gives back the
  typed `sut`; romanizing `суд` gives `sud`, so `суд` is refused instead of silently swapping the
  word. A token whose noise normalization already fully explains therefore keeps vocabulary
  correction from acting on it at all.
- **Morphology** (ranking). A candidate that is a known Ukrainian word form outranks corpus junk,
  and a candidate that preserves the token's inflectional ending outranks one that rewrites it.
- **Local query context** (ranking). A candidate that shares corpus chunks with the query's other
  in-vocabulary tokens outranks an equally-near candidate that never co-occurs with them.

When those signals still leave two candidates indistinguishable and the token is short enough for
the choice to be a coin flip, the restoration is refused rather than guessed.
"""

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from llb.rag.lexical import tokenize
from llb.rag.query_prep.base import (
    KIND_HOMOGLYPH,
    KIND_TRANSLITERATE,
    STEP_NORMALIZE,
    KnownWordProbe,
    QueryEdit,
)
from llb.rag.query_prep.distance import damerau_levenshtein
from llb.rag.query_prep.normalize import LATIN_TO_UKRAINIAN_CONFUSABLES, cyrillic_to_latin

_LOG = logging.getLogger(__name__)

# A candidate must reproduce the typed surface EXACTLY under the reversed transform. The lossy
# part of romanization (dropped soft sign / apostrophe) is what leaves several corpus surfaces
# compatible at all; anything beyond that is a different word, not a restoration of this one.
SURFACE_MAX_DISTANCE = 0
# Ukrainian inflection is suffixal, so the final characters carry the case/number the user typed.
MORPH_SUFFIX_CHARS = 2
# At or below this length a distance-1 neighborhood is dense with unrelated function words, so an
# otherwise-unresolved tie is refused instead of broken alphabetically, and an insertion/deletion
# candidate is refused outright: at three or four characters, dropping or adding a letter yields a
# DIFFERENT short word (`якв` -> `кв`, `зто` -> `то`) rather than a repair of this one. Only a
# transliteration provenance licenses a short length change, because that is exactly the character
# romanization is known to have dropped (the soft sign and the apostrophe).
AMBIGUOUS_TOKEN_MAX_CHARS = 4
# Rarest in-vocabulary query tokens used as context anchors (rarest first = most discriminative).
CONTEXT_MAX_ANCHORS = 8

# Characters romanization drops entirely, hence never part of a surface comparison.
_SEPARATORS = "'"


@dataclass(frozen=True)
class TokenProvenance:
    """The form a normalized token was actually typed in, and the edit kind that produced it."""

    noisy: str
    kind: str


def normalization_provenance(edits: Iterable[QueryEdit]) -> dict[str, TokenProvenance]:
    """Map each normalize-step replacement back to the single noisy token it came from.

    A replacement two DIFFERENT noisy tokens produced is dropped: its provenance is ambiguous, so
    constraining on either form would be arbitrary.
    """
    origins: dict[str, set[tuple[str, str]]] = {}
    for edit in edits:
        if edit.step == STEP_NORMALIZE:
            origins.setdefault(edit.replacement, set()).add((edit.original, edit.kind))
    resolved: dict[str, TokenProvenance] = {}
    for token, forms in origins.items():
        if len(forms) == 1:
            noisy, kind = next(iter(forms))
            resolved[token] = TokenProvenance(noisy=noisy, kind=kind)
    return resolved


def _fold_homoglyphs(text: str) -> str:
    """Latin look-alikes folded to their Cyrillic twins, so both spellings compare equal."""
    return "".join(LATIN_TO_UKRAINIAN_CONFUSABLES.get(char, char) for char in text)


def surface_distance(candidate: str, provenance: TokenProvenance) -> int:
    """Edit distance between the candidate's re-noised surface and the token as typed.

    Bounded by `SURFACE_MAX_DISTANCE`, so the return value is only meaningful as
    `<= SURFACE_MAX_DISTANCE` (compatible) or above it (a different word). An unknown edit kind
    imposes no constraint.
    """
    if provenance.kind == KIND_TRANSLITERATE:
        left = cyrillic_to_latin(candidate).replace(_SEPARATORS, "")
        right = provenance.noisy.replace(_SEPARATORS, "")
    elif provenance.kind == KIND_HOMOGLYPH:
        left = _fold_homoglyphs(candidate)
        right = _fold_homoglyphs(provenance.noisy)
    else:
        return 0
    return damerau_levenshtein(left, right, SURFACE_MAX_DISTANCE)


@dataclass(frozen=True)
class VocabularyContext:
    """Corpus token -> the chunk ids containing it; the local-context signal's whole index."""

    postings: Mapping[str, "frozenset[int]"]

    @classmethod
    def build(cls, texts: Iterable[str]) -> "VocabularyContext":
        """One tokenization pass over the indexed chunks, in build order (like the vector index)."""
        postings: dict[str, set[int]] = {}
        for chunk_id, text in enumerate(texts):
            for token in tokenize(text):
                postings.setdefault(token, set()).add(chunk_id)
        return cls({token: frozenset(ids) for token, ids in postings.items()})

    @property
    def tokens(self) -> "frozenset[str]":
        """The in-vocabulary token set the typo step protects (identical to `build_vocabulary`)."""
        return frozenset(self.postings)

    def anchors(self, query: str) -> tuple["frozenset[int]", ...]:
        """Chunk sets of the query's rarest in-vocabulary tokens, deterministically ordered.

        A token under correction is out-of-vocabulary by construction, so it never anchors itself.
        """
        known = {token for token in tokenize(query) if token in self.postings}
        ranked = sorted(known, key=lambda token: (len(self.postings[token]), token))
        return tuple(self.postings[token] for token in ranked[:CONTEXT_MAX_ANCHORS])

    def cooccurrence(self, candidate: str, anchors: Sequence["frozenset[int]"]) -> int:
        """How often the candidate shares a chunk with an anchor token (0 when it never does)."""
        chunks = self.postings.get(candidate)
        if not chunks:
            return 0
        return sum(len(chunks & anchor) for anchor in anchors)


def _rank_key(
    token: str,
    distance: int,
    candidate: str,
    known_word: KnownWordProbe | None,
    context: VocabularyContext | None,
    anchors: Sequence["frozenset[int]"],
) -> tuple[int, int, int, int, str]:
    """Ordering key: edit distance, then morphology, then local context, then a stable tie-break."""
    known_penalty = 0 if known_word is None or known_word(candidate) else 1
    suffix_penalty = 0 if candidate[-MORPH_SUFFIX_CHARS:] == token[-MORPH_SUFFIX_CHARS:] else 1
    cooccurrence = context.cooccurrence(candidate, anchors) if context is not None else 0
    return (distance, known_penalty, suffix_penalty, -cooccurrence, candidate)


def select_restoration(
    token: str,
    candidates: Sequence[tuple[int, str]],
    *,
    provenance: TokenProvenance | None = None,
    known_word: KnownWordProbe | None = None,
    context: VocabularyContext | None = None,
    anchors: Sequence["frozenset[int]"] = (),
) -> str | None:
    """The best `(distance, candidate)` under the restoration constraints, or None when unsafe.

    None means "leave the token alone": no candidate is compatible with the form the user typed,
    every candidate would resize a short token, or the survivors are indistinguishable on every
    signal and the token is short enough that an alphabetical tie-break would be a guess.
    """
    compatible = [
        (distance, candidate)
        for distance, candidate in candidates
        if provenance is None or surface_distance(candidate, provenance) <= SURFACE_MAX_DISTANCE
    ]
    short = len(token.replace(_SEPARATORS, "")) <= AMBIGUOUS_TOKEN_MAX_CHARS
    if short and (provenance is None or provenance.kind != KIND_TRANSLITERATE):
        compatible = [pair for pair in compatible if len(pair[1]) == len(token)]
    if not compatible:
        if candidates:
            _LOG.debug(
                "[query-prep] no restoration of %r is compatible with the typed form %r",
                token,
                provenance.noisy if provenance is not None else token,
            )
        return None
    ranked = sorted(
        (
            _rank_key(token, distance, candidate, known_word, context, anchors)
            for distance, candidate in compatible
        )
    )
    best = ranked[0]
    if len(ranked) > 1 and short and ranked[1][:-1] == best[:-1]:
        _LOG.info(
            "[query-prep] ambiguous restoration of %r (%s tie); left unchanged",
            token,
            " / ".join(key[-1] for key in ranked[:2]),
        )
        return None
    return best[-1]
