"""Lexical BM25 index + Ukrainian-aware normalization for hybrid retrieval.

Dense-only cosine loses exact surnames, article/law numbers, codes, abbreviations, and mixed
Ukrainian-English terms to semantically-close distractors. This module adds the lexical side:
a pure-Python BM25 index built over the SAME offset-exact chunk records as the vector index,
with Ukrainian-aware token normalization applied on the LEXICAL side only -- the stored chunk
text is never altered.

Normalization (always on): casefold, apostrophe-variant unification (U+2019 / U+02BC / `'` all
become U+0027), punctuation strip via word-token extraction. Opt-in lemmatization collapses
Ukrainian cases/inflection to lemmas at index AND query time (`pymorphy3` +
`pymorphy3-dicts-uk`; the lemmatizer callable is injectable for tests).

`rrf_fuse` implements weighted reciprocal-rank fusion over the dense + lexical rankings; it is
pure and backend-neutral, so every `VectorIndex` backend gains hybrid identically.
"""

import logging
import math
import re
from collections.abc import Callable, Hashable, Sequence
from typing import Any, TypeVar

_LOG = logging.getLogger(__name__)

# One token -> its lemma (identity when lemmatization is off).
Lemmatizer = Callable[[str], str]

# Apostrophe variants unified to U+0027 so copied and keyboard-typed forms index as one token.
# Every variant must ALSO be in-word for the token regex: a converted PDF writes `зобов’язання`
# with U+2019, and a regex that treats only U+0027 as in-word splits that into `зобов` + `язання`
# -- two half-words that can never match the keyboard-typed twin, no matter what the unification
# does afterwards. The regex is therefore derived from the same variant list.
_APOSTROPHES = "'‘’ʼ`"
_APOSTROPHE_VARIANTS = str.maketrans({variant: "'" for variant in _APOSTROPHES})
# Word tokens: letters/digits plus in-word apostrophes; everything else is punctuation.
_TOKEN_RE = re.compile(rf"[\w{re.escape(_APOSTROPHES)}]+")

# BM25 constants (Robertson/Sparck Jones defaults; recorded in the persisted index meta).
BM25_K1 = 1.5
BM25_B = 0.75
# Standard RRF rank damping constant (Cormack et al. 2009).
RRF_K = 60
# Persisted-postings format AND tokenizer generation. Bump it whenever tokenization changes: the
# postings are tokenizer output, so an index written by an older tokenizer answers a query the new
# one produced with the wrong terms. `LexicalIndex.load` refuses a mismatch (v1 -> v2: apostrophe
# variants became in-word characters).
LEXICAL_INDEX_VERSION = "bm25-uk-v2"

RankedId = TypeVar("RankedId", bound=Hashable)


class _CachedLemmatizer:
    def __init__(self, analyzer: Any):
        self.analyzer = analyzer
        self.cache: dict[str, str] = {}

    def __call__(self, token: str) -> str:
        hit = self.cache.get(token)
        if hit is None:
            parses = self.analyzer.parse(token)
            hit = parses[0].normal_form if parses else token
            self.cache[token] = hit
        return hit


class _CachedWordProbe:
    def __init__(self, analyzer: Any):
        self.analyzer = analyzer
        self.cache: dict[str, bool] = {}

    def __call__(self, token: str) -> bool:
        hit = self.cache.get(token)
        if hit is None:
            hit = bool(self.analyzer.word_is_known(token))
            self.cache[token] = hit
        return hit


def _validate_rrf_inputs(
    rankings: Sequence[Sequence[RankedId]], weights: Sequence[float], k_const: int
) -> float:
    if len(rankings) != len(weights):
        raise ValueError("RRF rankings and weights must have the same length")
    if k_const < 0:
        raise ValueError(f"RRF k constant must be non-negative, got {k_const}")
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError(f"RRF weights must be non-negative, got {list(weights)}")
    total_weight = sum(weights)
    if total_weight <= 0.0:
        raise ValueError("RRF weights must contain at least one positive value")
    return total_weight


def _add_rrf_lane(
    ranking: Sequence[RankedId],
    weight: float,
    lane: int,
    k_const: int,
    scores: dict[RankedId, float],
    tie_keys: dict[RankedId, tuple[int, int, int]],
    encounter: int,
) -> int:
    seen: set[RankedId] = set()
    for rank, item in enumerate(ranking, 1):
        if item in seen:
            continue
        seen.add(item)
        scores[item] = scores.get(item, 0.0) + weight / (k_const + rank)
        if item not in tie_keys:
            tie_keys[item] = (lane, rank, encounter)
            encounter += 1
    return encounter


def normalize_token(token: str) -> str:
    """Casefold + apostrophe unification + edge-apostrophe strip (matching side only).

    Edge apostrophes are stripped AFTER unification, so a quoted `'слово'`, a backticked
    `` `слово` ``, and a typographic `‘слово’` all normalize onto the bare term.
    """
    return token.translate(_APOSTROPHE_VARIANTS).casefold().strip("'")


def tokenize(text: str, lemmatizer: Lemmatizer | None = None) -> list[str]:
    """Normalized word tokens of `text`; `lemmatizer` (when given) maps each to its lemma.

    A token keeps whichever apostrophe variant the source used (see `_TOKEN_RE`) and
    `normalize_token` unifies it, so the index and the query agree on one surface form.
    """
    tokens = [normalize_token(t) for t in _TOKEN_RE.findall(text)]
    tokens = [t for t in tokens if t]
    if lemmatizer is None:
        return tokens
    return [lemmatizer(t) for t in tokens]


def load_uk_lemmatizer() -> Lemmatizer:
    """The pymorphy3 Ukrainian lemmatizer (first-parse normal form), memoized per token."""
    import pymorphy3

    analyzer = pymorphy3.MorphAnalyzer(lang="uk")
    return _CachedLemmatizer(analyzer)


def load_uk_word_probe() -> Callable[[str], bool]:
    """A pymorphy3 "is this a known Ukrainian word form?" probe, memoized per token.

    Backs the opt-in morphology guard of the query-prep `typos` step: a grammatically valid
    inflected query form (`поділяють`, `документами`) is NOT a misspelling and must not be
    "corrected" to a different corpus surface form -- the index+query lemmatization already
    matches valid inflections.
    """
    import pymorphy3

    analyzer = pymorphy3.MorphAnalyzer(lang="uk")
    return _CachedWordProbe(analyzer)


def ukrainian_lemma(token: str) -> str:
    """Normalize and lemmatize a token for morphology-aware topic grouping."""
    global _UK_LEMMATIZER
    if _UK_LEMMATIZER is None:
        _UK_LEMMATIZER = load_uk_lemmatizer()
    return _UK_LEMMATIZER(normalize_token(token))


_UK_LEMMATIZER: Lemmatizer | None = None


def rrf_fuse(
    dense: list[int], lexical: list[int], weight: float, k_const: int = RRF_K
) -> list[tuple[int, float]]:
    """Weighted reciprocal-rank fusion of two ranked id lists, best first.

    score(id) = weight * 1/(k_const + dense_rank) + (1 - weight) * 1/(k_const + lexical_rank),
    with an absent id contributing nothing from that side. `weight`=1 reproduces the dense
    order; `weight`=0 the lexical order. Ties prefer the dense lane and stable encounter order,
    so the fusion is deterministic for any dense backend.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"fusion weight must be within [0, 1], got {weight}")
    return weighted_rrf_fuse([dense, lexical], [weight, 1.0 - weight], k_const=k_const)


def weighted_rrf_fuse(
    rankings: Sequence[Sequence[RankedId]],
    weights: Sequence[float],
    *,
    k_const: int = RRF_K,
) -> list[tuple[RankedId, float]]:
    """Fuse any number of ranked lists using normalized weighted RRF.

    Zero-weight lanes are ignored completely, including candidate membership. This makes a
    weight endpoint an exact passthrough instead of appending zero-score candidates from a
    disabled lane. Ties prefer the earliest lane, then the best rank in that lane, then stable
    encounter order. Duplicate ids inside one lane keep their first rank.
    """
    total_weight = _validate_rrf_inputs(rankings, weights, k_const)
    scores: dict[RankedId, float] = {}
    tie_keys: dict[RankedId, tuple[int, int, int]] = {}
    encounter = 0
    for lane, (ranking, raw_weight) in enumerate(zip(rankings, weights)):
        if raw_weight == 0.0:
            continue
        encounter = _add_rrf_lane(
            ranking,
            raw_weight / total_weight,
            lane,
            k_const,
            scores,
            tie_keys,
            encounter,
        )
    return sorted(scores.items(), key=lambda pair: (-pair[1], tie_keys[pair[0]]))
