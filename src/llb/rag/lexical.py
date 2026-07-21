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

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Sequence
from pathlib import Path
from typing import TypeVar

_LOG = logging.getLogger(__name__)

# One token -> its lemma (identity when lemmatization is off).
Lemmatizer = Callable[[str], str]

# Apostrophe variants unified to U+0027 so copied and keyboard-typed forms index as one token.
_APOSTROPHE_VARIANTS = str.maketrans({"‘": "'", "’": "'", "ʼ": "'", "`": "'"})
# Word tokens: letters/digits plus in-word apostrophes; everything else is punctuation.
_TOKEN_RE = re.compile(r"[\w']+")

# BM25 constants (Robertson/Sparck Jones defaults; recorded in the persisted index meta).
BM25_K1 = 1.5
BM25_B = 0.75
# Standard RRF rank damping constant (Cormack et al. 2009).
RRF_K = 60
LEXICAL_INDEX_VERSION = "bm25-uk-v1"

RankedId = TypeVar("RankedId", bound=Hashable)


def normalize_token(token: str) -> str:
    """Casefold + apostrophe unification + edge-apostrophe strip (matching side only)."""
    return token.translate(_APOSTROPHE_VARIANTS).casefold().strip("'")


def tokenize(text: str, lemmatizer: Lemmatizer | None = None) -> list[str]:
    """Normalized word tokens of `text`; `lemmatizer` (when given) maps each to its lemma."""
    tokens = [normalize_token(t) for t in _TOKEN_RE.findall(text)]
    tokens = [t for t in tokens if t]
    if lemmatizer is None:
        return tokens
    return [lemmatizer(t) for t in tokens]


def load_uk_lemmatizer() -> Lemmatizer:
    """The pymorphy3 Ukrainian lemmatizer (first-parse normal form), memoized per token."""
    import pymorphy3

    analyzer = pymorphy3.MorphAnalyzer(lang="uk")
    cache: dict[str, str] = {}

    def lemma(token: str) -> str:
        hit = cache.get(token)
        if hit is None:
            parses = analyzer.parse(token)
            hit = parses[0].normal_form if parses else token
            cache[token] = hit
        return hit

    return lemma


def load_uk_word_probe() -> Callable[[str], bool]:
    """A pymorphy3 "is this a known Ukrainian word form?" probe, memoized per token.

    Backs the opt-in morphology guard of the query-prep `typos` step: a grammatically valid
    inflected query form (`поділяють`, `документами`) is NOT a misspelling and must not be
    "corrected" to a different corpus surface form -- the index+query lemmatization already
    matches valid inflections.
    """
    import pymorphy3

    analyzer = pymorphy3.MorphAnalyzer(lang="uk")
    cache: dict[str, bool] = {}

    def known(token: str) -> bool:
        hit = cache.get(token)
        if hit is None:
            hit = bool(analyzer.word_is_known(token))
            cache[token] = hit
        return hit

    return known


def ukrainian_lemma(token: str) -> str:
    """Normalize and lemmatize a token for morphology-aware topic grouping."""
    global _UK_LEMMATIZER
    if _UK_LEMMATIZER is None:
        _UK_LEMMATIZER = load_uk_lemmatizer()
    return _UK_LEMMATIZER(normalize_token(token))


_UK_LEMMATIZER: Lemmatizer | None = None


class LexicalIndex:
    """Deterministic pure-Python BM25 over chunk texts (build-order ids, like the vector index)."""

    def __init__(
        self,
        postings: dict[str, list[tuple[int, int]]],
        doc_lengths: list[int],
        lemmatize: bool,
        lemmatizer: Lemmatizer | None = None,
    ):
        self.postings = postings  # term -> [(chunk_ordinal, term_frequency)] sorted by ordinal
        self.doc_lengths = doc_lengths
        self.lemmatize = lemmatize
        self._lemmatizer = lemmatizer
        self.n_docs = len(doc_lengths)
        self.avg_doc_len = (sum(doc_lengths) / self.n_docs) if self.n_docs else 0.0

    @classmethod
    def build(
        cls, texts: Iterable[str], lemmatize: bool = False, lemmatizer: Lemmatizer | None = None
    ) -> "LexicalIndex":
        """Index `texts` in order. With `lemmatize`, tokens collapse to lemmas at index time
        (query tokens are lemmatized identically in `search`); the texts themselves are never
        modified. `lemmatizer` injects a fake for tests; default is the pymorphy3 Ukrainian one.
        """
        if lemmatize and lemmatizer is None:
            lemmatizer = load_uk_lemmatizer()
        postings: dict[str, list[tuple[int, int]]] = {}
        doc_lengths: list[int] = []
        for ordinal, text in enumerate(texts):
            tokens = tokenize(text, lemmatizer if lemmatize else None)
            doc_lengths.append(len(tokens))
            for term, tf in sorted(Counter(tokens).items()):
                postings.setdefault(term, []).append((ordinal, tf))
        return cls(postings, doc_lengths, lemmatize, lemmatizer)

    def _query_lemmatizer(self) -> Lemmatizer | None:
        if not self.lemmatize:
            return None
        if self._lemmatizer is None:  # loaded index: resolve the real lemmatizer lazily
            self._lemmatizer = load_uk_lemmatizer()
        return self._lemmatizer

    def search(
        self, query: str, k: int, allowed: set[int] | None = None
    ) -> list[tuple[int, float]]:
        """Top-k `(chunk_ordinal, bm25_score)` for `query`, best first, ties broken by ordinal.

        `allowed` restricts candidates to those ordinals (the metadata-filter seam applies
        BEFORE fusion); only chunks matching at least one query term are returned.
        """
        if k < 1 or not self.n_docs:
            return []
        scores: dict[int, float] = {}
        for term in tokenize(query, self._query_lemmatizer()):
            entries = self.postings.get(term)
            if not entries:
                continue
            idf = math.log(1.0 + (self.n_docs - len(entries) + 0.5) / (len(entries) + 0.5))
            for ordinal, tf in entries:
                if allowed is not None and ordinal not in allowed:
                    continue
                norm = 1.0 - BM25_B + BM25_B * (self.doc_lengths[ordinal] / self.avg_doc_len)
                scores[ordinal] = scores.get(ordinal, 0.0) + idf * (
                    tf * (BM25_K1 + 1.0) / (tf + BM25_K1 * norm)
                )
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:k]

    def save(self, path: Path | str) -> None:
        payload = {
            "version": LEXICAL_INDEX_VERSION,
            "lemmatize": self.lemmatize,
            "doc_lengths": self.doc_lengths,
            "postings": {term: entries for term, entries in sorted(self.postings.items())},
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "LexicalIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        postings = {
            term: [(int(ordinal), int(tf)) for ordinal, tf in entries]
            for term, entries in payload["postings"].items()
        }
        return cls(postings, [int(n) for n in payload["doc_lengths"]], bool(payload["lemmatize"]))


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
    if len(rankings) != len(weights):
        raise ValueError("RRF rankings and weights must have the same length")
    if k_const < 0:
        raise ValueError(f"RRF k constant must be non-negative, got {k_const}")
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError(f"RRF weights must be non-negative, got {list(weights)}")
    total_weight = sum(weights)
    if total_weight <= 0.0:
        raise ValueError("RRF weights must contain at least one positive value")

    scores: dict[RankedId, float] = {}
    tie_keys: dict[RankedId, tuple[int, int, int]] = {}
    encounter = 0
    for lane, (ranking, raw_weight) in enumerate(zip(rankings, weights)):
        if raw_weight == 0.0:
            continue
        weight = raw_weight / total_weight
        seen: set[RankedId] = set()
        for rank, item in enumerate(ranking, 1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + weight / (k_const + rank)
            if item not in tie_keys:
                tie_keys[item] = (lane, rank, encounter)
                encounter += 1
    return sorted(scores.items(), key=lambda pair: (-pair[1], tie_keys[pair[0]]))
