"""Obtain a corpus's null distribution of chunk-pair similarity: enumerate it, or sample it.

Split from `null_calibration` along the seam between HOW the distribution is measured and WHAT
threshold is decided from it. Enumeration is preferred wherever the pair space allows, because
sampling puts a `1/N` floor under the estimable tail (see `MAX_EXHAUSTIVE_PAIRS`).
"""

import logging
import random

from llb.conflicts.null_distribution import (
    DEFAULT_NULL_SAMPLE_PAIRS,
    DEFAULT_NULL_SEED,
    MAX_EXHAUSTIVE_PAIRS,
    MIN_NULL_PAIRS,
    NullDistribution,
)
from llb.conflicts.vectorops import VectorSet
from llb.core.contracts.rag import ChunkRecord

_LOG = logging.getLogger(__name__)

_MAX_SAMPLE_ATTEMPTS_FACTOR = 4
"""Rejection-sampling budget: cross-document pairs can be a small share of a lopsided corpus."""


def _exhaustive_similarities(
    vectors: VectorSet,
    chunks: list[ChunkRecord],
    ordinals: list[int],
) -> list[float] | None:
    """Every comparable cross-document similarity, or None when that is not worth materializing."""
    if total_cross_document_pairs(chunks, ordinals) > MAX_EXHAUSTIVE_PAIRS:
        return None
    doc_index: dict[str, int] = {}
    codes = []
    for ordinal in ordinals:
        doc_id = chunks[ordinal]["doc_id"]
        codes.append(doc_index.setdefault(doc_id, len(doc_index)))
    return vectors.cross_group_similarities(ordinals, codes)


def total_cross_document_pairs(chunks: list[ChunkRecord], ordinals: list[int]) -> int:
    """Comparable pairs the scan will actually consider: all pairs minus same-document ones.

    This is the denominator the candidate budget divides by, so it must match the tier's own
    filtering exactly -- counting same-document pairs would understate the resolved threshold.
    """
    counts: dict[str, int] = {}
    for ordinal in ordinals:
        doc_id = chunks[ordinal]["doc_id"]
        counts[doc_id] = counts.get(doc_id, 0) + 1
    total = len(ordinals) * (len(ordinals) - 1) // 2
    return total - sum(count * (count - 1) // 2 for count in counts.values())


def _cross_document_sample(
    chunks: list[ChunkRecord],
    ordinals: list[int],
    sample_pairs: int,
    seed: int,
) -> tuple[list[tuple[int, int]], bool]:
    """Random distinct cross-document ordinal pairs, or every such pair when they are few.

    Enumerating exhaustively for a small corpus is not an optimization -- it removes sampling
    error entirely, so a fixture-sized corpus resolves one exact threshold rather than a noisy
    one that moves with the seed.
    """
    if len(ordinals) * (len(ordinals) - 1) // 2 <= sample_pairs:
        exhaustive = [
            (ordinals[i], ordinals[j])
            for i in range(len(ordinals))
            for j in range(i + 1, len(ordinals))
            if chunks[ordinals[i]]["doc_id"] != chunks[ordinals[j]]["doc_id"]
        ]
        return exhaustive, True

    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    attempts = 0
    budget = sample_pairs * _MAX_SAMPLE_ATTEMPTS_FACTOR
    while len(seen) < sample_pairs and attempts < budget:
        attempts += 1
        left = rng.choice(ordinals)
        right = rng.choice(ordinals)
        if left == right or chunks[left]["doc_id"] == chunks[right]["doc_id"]:
            continue
        seen.add((left, right) if left < right else (right, left))
    return sorted(seen), False


def estimate_null_distribution(
    vectors: VectorSet,
    chunks: list[ChunkRecord],
    allowed: set[int],
    *,
    sample_pairs: int = DEFAULT_NULL_SAMPLE_PAIRS,
    seed: int = DEFAULT_NULL_SEED,
) -> NullDistribution | None:
    """Sample the corpus's own cross-document pair similarities, or None when too few exist.

    Sampling runs in whatever space the caller passes -- centered or raw -- because a quantile of
    the wrong space would resolve to a threshold the pair scan never sees.
    """
    ordinals = sorted(allowed)
    if len(ordinals) < 2:
        return None
    total = total_cross_document_pairs(chunks, ordinals)
    values = _exhaustive_similarities(vectors, chunks, ordinals)
    if values is not None:
        if len(values) < MIN_NULL_PAIRS:
            _LOG.warning(
                "[conflicts] threshold calibration needs at least %d comparable cross-document "
                "pairs; this corpus has %d, so the fixed threshold is kept",
                MIN_NULL_PAIRS,
                len(values),
            )
            return None
        return NullDistribution(
            similarities=sorted(values),
            n_pairs=len(values),
            total_pairs=total,
            seed=seed,
            exhaustive=True,
        )
    pairs, _ = _cross_document_sample(chunks, ordinals, sample_pairs, seed)
    if len(pairs) < MIN_NULL_PAIRS:
        _LOG.warning(
            "[conflicts] threshold calibration needs at least %d comparable cross-document "
            "pairs; this corpus has %d, so the fixed threshold is kept",
            MIN_NULL_PAIRS,
            len(pairs),
        )
        return None
    return NullDistribution(
        similarities=sorted(vectors.pair_similarities(pairs)),
        n_pairs=len(pairs),
        total_pairs=total,
        seed=seed,
        exhaustive=False,
    )
