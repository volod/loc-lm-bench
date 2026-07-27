"""Corpus-calibrated cosine threshold: null-distribution sampling and knob precedence."""

import random

from llb.conflicts.constants import TIER_SEMANTIC
from llb.conflicts.null_distribution import NullDistribution
from llb.conflicts.vectorops import VectorSet


def _semantic_stats(result):
    return next(stat for stat in result.tiers if stat.tier == TIER_SEMANTIC)


def _synthetic_store(n_docs: int = 12, per_doc: int = 12, dim: int = 32, seed: int = 7):
    """Random unit vectors labelled with doc ids -- a corpus with no real duplication."""
    rng = random.Random(seed)
    chunks = []
    rows = []
    for doc in range(n_docs):
        for index in range(per_doc):
            chunks.append(
                {
                    "doc_id": f"doc-{doc}.md",
                    "chunk_id": f"doc-{doc}-{index}",
                    "char_start": 0,
                    "char_end": 100,
                    "text": "word " * 40,
                }
            )
            rows.append([rng.gauss(0.0, 1.0) for _ in range(dim)])
    return VectorSet(rows), chunks


def _distribution(**kwargs) -> NullDistribution:
    defaults = dict(
        similarities=[0.1, 0.2, 0.3, 0.4], n_pairs=4, total_pairs=4, seed=0, exhaustive=True
    )
    return NullDistribution(**{**defaults, **kwargs})
