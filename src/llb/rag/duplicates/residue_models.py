"""Contracts for post-collapse duplicate-residue reports."""

from typing_extensions import TypedDict

from llb.rag.duplicates.models import DuplicateStats


class NearDuplicateBand(TypedDict):
    threshold: float
    pairs: int
    chunks: int
    chunk_share: float
    normalized_pairs: int
    masked_pairs: int


class ResiduePair(TypedDict):
    cosine: float
    same_document: bool
    normalized_equal: bool
    masked_equal: bool
    a: str
    b: str


class ResidueReport(TypedDict):
    n_indexed: int
    store_tier: str
    tiers: dict[str, DuplicateStats]
    bands: list[NearDuplicateBand]
    near_duplicate_examples: list[ResiduePair]
    digit_merge_examples: list[ResiduePair]
