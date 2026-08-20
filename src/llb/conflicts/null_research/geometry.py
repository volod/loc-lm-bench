"""Vector geometry shared by the independent-null research candidates."""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import random
import re

from llb.conflicts.constants import MIN_CENTERING_VECTORS, MIN_CLAIM_TOKENS
from llb.conflicts.corpus import load_corpus_docs
from llb.conflicts.tiers.semantic_filter import select_content_chunks
from llb.conflicts.store_access import StoreView
from llb.conflicts.semantic_tree.vector_math import Vector, dot
from llb.conflicts.semantic_tree.vectorops import VectorSet
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord

EmbedTexts = Callable[[list[str]], object]
DocPair = tuple[str, str]

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class CorpusGeometry:
    """One corpus in exactly the filtered, centered space used by the semantic tier."""

    name: str
    corpus_root: str
    store_dir: str
    embedding_model: str
    corpus_fingerprint: str
    chunks: list[ChunkRecord]
    raw_vectors: VectorSet
    vectors: VectorSet
    allowed: list[int]
    observed_similarities: list[float]
    document_maxima: dict[DocPair, float]
    center_mean: Vector | None
    excluded: dict[str, int]

    @property
    def centered(self) -> bool:
        return self.center_mean is not None


def _document_maxima(
    vectors: VectorSet, chunks: list[ChunkRecord], allowed: list[int]
) -> dict[DocPair, float]:
    by_doc: dict[str, list[int]] = defaultdict(list)
    for ordinal in allowed:
        by_doc[chunks[ordinal]["doc_id"]].append(ordinal)
    maxima: dict[DocPair, float] = {}
    doc_ids = sorted(by_doc)
    for position, left_doc in enumerate(doc_ids):
        for right_doc in doc_ids[position + 1 :]:
            values = vectors.cross_similarities(vectors, by_doc[left_doc], by_doc[right_doc])
            if values:
                maxima[(left_doc, right_doc)] = max(values)
    return maxima


def prepare_geometry(
    name: str,
    corpus_root: Path | str,
    store: StoreView,
    *,
    min_claim_tokens: int = MIN_CLAIM_TOKENS,
    center_vectors: bool = True,
) -> CorpusGeometry:
    """Reconstruct the audit's comparable population without running candidate generation."""
    docs = load_corpus_docs(corpus_root)
    body_offsets = {doc.doc_id: doc.body_offset for doc in docs}
    selection = select_content_chunks(store.chunks, body_offsets, min_tokens=min_claim_tokens)
    allowed = sorted(selection.ordinals)
    centered = center_vectors and len(store.vectors) >= MIN_CENTERING_VECTORS
    mean = store.vectors.mean_vector() if centered else None
    vectors = store.vectors.centered(mean) if mean is not None else store.vectors
    doc_codes: dict[str, int] = {}
    groups = [
        doc_codes.setdefault(store.chunks[index]["doc_id"], len(doc_codes)) for index in allowed
    ]
    observed = sorted(vectors.cross_group_similarities(allowed, groups))
    return CorpusGeometry(
        name=name,
        corpus_root=str(corpus_root),
        store_dir=str(store.index_dir),
        embedding_model=store.embedding_model,
        corpus_fingerprint=str(store.meta.get("corpus_fingerprint", "")),
        chunks=store.chunks,
        raw_vectors=store.vectors,
        vectors=vectors,
        allowed=allowed,
        observed_similarities=observed,
        document_maxima=_document_maxima(vectors, store.chunks, allowed),
        center_mean=mean,
        excluded=selection.stats(),
    )


def cross_corpus_null(target: CorpusGeometry, reference: CorpusGeometry) -> list[float]:
    """Target-reference similarities transformed by the target corpus's centering mean."""
    if target.raw_vectors.dim != reference.raw_vectors.dim:
        raise ValueError("target and reference stores use different vector dimensions")
    target_docs = {target.chunks[index]["doc_id"] for index in target.allowed}
    reference_docs = {reference.chunks[index]["doc_id"] for index in reference.allowed}
    overlap = target_docs & reference_docs
    if overlap:
        preview = ", ".join(sorted(overlap)[:3])
        raise ValueError(
            "an independent cross-corpus null requires disjoint document ids; overlap: " + preview
        )
    reference_vectors = (
        reference.raw_vectors.centered(target.center_mean)
        if target.center_mean is not None
        else reference.raw_vectors
    )
    return sorted(
        target.vectors.cross_similarities(reference_vectors, target.allowed, reference.allowed)
    )


def _shuffle_tokens(text: str, rng: random.Random) -> str:
    tokens = text.split()
    if len(tokens) < 2:
        return text
    shuffled = list(tokens)
    rng.shuffle(shuffled)
    if shuffled == tokens:
        shuffled = shuffled[1:] + shuffled[:1]
    return " ".join(shuffled)


def _shuffle_sentences(text: str, rng: random.Random) -> str:
    sentences = [part for part in _SENTENCE_BOUNDARY.split(text) if part]
    if len(sentences) < 2:
        return _shuffle_tokens(text, rng)
    shuffled = list(sentences)
    rng.shuffle(shuffled)
    if shuffled == sentences:
        shuffled = shuffled[1:] + shuffled[:1]
    return " ".join(shuffled)


def permutation_null(
    target: CorpusGeometry,
    embed: EmbedTexts,
    *,
    mode: str,
    permutations: int,
    seed: int,
) -> list[float]:
    """Original-to-shuffled similarities for deterministic token or sentence destruction."""
    if mode not in {"token", "sentence"}:
        raise ValueError(f"unknown permutation mode {mode!r}")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    shuffle = _shuffle_tokens if mode == "token" else _shuffle_sentences
    texts: list[str] = []
    for repeat in range(permutations):
        texts.extend(
            shuffle(
                target.chunks[ordinal]["text"],
                random.Random(seed + repeat * len(target.chunks) + ordinal),
            )
            for ordinal in target.allowed
        )
    shuffled = VectorSet.from_any(embed(texts))
    if shuffled.dim != target.vectors.dim:
        raise ValueError("permutation embedder dimension differs from the target store")
    if target.center_mean is not None:
        shuffled = shuffled.centered(target.center_mean)
    width = len(target.allowed)
    values = [
        dot(target.vectors.row(ordinal), shuffled.row(repeat * width + position))
        for repeat in range(permutations)
        for position, ordinal in enumerate(target.allowed)
    ]
    return sorted(values)


def held_out_document_null(target: CorpusGeometry) -> list[float]:
    """The maximum chunk cosine for each held-out unordered document pair."""
    return sorted(target.document_maxima.values())


def geometry_payload(geometry: CorpusGeometry) -> JsonObject:
    return {
        "corpus_root": geometry.corpus_root,
        "store_dir": str(geometry.store_dir),
        "embedding_model": geometry.embedding_model,
        "corpus_fingerprint": geometry.corpus_fingerprint,
        "dimensions": geometry.vectors.dim,
        "chunks": len(geometry.chunks),
        "comparable_chunks": len(geometry.allowed),
        "unique_comparable_texts": len(
            {geometry.chunks[index]["text"] for index in geometry.allowed}
        ),
        "documents": len({geometry.chunks[index]["doc_id"] for index in geometry.allowed}),
        "comparable_chunk_pairs": len(geometry.observed_similarities),
        "document_pairs": len(geometry.document_maxima),
        "centered": geometry.centered,
        **geometry.excluded,
    }
