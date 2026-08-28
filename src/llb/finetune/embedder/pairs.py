"""Contrastive (question, gold-chunk, hard-negative) export from tuning-split gold items.

A retrieval fine-tune needs three things per question: the chunk that answers it, chunks that look
like they do and do not, and a guarantee that neither came from data the final verdict is read on.

  - **The positive is derived, never labelled.** A gold item labels SOURCE SPANS, so the positive
    chunks are the ones whose character range overlaps a span (`llb.rag.retrieval.chunk_hits_span`,
    the same predicate the retrieval metric scores with). Training and scoring therefore agree on
    what a hit is; if they did not, an uplift here would not have to be an uplift there.
  - **The negatives are HARD, and rectangular.** BM25 over the same chunks supplies the passages
    that share the question's words without carrying its evidence -- the confusions a general
    encoder actually makes on a domain corpus. A row short of lexical matches is filled from the
    rest of the corpus deterministically, and if even that cannot reach the requested count on
    EVERY row, the width drops to what the corpus can supply for all of them rather than leaving a
    ragged set no batch can hold. The manifest records the requested count, the width actually
    used, and how many rows needed the fill, so a thin corpus reads as thin rather than as hard.
  - **The split is the boundary.** Only verified TUNING items are read here, and the manifest
    carries the item ids and split counts the guard re-checks against the gold set itself
    (`llb.finetune.guard.assert_tuning_only`).

Pure Python over chunk records: no torch, no encoder, no store. The chunker and the BM25 index are
the ones the retrieval path uses, so the pairs describe the chunks a query will actually meet.
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.executor.cases import spans_as_dicts
from llb.finetune.dataset import TUNING_SPLIT
from llb.finetune.dataset_io import _write_jsonl
from llb.finetune.embedder.manifest import (
    PAIRS_KIND,
    PAIRS_MANIFEST,
    pairs_digest,
    write_json_manifest,
)
from llb.finetune.embedder.negatives import (
    DEFAULT_NEGATIVES,
    hard_negatives,
    row_negatives,
)
from llb.goldset.schema import GoldItem, load_goldset
from llb.prep.corpus.fingerprints import corpus_fingerprint
from llb.rag.chunking.corpus import chunk_corpus
from llb.rag.retrieval import chunk_hits_any, span_char_coverage
from llb.rag.vector_store.lexical import Lemmatizer
from llb.rag.vector_store.lexical_index import LexicalIndex

_LOG = logging.getLogger(__name__)

PAIRS_FILENAME = "pairs.jsonl"

# How many gold chunks per question become anchors (the negatives ride in `negatives.py`).
DEFAULT_MAX_POSITIVES = 2


def export_contrastive_pairs(
    *,
    goldset_path: Path | str,
    corpus_root: Path | str,
    out_dir: Path | str,
    strategy: str = "recursive",
    size: int = 800,
    overlap: int = 120,
    max_positives: int = DEFAULT_MAX_POSITIVES,
    negatives: int = DEFAULT_NEGATIVES,
    lexical_lemmas: bool = False,
    lemmatizer: Lemmatizer | None = None,
    seed: int = 13,
) -> JsonObject:
    """Write `pairs.jsonl` + `pairs_manifest.json` for the verified tuning-split items.

    Raises `ValueError` when the corpus yields no chunks or no item has a retrievable positive --
    both mean there is nothing to train on, and a zero-row dataset would otherwise train an
    encoder into a manifest that claims it learned something.
    """
    items = [
        item for item in load_goldset(goldset_path) if item.verified and item.split == TUNING_SPLIT
    ]
    if not items:
        raise ValueError("embedder fine-tune export found no verified tuning-split gold items")
    chunks = chunk_corpus(Path(corpus_root), strategy, size, overlap)
    if not chunks:
        raise ValueError(f"no chunks produced from corpus at {corpus_root}")
    lexical = LexicalIndex.build(
        [str(chunk["text"]) for chunk in chunks], lemmatize=lexical_lemmas, lemmatizer=lemmatizer
    )
    rng = random.Random(seed)

    draft = _draft_rows(items, chunks, lexical, max_positives, negatives, rng)
    records = [
        _pair_record(item, positive, candidates[: draft.width])
        for item, positive, candidates, _n_hard in draft.rows
    ]
    out = Path(out_dir)
    _write_jsonl(out / PAIRS_FILENAME, records)
    manifest: JsonObject = {
        "kind": PAIRS_KIND,
        "dataset_digest": pairs_digest(records),
        "goldset_path": str(goldset_path),
        "corpus_root": str(corpus_root),
        "corpus_fingerprint": corpus_fingerprint(corpus_root),
        "chunking": {"strategy": strategy, "size": size, "overlap": overlap},
        "lexical_lemmas": bool(lexical_lemmas),
        "item_ids": sorted(draft.covered),
        "split_counts": {TUNING_SPLIT: len(draft.covered)},
        "items_without_positive": sorted(draft.uncovered),
        "n_pairs": len(records),
        "n_chunks": len(chunks),
        "requested_negatives": negatives,
        "negatives_per_pair": draft.width,
        "rows_needing_filled_negatives": draft.rows_needing_fill,
        "max_positives_per_item": max_positives,
        "seed": seed,
    }
    write_json_manifest(out / PAIRS_MANIFEST, manifest)
    return manifest


# One drafted row before the common width is applied: the item, its positive chunk, the negatives
# the corpus could supply for it, and how many of those were lexically hard rather than filled.
DraftedRow = tuple[GoldItem, ChunkRecord, list[ChunkRecord], int]


@dataclass(frozen=True)
class PairDraft:
    """Every row the gold set produced, plus the width they all end up sharing."""

    rows: list[DraftedRow]
    covered: list[str]
    uncovered: list[str]

    @property
    def width(self) -> int:
        """Negatives per row: what the THINNEST row can supply, so no row goes ragged."""
        return min(len(candidates) for _i, _p, candidates, _h in self.rows)

    @property
    def rows_needing_fill(self) -> int:
        """Rows the lexical index could not fill on its own at the shared width."""
        return sum(1 for _i, _p, _c, n_hard in self.rows if n_hard < self.width)


def _draft_rows(
    items: list[GoldItem],
    chunks: list[ChunkRecord],
    lexical: LexicalIndex,
    max_positives: int,
    negatives: int,
    rng: random.Random,
) -> PairDraft:
    """Draw the positives and negatives for every item, and report what the corpus could not do."""
    rows: list[DraftedRow] = []
    covered: list[str] = []
    uncovered: list[str] = []
    for item in items:
        positives = _positive_chunks(item, chunks, max_positives)
        if not positives:
            uncovered.append(item.id)
            continue
        covered.append(item.id)
        spans = spans_as_dicts(item)
        pool = hard_negatives(item.question, spans, chunks, lexical, negatives)
        for positive in positives:
            candidates, n_hard = row_negatives(spans, positive, chunks, pool, negatives, rng)
            rows.append((item, positive, candidates, n_hard))
    if not rows:
        raise ValueError(
            "no tuning-split gold item has a chunk overlapping its source spans; the corpus and "
            "the gold set do not describe the same documents"
        )
    draft = PairDraft(rows=rows, covered=covered, uncovered=uncovered)
    _report_draft(draft, negatives)
    return draft


def _report_draft(draft: PairDraft, negatives: int) -> None:
    """Say out loud what the corpus could not supply, so a thin export never reads as a rich one."""
    if draft.width < negatives:
        _LOG.warning(
            "[finetune-embedder] the corpus supplies at most %d negative(s) for some rows; every "
            "row carries %d rather than the requested %d",
            draft.width,
            draft.width,
            negatives,
        )
    if draft.uncovered:
        _LOG.warning(
            "[finetune-embedder] %d tuning item(s) have no chunk over their gold spans and were "
            "skipped: %s",
            len(draft.uncovered),
            ", ".join(draft.uncovered[:5]) + ("..." if len(draft.uncovered) > 5 else ""),
        )


def _positive_chunks(
    item: GoldItem, chunks: list[ChunkRecord], max_positives: int
) -> list[ChunkRecord]:
    """The chunks that carry this item's evidence, most of it first.

    Ranking by covered gold characters (not by build order) is what makes a truncated list the
    BEST positives rather than the first ones -- a span split across a chunk boundary leaves one
    chunk holding a sentence and the next holding three words.
    """
    spans = spans_as_dicts(item)
    scored = [
        (_covered_chars(chunk, spans), ordinal, chunk)
        for ordinal, chunk in enumerate(chunks)
        if chunk_hits_any(chunk, spans)
    ]
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [chunk for _coverage, _ordinal, chunk in scored[:max_positives]]


def _covered_chars(chunk: ChunkRecord, spans: list[SourceSpanRecord]) -> float:
    """How much of this item's labelled evidence one chunk carries, summed over its spans."""
    return sum((span_char_coverage([chunk], span) for span in spans), 0.0)


def _pair_record(item: GoldItem, positive: ChunkRecord, negatives: list[ChunkRecord]) -> JsonObject:
    """One training row: the question, its evidence chunk, and the passages it must outrank."""
    return {
        "item_id": item.id,
        "split": item.split,
        "question": item.question,
        "positive": str(positive["text"]),
        "positive_chunk_id": str(positive.get("chunk_id", "")),
        "negatives": [str(chunk["text"]) for chunk in negatives],
        "negative_chunk_ids": [str(chunk.get("chunk_id", "")) for chunk in negatives],
    }
