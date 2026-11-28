"""Run paired retrieval with fresh, identically configured indexes."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from llb.artifacts.retrieval_graph.opaque import content_digest
from llb.bench.common import new_run_timestamp
from llb.core.config import RunConfig
from llb.core.fsutil import atomic_write_text
from llb.rag.book_comparison.inputs import (
    BookQuestion,
    load_book_pair,
    load_book_questions,
)
from llb.rag.cited_corpus import CitedDocument
from llb.rag.book_comparison.metrics import (
    PairedRank,
    Ranks,
    ReadingSettings,
    declared_settings,
    read_slices,
    retrieve_ranks,
)
from llb.rag.vector_store.configured import build_configured_index

METHOD = "compare-book-retrieval"


@dataclass(frozen=True)
class BookRetrievalRun:
    projection: Path
    baseline: Path
    goldset: Path

    def inputs(self) -> dict[str, Path]:
        return {"projection": self.projection, "baseline": self.baseline, "goldset": self.goldset}


def _read_lane(
    config: RunConfig,
    index_dir: Path,
    docs: dict[str, CitedDocument],
    items: list[BookQuestion],
    k: int,
    *,
    baseline: bool,
) -> tuple[list[Ranks], dict[str, object]]:
    store = build_configured_index(config)
    store.fusion_weight = config.fusion_weight
    store.fusion_candidates = config.fusion_candidates
    ranks = retrieve_ranks(store, items, docs, k=k, baseline=baseline)
    store.save(index_dir)
    return ranks, {**store.meta, "index": index_dir.name}


def _item_row(row: PairedRank) -> dict[str, object]:
    return {
        "item_id": row.item.item_id,
        "pdf_pages": sorted(row.item.pages),
        "baseline_text_present": row.item.baseline_text_present,
        "evidence_parts": row.item.evidence_parts,
        "baseline_parts_present": row.item.present_parts,
        "baseline_span_matches": len(row.item.baseline_spans),
        "projection": asdict(row.projection),
        "baseline": asdict(row.baseline),
    }


def run_book_comparison(
    inputs: BookRetrievalRun, config: RunConfig, settings: ReadingSettings
) -> tuple[Path, dict[str, dict[str, object]]]:
    """Validate gold and PDF provenance before building either lane."""
    _, stamp = new_run_timestamp()
    out_dir = config.data_dir / METHOD / stamp
    paths = inputs.inputs()
    if any(out_dir.is_relative_to(path.resolve()) for path in paths.values()):
        raise ValueError("comparison output must be outside its inputs")

    digests = {name: content_digest(path) for name, path in paths.items()}
    pair = load_book_pair(inputs.projection, inputs.baseline)
    items = load_book_questions(inputs.goldset, inputs.projection, pair)

    out_dir.mkdir(parents=True, exist_ok=False)
    projection, projection_index = _read_lane(
        config.with_overrides(corpus_root=inputs.projection),
        out_dir / "projection",
        pair.projection,
        items,
        settings.k,
        baseline=False,
    )
    baseline, baseline_index = _read_lane(
        config.with_overrides(corpus_root=inputs.baseline),
        out_dir / "baseline",
        {pair.baseline.doc_id: pair.baseline},
        items,
        settings.k,
        baseline=True,
    )
    if any(content_digest(path) != digests[name] for name, path in paths.items()):
        raise ValueError("comparison inputs changed during retrieval")
    rows = [PairedRank(*row) for row in zip(items, projection, baseline, strict=True)]
    slices = read_slices(rows, settings)
    report = {
        "source_pdf_sha256": pair.source_sha256,
        "baseline_parser": pair.baseline.citations.parser,
        "inputs": {
            name: {"path": str(path), "digest": digests[name]} for name, path in paths.items()
        },
        "settings": {
            **declared_settings(settings),
            "fusion_weight": config.fusion_weight,
            "fusion_candidates": config.fusion_candidates,
        },
        "indexes": {"projection": projection_index, "baseline": baseline_index},
        "slices": slices,
        "items": [_item_row(row) for row in rows],
    }
    atomic_write_text(out_dir / "report.json", json.dumps(report, indent=2) + "\n")
    return out_dir, slices
