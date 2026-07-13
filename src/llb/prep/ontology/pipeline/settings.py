"""The two data objects of a draft run: `DraftSettings` (every knob, journaled for `--resume` and
echoed into provenance) and `PipelineResult` (the in-memory handle on a completed run).

Collecting the knobs in one object instead of threading ~16 loose parameters keeps `draft_goldset`
readable and gives resume/journal/provenance a single source of truth.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llb.goldset.chains import ChainItem
from llb.goldset.schema import GoldItem
from llb.prep.frontier import ProvenanceLog
from llb.prep.ontology.constants import (
    DEFAULT_MAX_ITEMS,
    DEFAULT_MULTI_HOP_MAX_PATHS,
    EXTRACT_CHUNK_OVERLAP,
    EXTRACT_CONCURRENCY,
    EXTRACT_MAX_CHARS,
)
from llb.prep.ontology.models import (
    DocExtraction,
    DocRecord,
    DraftSeed,
    ItemLabels,
    OntologyCandidate,
)


def _opt_str(value: Path | str | None) -> str | None:
    return str(value) if value is not None else None


@dataclass
class DraftSettings:
    """All knobs of one draft run, journaled for `--resume` and echoed into provenance.

    Collecting them in one object (instead of threading ~16 loose parameters through the
    pipeline) keeps `draft_goldset` readable and gives resume/journal/provenance a single
    source of truth.
    """

    corpus_root: str
    max_items: int = DEFAULT_MAX_ITEMS
    seed: int = 13
    doc_limit: int | None = None
    extract_max_chars: int | None = None
    extract_chunk_overlap: int | None = None
    extract_concurrency: int | None = None
    retrieval_index_dir: Path | str | None = None
    retrieval_k: int = 10
    drop_nonretrievable_needles: bool = False
    coverage_target: int | None = None
    multi_hop: bool = False
    chains: bool = False
    multi_hop_max_paths: int = DEFAULT_MULTI_HOP_MAX_PATHS
    dedup_against: list[Path | str] | None = None
    graph_dir: Path | str | None = None
    rejection_feedback: Path | str | None = None

    def apply_resume_meta(self, meta: dict[str, Any]) -> None:
        """Overwrite knobs with the pinned values of the interrupted run being resumed."""
        self.corpus_root = str(meta.get("corpus_root", self.corpus_root))
        self.seed = int(meta.get("seed", self.seed))
        self.max_items = int(meta.get("max_items", self.max_items))
        doc_limit = meta.get("doc_limit", self.doc_limit)
        self.doc_limit = int(doc_limit) if doc_limit is not None else None
        self.extract_max_chars = meta.get("extract_max_chars", self.extract_max_chars)
        self.extract_chunk_overlap = meta.get("extract_chunk_overlap", self.extract_chunk_overlap)
        self.extract_concurrency = meta.get("extract_concurrency", self.extract_concurrency)
        self.retrieval_index_dir = meta.get("retrieval_index_dir") or self.retrieval_index_dir
        self.retrieval_k = int(meta.get("retrieval_k", self.retrieval_k))
        self.drop_nonretrievable_needles = bool(
            meta.get("drop_nonretrievable_needles", self.drop_nonretrievable_needles)
        )
        coverage = meta.get("coverage_target", self.coverage_target)
        self.coverage_target = int(coverage) if coverage is not None else None
        self.multi_hop = bool(meta.get("multi_hop", self.multi_hop))
        self.chains = bool(meta.get("chains", self.chains))
        self.multi_hop_max_paths = int(meta.get("multi_hop_max_paths", self.multi_hop_max_paths))
        dedup = meta.get("dedup_against")
        self.dedup_against = list(dedup) if dedup is not None else self.dedup_against
        self.graph_dir = meta.get("graph_dir") or self.graph_dir
        self.rejection_feedback = meta.get("rejection_feedback") or self.rejection_feedback

    def validate(self) -> None:
        if self.doc_limit is not None and self.doc_limit < 1:
            raise ValueError("doc_limit must be >= 1 when set")
        if self.extract_concurrency is not None and self.extract_concurrency < 1:
            raise ValueError("extract_concurrency must be >= 1 when set")
        if self.retrieval_k < 1:
            raise ValueError("retrieval_k must be >= 1")

    @property
    def resolved_extract_max_chars(self) -> int:
        return self.extract_max_chars if self.extract_max_chars is not None else EXTRACT_MAX_CHARS

    @property
    def resolved_extract_overlap(self) -> int:
        return (
            self.extract_chunk_overlap
            if self.extract_chunk_overlap is not None
            else EXTRACT_CHUNK_OVERLAP
        )

    @property
    def resolved_extract_concurrency(self) -> int:
        return (
            self.extract_concurrency
            if self.extract_concurrency is not None
            else EXTRACT_CONCURRENCY
        )

    def pinned_payload(self) -> dict[str, object]:
        """Determinism-critical settings recorded in the journal meta sidecar for resume."""
        return {
            "corpus_root": self.corpus_root,
            "seed": self.seed,
            "max_items": self.max_items,
            "doc_limit": self.doc_limit,
            "extract_max_chars": self.resolved_extract_max_chars,
            "extract_chunk_overlap": self.resolved_extract_overlap,
            "extract_concurrency": self.resolved_extract_concurrency,
            "retrieval_index_dir": _opt_str(self.retrieval_index_dir),
            "retrieval_k": self.retrieval_k,
            "drop_nonretrievable_needles": self.drop_nonretrievable_needles,
            "coverage_target": self.coverage_target,
            "multi_hop": self.multi_hop,
            "chains": self.chains,
            "multi_hop_max_paths": self.multi_hop_max_paths,
            "dedup_against": [str(path) for path in self.dedup_against]
            if self.dedup_against
            else None,
            "graph_dir": _opt_str(self.graph_dir),
            "rejection_feedback": _opt_str(self.rejection_feedback),
        }

    def provenance_settings(self, resumed: bool) -> dict[str, object]:
        """The `settings` block of the bundle provenance record."""
        return {
            "max_items": self.max_items,
            "seed": self.seed,
            "doc_limit": self.doc_limit,
            "extract_max_chars": self.resolved_extract_max_chars,
            "extract_chunk_overlap": self.resolved_extract_overlap,
            "extract_concurrency": self.resolved_extract_concurrency,
            "coverage_target": self.coverage_target,
            "multi_hop": self.multi_hop,
            "chains": self.chains,
            "multi_hop_max_paths": self.multi_hop_max_paths,
            "dedup_against": [str(path) for path in self.dedup_against]
            if self.dedup_against
            else None,
            "graph_dir": _opt_str(self.graph_dir),
            "rejection_feedback": _opt_str(self.rejection_feedback),
            "needle_retrieval_index_dir": _opt_str(self.retrieval_index_dir),
            "needle_retrieval_k": self.retrieval_k,
            "drop_nonretrievable_needles": self.drop_nonretrievable_needles,
            "resumed": resumed,
        }


@dataclass
class PipelineResult:
    """Programmatic handle on a draft run (also the basis for the provenance record)."""

    out_dir: Path
    docs: list[DocRecord]
    extractions: list[DocExtraction]
    ontology: OntologyCandidate
    seeds: list[DraftSeed]
    items: list[GoldItem]
    corpus_root: Path
    chains: list[ChainItem] = field(default_factory=list)
    elapsed_s: float = 0.0
    calibration_report: dict[str, object] | None = None
    item_labels: dict[str, ItemLabels] = field(default_factory=dict)
    coverage_report: dict[str, object] | None = None
    dedup_report: dict[str, object] | None = None
    applied_feedback: dict[str, object] | None = None
    log: ProvenanceLog = field(default_factory=ProvenanceLog)
