"""Stage 7 -- orchestrate the ontology-assisted gold-set drafting pipeline.

Runs the grained stages in order:

    1 inventory -> 2 extract -> 3 induce ontology -> 4 sample coverage
    -> 5 draft QA -> 6 ground/dedup/reject -> 7 emit bundle

and writes a self-contained, traceable bundle under `$DATA_DIR/prepare-goldset/<timestamp>/`:
the `verified=false` canonical drafts, a copy of the corpus they index (so the validator runs
on the bundle), the induced ontology, the per-document extraction, and a provenance record
linking ontology / extraction / endpoint / prompt / model / cost / document hashes. Nothing is
verified -- a frontier cross-check and a human sample-verify (human verification gate) gate any scoring.

`complete` and `extraction_adapter` are injectable, so the whole flow is unit-tested with a
fake endpoint and never needs a server or a provider key.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from llb.graph.model import KnowledgeGraph

from llb.goldset.chains import ChainItem, dump_chains
from llb.goldset.schema import GoldItem, Split, dump_goldset
from llb.goldset.splits import assign_splits
from llb.core.paths import resolve_data_dir
from llb.prep.frontier import LLMComplete, ProvenanceLog
from llb.prep.ontology.artifacts import (
    copy_pdf_citation_sidecars,
    required_gate_names,
    write_calibration_artifacts,
)
from llb.prep.ontology.constants import (
    CORPUS_DIRNAME,
    CHAINS_FILENAME,
    DEFAULT_MAX_ITEMS,
    DEFAULT_MULTI_HOP_MAX_PATHS,
    EXTRACT_CHUNK_OVERLAP,
    EXTRACT_CONCURRENCY,
    EXTRACT_MAX_CHARS,
    EXTRACTION_FILENAME,
    EXTRACTION_JOURNAL_FILENAME,
    EXTRACTION_JOURNAL_META_FILENAME,
    EXTRACTION_JOURNAL_META_KIND,
    GOLDSET_FILENAME,
    METHOD_DIR,
    ONTOLOGY_FILENAME,
    PDF_ONTOLOGY_REPORT_FILENAME,
    PROVENANCE_FILENAME,
    PROVENANCE_KIND,
)
from llb.prep.ontology.coverage import build_seeds, coverage_report, select_seeds
from llb.prep.ontology.dedup import QuestionEmbedder
from llb.prep.ontology.draft import draft_items, draft_prompt
from llb.prep.ontology.endpoint import EndpointConfig, build_complete
from llb.prep.ontology.extract import (
    ExtractionAdapter,
    LLMExtractionAdapter,
    extract_corpus,
    extraction_prompt,
)
from llb.prep.ontology.induce import induce_ontology, ontology_constraints
from llb.prep.ontology.inventory import inventory_corpus
from llb.prep.ontology.journal import ExtractionJournal
from llb.prep.ontology.models import (
    DocExtraction,
    DocRecord,
    DraftSeed,
    ItemLabels,
    OntologyCandidate,
)
from llb.prep.ontology.needles import NeedleRetriever
from llb.prep.ontology.refine import refine_drafts_labeled

_LOG = logging.getLogger(__name__)
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


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


def _opt_str(value: Path | str | None) -> str | None:
    return str(value) if value is not None else None


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


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime(_TIMESTAMP_FORMAT)


def default_out_dir() -> Path:
    return resolve_data_dir() / METHOD_DIR / _timestamp()


def _journal_meta_path(out_dir: Path) -> Path:
    return out_dir / EXTRACTION_JOURNAL_META_FILENAME


def _clear_fresh_extraction_journal(out_dir: Path) -> None:
    """Drop prior resumability state when the caller starts a fresh run in an existing bundle dir."""
    for name in (EXTRACTION_JOURNAL_FILENAME, EXTRACTION_JOURNAL_META_FILENAME):
        path = out_dir / name
        if path.exists():
            path.unlink()


def _write_journal_meta(out_dir: Path, pinned: dict[str, object], endpoint: EndpointConfig) -> None:
    """Record the determinism-critical settings + endpoint identity so a resume reproduces the run.

    Written once at the start of a fresh run (before any model call) so the sidecar survives a kill
    at any point during extraction.
    """
    payload = {
        "kind": EXTRACTION_JOURNAL_META_KIND,
        "endpoint": endpoint.provenance(),
        **pinned,
    }
    _journal_meta_path(out_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_journal_meta(out_dir: Path | str) -> dict[str, object]:
    """Read the journal meta sidecar for `--resume`. Raises a clear error when it is absent."""
    path = _journal_meta_path(Path(out_dir))
    if not path.is_file():
        raise ValueError(
            f"cannot resume: no {EXTRACTION_JOURNAL_META_FILENAME} in {out_dir} "
            "(a resumable draft writes it at the start of extraction)"
        )
    meta = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"malformed journal meta: {path}")
    return meta


def _prompt_fingerprints() -> dict[str, str]:
    """sha256 of the exact template wording, so a run records WHICH prompts produced it."""
    placeholder_seed = DraftSeed(
        doc_id="<doc>",
        kind="fact",
        section_title="<section>",
        difficulty="medium",
        strata={},
        evidence={"doc_id": "<doc>", "char_start": 0, "char_end": 1, "text": "x"},  # type: ignore[arg-type]
    )
    from llb.prep.ontology.models import MultiHopSeed, MultiHopStep
    from llb.prep.ontology.multi_hop import multi_hop_prompt

    placeholder_step = MultiHopStep(
        subject="<a>",
        relation="<r>",
        object="<b>",
        section_title="<section>",
        evidence={"doc_id": "<doc>", "char_start": 0, "char_end": 1, "text": "x"},  # type: ignore[arg-type]
    )
    placeholder_chain = MultiHopSeed(
        steps=[placeholder_step, placeholder_step], bridge="<b>", start="<a>", end="<c>"
    )
    extract_tmpl = extraction_prompt("<doc>", "<text>")
    draft_tmpl = draft_prompt(placeholder_seed, "<context>")
    multi_hop_tmpl = multi_hop_prompt(placeholder_chain, "<context>")
    return {
        "extraction": hashlib.sha256(extract_tmpl.encode("utf-8")).hexdigest(),
        "draft": hashlib.sha256(draft_tmpl.encode("utf-8")).hexdigest(),
        "multi_hop": hashlib.sha256(multi_hop_tmpl.encode("utf-8")).hexdigest(),
    }


def _load_path_graph(
    graph_dir: Path | str | None,
    extractions: list[DocExtraction],
    docs: list[DocRecord],
    ontology: OntologyCandidate,
) -> "KnowledgeGraph":
    """The knowledge graph the multi-hop walker reads: a persisted store, else built in-run."""
    if graph_dir is not None:
        from llb.graph.store import GraphStore

        return GraphStore.load(graph_dir).graph
    from llb.graph.build import build_graph

    return build_graph(extractions, docs, ontology)


def _multi_hop_stage(
    complete: LLMComplete,
    docs: list[DocRecord],
    extractions: list[DocExtraction],
    ontology: OntologyCandidate,
    *,
    graph_dir: Path | str | None,
    max_paths: int,
    seed: int,
) -> tuple[list[GoldItem], dict[str, ItemLabels]]:
    """Walk 2-hop graph paths and draft multi-span multi-hop chain items (yield-max)."""
    from llb.prep.ontology.graph_paths import walk_two_hop_paths
    from llb.prep.ontology.multi_hop import build_multi_hop_items, draft_multi_hop

    graph = _load_path_graph(graph_dir, extractions, docs, ontology)
    seeds = walk_two_hop_paths(graph, max_paths=max_paths, seed=seed)
    raw = draft_multi_hop(complete, docs, seeds)
    return build_multi_hop_items(docs, seeds, raw)


def _chain_stage(
    docs: list[DocRecord],
    extractions: list[DocExtraction],
    ontology: OntologyCandidate,
    *,
    graph_dir: Path | str | None,
    max_paths: int,
    seed: int,
) -> list[ChainItem]:
    """Walk 2-hop graph paths and emit ordered chain-of-questions items."""
    from llb.prep.ontology.chains import build_chain_items
    from llb.prep.ontology.graph_paths import walk_chain_paths

    graph = _load_path_graph(graph_dir, extractions, docs, ontology)
    seeds = walk_chain_paths(graph, max_paths=max_paths, seed=seed)
    return build_chain_items(docs, seeds)


def _dedup_stage(
    items: list[GoldItem],
    labels: dict[str, ItemLabels],
    *,
    dedup_against: list[Path | str],
    embedder: QuestionEmbedder | None,
) -> tuple[list[GoldItem], dict[str, ItemLabels], dict[str, object]]:
    """Drop near-duplicates of prior-bundle questions (pinned E5); prune their labels (yield-max)."""
    from llb.prep.ontology.dedup import (
        E5QuestionEmbedder,
        NearDuplicateFilter,
        load_prior_questions,
    )

    prior = load_prior_questions(dedup_against)
    resolved = embedder if embedder is not None else E5QuestionEmbedder()
    kept, report = NearDuplicateFilter(prior, resolved).filter(items)
    kept_ids = {item.id for item in kept}
    kept_labels = {item_id: label for item_id, label in labels.items() if item_id in kept_ids}
    report["prior_bundles"] = [str(path) for path in dedup_against]
    return kept, kept_labels, report


def _write_corpus_copy(source_root: Path, corpus_dir: Path, docs: list[DocRecord]) -> None:
    """Copy inventoried docs verbatim so spans stay exact and the bundle self-validates."""
    for doc in docs:
        target = corpus_dir / doc.doc_id
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(doc.text, encoding="utf-8")
    copy_pdf_citation_sidecars(source_root, corpus_dir, [doc.doc_id for doc in docs])


def _label_counts(result: PipelineResult) -> dict[str, dict[str, int]]:
    """Question-type and difficulty distributions over the drafted items (from item labels)."""
    by_type: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    for item in result.items:
        label = result.item_labels.get(item.id)
        qtype = label.question_type if label else "factoid"
        difficulty = label.difficulty if label else "medium"
        by_type[qtype] = by_type.get(qtype, 0) + 1
        by_difficulty[difficulty] = by_difficulty.get(difficulty, 0) + 1
    return {
        "question_type_distribution": dict(sorted(by_type.items())),
        "difficulty_distribution": dict(sorted(by_difficulty.items())),
    }


def _provenance(
    result: PipelineResult, endpoint: EndpointConfig, seed: int, settings: dict[str, object]
) -> dict[str, object]:
    n_multi_hop = sum(
        1
        for item in result.items
        if (label := result.item_labels.get(item.id)) and label.question_type == "multi-hop"
    )
    provenance: dict[str, object] = {
        "kind": PROVENANCE_KIND,
        "synthetic": False,  # drafted FROM a real corpus (vs planted synthetic docs)
        "endpoint": endpoint.provenance(),
        "prompts": _prompt_fingerprints(),
        "seed": seed,
        "settings": settings,
        "elapsed_s": round(result.elapsed_s, 3),
        "documents": [
            {"doc_id": doc.doc_id, "sha256": doc.sha256, "n_chars": doc.n_chars}
            for doc in result.docs
        ],
        "stages": {
            "documents": len(result.docs),
            "entities": sum(len(e.entities) for e in result.extractions),
            "events": sum(len(e.events) for e in result.extractions),
            "claims": sum(len(e.claims) for e in result.extractions),
            "facts": sum(len(e.facts) for e in result.extractions),
            "ontology_entity_types": len(result.ontology.entity_types),
            "ontology_relation_types": len(result.ontology.relation_types),
            "seeds": len(result.seeds),
            "multi_hop_items": n_multi_hop,
            "chains": len(result.chains),
            "items": len(result.items),
        },
        "labels": _label_counts(result),
        "ontology": result.ontology.model_dump(),
        "n_items": len(result.items),
        "cost": result.log.summary(),
    }
    if result.coverage_report is not None:
        provenance["seed_coverage"] = result.coverage_report
    if result.dedup_report is not None:
        provenance["dedup"] = result.dedup_report
    if result.applied_feedback is not None:
        provenance["applied_feedback"] = result.applied_feedback
    return provenance


def _load_retrieval_store(index_dir: Path | str | None) -> NeedleRetriever | None:
    if index_dir is None:
        return None
    from llb.rag.store import RagStore

    return RagStore.load(index_dir)


def _write_bundle(
    result: PipelineResult,
    endpoint: EndpointConfig,
    seed: int,
    settings: dict[str, object],
    *,
    retrieval_store: NeedleRetriever | None = None,
    retrieval_k: int = 10,
    drop_nonretrievable_needles: bool = False,
) -> None:
    out_dir = result.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_goldset(result.items, out_dir / GOLDSET_FILENAME)
    if result.chains:
        dump_chains(result.chains, out_dir / CHAINS_FILENAME)
    _write_corpus_copy(result.corpus_root, out_dir / CORPUS_DIRNAME, result.docs)
    (out_dir / ONTOLOGY_FILENAME).write_text(
        json.dumps(result.ontology.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / EXTRACTION_FILENAME).open("w", encoding="utf-8") as fh:
        for extraction in result.extractions:
            fh.write(json.dumps(extraction.model_dump(), ensure_ascii=False) + "\n")
    (out_dir / PROVENANCE_FILENAME).write_text(
        json.dumps(_provenance(result, endpoint, seed, settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result.calibration_report = write_calibration_artifacts(
        out_dir,
        result.docs,
        result.extractions,
        result.ontology,
        result.items,
        elapsed_s=result.elapsed_s,
        settings=settings,
        retrieval_store=retrieval_store,
        retrieval_k=retrieval_k,
        drop_nonretrievable_needles=drop_nonretrievable_needles,
        item_labels=result.item_labels,
        coverage_matrix=result.coverage_report,
        dedup_report=result.dedup_report,
    )
    _LOG.info(
        "[ontology] wrote %d drafts and %d chains (verified=false) + provenance -> %s",
        len(result.items),
        len(result.chains),
        out_dir,
    )
    _log_calibration_gates(result.calibration_report, out_dir)


def _log_calibration_gates(report: dict[str, object] | None, out_dir: Path) -> None:
    """Surface the calibration roll-up so `prepare-goldset-draft` (and the quickstart wrapper) act
    on the gate, not just record it. A failing gate is a WARNING, never fatal: the bundle is always
    written for inspection, and the human verification gate remains the real block on scoring."""
    gates = report.get("gates") if isinstance(report, dict) else None
    if not isinstance(gates, dict):
        return
    if gates.get("passed"):
        _LOG.info(
            "[ontology] calibration gates passed -> %s", out_dir / PDF_ONTOLOGY_REPORT_FILENAME
        )
        return
    # name only the REQUIRED gates that blocked the roll-up (informational gates like
    # nonzero_grounded_facts, and the needle gate on a non-PDF corpus, never appear here)
    required = required_gate_names(bool(gates.get("pdf_citation_gate_applicable")))
    failed = [name for name in required if not gates.get(name)]
    _LOG.warning(
        "[ontology] calibration gates NOT passed (%s); inspect %s before accepting this bundle",
        ", ".join(failed) or "see report",
        out_dir / PDF_ONTOLOGY_REPORT_FILENAME,
    )


def _prepare_bundle_dir(
    resolved_out: Path, settings: DraftSettings, endpoint: EndpointConfig, resume: bool
) -> ExtractionJournal:
    """Create the bundle dir, pin settings (fresh run only), and open the extraction journal."""
    resolved_out.mkdir(parents=True, exist_ok=True)
    if not resume:
        _clear_fresh_extraction_journal(resolved_out)
        _write_journal_meta(resolved_out, settings.pinned_payload(), endpoint)
    journal = ExtractionJournal(resolved_out / EXTRACTION_JOURNAL_FILENAME)
    journal.load()
    return journal


def _feedback_adjusted_hint(
    draft_hint: str, rejection_feedback: Path | str
) -> tuple[str, dict[str, object]]:
    """Tighten the draft prompt with verify-gate rejection feedback; return the applied block."""
    from llb.prep.ontology.feedback import (
        applied_feedback_block,
        feedback_hint_text,
        feedback_hints,
        load_rejection_feedback,
    )

    hints = feedback_hints(load_rejection_feedback(rejection_feedback))
    applied = applied_feedback_block(rejection_feedback, hints)
    hint_text = feedback_hint_text(hints)
    if hint_text:
        draft_hint = f"{draft_hint}\n{hint_text}" if draft_hint else hint_text
        _LOG.info(
            "[ontology] applying rejection feedback (%d hint(s)) from %s",
            len(hints),
            rejection_feedback,
        )
    return draft_hint, applied


def _draft_stage(
    complete: LLMComplete,
    docs: list[DocRecord],
    extractions: list[DocExtraction],
    ontology: OntologyCandidate,
    settings: DraftSettings,
) -> tuple[list[GoldItem], dict[str, ItemLabels], dict[str, object], dict[str, object] | None]:
    """Stages 4-5: seed selection + QA drafting. Returns items, labels, coverage, feedback."""
    pool = build_seeds(docs, extractions)
    seeds = select_seeds(
        pool,
        max_items=settings.max_items,
        seed=settings.seed,
        coverage_target=settings.coverage_target,
    )
    cov_report = coverage_report(
        pool, seeds, coverage_target=settings.coverage_target, max_items=settings.max_items
    )
    draft_hint = ontology_constraints(ontology)
    applied_feedback: dict[str, object] | None = None
    if settings.rejection_feedback is not None:
        draft_hint, applied_feedback = _feedback_adjusted_hint(
            draft_hint, settings.rejection_feedback
        )
    raw_drafts = draft_items(complete, docs, seeds, draft_hint)
    items, item_labels = refine_drafts_labeled(docs, raw_drafts)
    return items, item_labels, {"seeds": seeds, "coverage": cov_report}, applied_feedback


def _graph_stages(
    complete: LLMComplete,
    docs: list[DocRecord],
    extractions: list[DocExtraction],
    ontology: OntologyCandidate,
    settings: DraftSettings,
    items: list[GoldItem],
    item_labels: dict[str, ItemLabels],
) -> tuple[list[GoldItem], dict[str, ItemLabels], list[ChainItem]]:
    """Optional graph-walk stages: multi-hop items and ordered question chains."""
    if settings.multi_hop:
        mh_items, mh_labels = _multi_hop_stage(
            complete,
            docs,
            extractions,
            ontology,
            graph_dir=settings.graph_dir,
            max_paths=settings.multi_hop_max_paths,
            seed=settings.seed,
        )
        items = items + mh_items
        item_labels = {**item_labels, **mh_labels}
    chain_items: list[ChainItem] = []
    if settings.chains:
        chain_items = _chain_stage(
            docs,
            extractions,
            ontology,
            graph_dir=settings.graph_dir,
            max_paths=settings.multi_hop_max_paths,
            seed=settings.seed,
        )
    return items, item_labels, chain_items


def draft_goldset(
    corpus_root: Path | str,
    endpoint: EndpointConfig,
    *,
    complete: LLMComplete | None = None,
    extraction_adapter: ExtractionAdapter | None = None,
    max_items: int = DEFAULT_MAX_ITEMS,
    seed: int = 13,
    out_dir: Path | str | None = None,
    doc_limit: int | None = None,
    extract_max_chars: int | None = None,
    extract_chunk_overlap: int | None = None,
    extract_concurrency: int | None = None,
    retrieval_index_dir: Path | str | None = None,
    retrieval_k: int = 10,
    drop_nonretrievable_needles: bool = False,
    coverage_target: int | None = None,
    multi_hop: bool = False,
    chains: bool = False,
    multi_hop_max_paths: int = DEFAULT_MULTI_HOP_MAX_PATHS,
    dedup_against: list[Path | str] | None = None,
    graph_dir: Path | str | None = None,
    dedup_embedder: QuestionEmbedder | None = None,
    rejection_feedback: Path | str | None = None,
    write: bool = True,
    resume: bool = False,
) -> PipelineResult:
    """Run stages 1-7 and (by default) write the bundle. Returns the in-memory result.

    Yield-max knobs: `coverage_target` drafts up to N seeds per stratum bucket instead of the flat
    `max_items` cap; `multi_hop` also drafts multi-span questions walked from the knowledge
    graph (built in-run, or loaded from `graph_dir`); `chains` emits ordered chain-of-questions
    rows from the same graph paths; `dedup_against` drops questions that are pinned-E5
    near-duplicates of the listed prior bundles. `rejection_feedback`
    (draft-feedback-rejection-reasons) points at a verify-gate `rejection_reasons.json`; its
    dominant reject codes tighten the draft prompts deterministically, and the applied hints +
    file digest land in provenance. `resume=True` re-enters an existing bundle: it reads
    the pinned settings from the journal meta, reuses journaled extraction windows instead of
    re-calling the model, and replays the deterministic seed/draft/emit stages -- producing the same
    bundle as an uninterrupted run.
    """
    started = perf_counter()
    resolved_out = Path(out_dir) if out_dir is not None else default_out_dir()
    settings = DraftSettings(
        corpus_root=str(corpus_root),
        max_items=max_items,
        seed=seed,
        doc_limit=doc_limit,
        extract_max_chars=extract_max_chars,
        extract_chunk_overlap=extract_chunk_overlap,
        extract_concurrency=extract_concurrency,
        retrieval_index_dir=retrieval_index_dir,
        retrieval_k=retrieval_k,
        drop_nonretrievable_needles=drop_nonretrievable_needles,
        coverage_target=coverage_target,
        multi_hop=multi_hop,
        chains=chains,
        multi_hop_max_paths=multi_hop_max_paths,
        dedup_against=dedup_against,
        graph_dir=graph_dir,
        rejection_feedback=rejection_feedback,
    )
    if resume:
        if not write:
            raise ValueError("resume requires write=True (it re-enters an existing bundle)")
        settings.apply_resume_meta(cast(dict[str, Any], load_journal_meta(resolved_out)))
    settings.validate()

    journal: ExtractionJournal | None = None
    if write:
        journal = _prepare_bundle_dir(resolved_out, settings, endpoint, resume)
    retrieval_store = _load_retrieval_store(settings.retrieval_index_dir) if write else None

    # Stages 1-3: inventory -> extract -> induce ontology.
    log = ProvenanceLog()
    complete = complete if complete is not None else build_complete(endpoint, log)
    adapter = extraction_adapter or LLMExtractionAdapter(
        complete,
        max_chars=settings.resolved_extract_max_chars,
        chunk_overlap=settings.resolved_extract_overlap,
        concurrency=settings.resolved_extract_concurrency,
        journal=journal,
    )
    docs = inventory_corpus(Path(settings.corpus_root))
    if settings.doc_limit is not None:
        docs = docs[: settings.doc_limit]
    extractions = extract_corpus(docs, adapter)
    ontology = induce_ontology(extractions)

    # Stages 4-6: seed/draft, optional graph walks, optional cross-bundle dedup.
    items, item_labels, seed_info, applied_feedback = _draft_stage(
        complete, docs, extractions, ontology, settings
    )
    items, item_labels, chain_items = _graph_stages(
        complete, docs, extractions, ontology, settings, items, item_labels
    )
    dedup_report: dict[str, object] | None = None
    if settings.dedup_against:
        items, item_labels, dedup_report = _dedup_stage(
            items, item_labels, dedup_against=settings.dedup_against, embedder=dedup_embedder
        )

    splits = assign_splits([it.id for it in items], seed=settings.seed)
    for it in items:
        it.split = cast(Split, splits[it.id])

    result = PipelineResult(
        out_dir=resolved_out,
        docs=docs,
        extractions=extractions,
        ontology=ontology,
        seeds=cast(list[DraftSeed], seed_info["seeds"]),
        items=items,
        chains=chain_items,
        corpus_root=Path(settings.corpus_root),
        elapsed_s=perf_counter() - started,
        item_labels=item_labels,
        coverage_report=cast("dict[str, object] | None", seed_info["coverage"]),
        dedup_report=dedup_report,
        applied_feedback=applied_feedback,
        log=log,
    )
    if write:
        _write_bundle(
            result,
            endpoint,
            settings.seed,
            settings.provenance_settings(resumed=resume),
            retrieval_store=retrieval_store,
            retrieval_k=settings.retrieval_k,
            drop_nonretrievable_needles=settings.drop_nonretrievable_needles,
        )
    return result
