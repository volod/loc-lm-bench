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
from typing import Any, cast

from llb.goldset.schema import GoldItem, Split, dump_goldset
from llb.goldset.splits import assign_splits
from llb.paths import resolve_data_dir
from llb.prep.frontier import LLMComplete, ProvenanceLog
from llb.prep.ontology.artifacts import (
    copy_pdf_citation_sidecars,
    required_gate_names,
    write_calibration_artifacts,
)
from llb.prep.ontology.constants import (
    CORPUS_DIRNAME,
    DEFAULT_MAX_ITEMS,
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
from llb.prep.ontology.coverage import sample_seeds
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
    OntologyCandidate,
)
from llb.prep.ontology.needles import NeedleRetriever
from llb.prep.ontology.refine import refine_drafts

_LOG = logging.getLogger(__name__)
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


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
    elapsed_s: float = 0.0
    calibration_report: dict[str, object] | None = None
    log: ProvenanceLog = field(default_factory=ProvenanceLog)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime(_TIMESTAMP_FORMAT)


def default_out_dir() -> Path:
    return resolve_data_dir() / METHOD_DIR / _timestamp()


def _journal_meta_path(out_dir: Path) -> Path:
    return out_dir / EXTRACTION_JOURNAL_META_FILENAME


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
    extract_tmpl = extraction_prompt("<doc>", "<text>")
    draft_tmpl = draft_prompt(placeholder_seed, "<context>")
    return {
        "extraction": hashlib.sha256(extract_tmpl.encode("utf-8")).hexdigest(),
        "draft": hashlib.sha256(draft_tmpl.encode("utf-8")).hexdigest(),
    }


def _write_corpus_copy(source_root: Path, corpus_dir: Path, docs: list[DocRecord]) -> None:
    """Copy inventoried docs verbatim so spans stay exact and the bundle self-validates."""
    for doc in docs:
        target = corpus_dir / doc.doc_id
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(doc.text, encoding="utf-8")
    copy_pdf_citation_sidecars(source_root, corpus_dir, [doc.doc_id for doc in docs])


def _provenance(
    result: PipelineResult, endpoint: EndpointConfig, seed: int, settings: dict[str, object]
) -> dict[str, object]:
    return {
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
            "items": len(result.items),
        },
        "ontology": result.ontology.model_dump(),
        "n_items": len(result.items),
        "cost": result.log.summary(),
    }


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
    )
    _LOG.info(
        "[ontology] wrote %d drafts (verified=false) + provenance -> %s",
        len(result.items),
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
    write: bool = True,
    resume: bool = False,
) -> PipelineResult:
    """Run stages 1-7 and (by default) write the bundle. Returns the in-memory result.

    `resume=True` re-enters an existing bundle: it reads the pinned settings from the journal meta,
    reuses journaled extraction windows instead of re-calling the model, and replays the
    deterministic seed/draft/emit stages -- producing the same bundle as an uninterrupted run.
    """
    started = perf_counter()
    resolved_out = Path(out_dir) if out_dir is not None else default_out_dir()
    if resume:
        if not write:
            raise ValueError("resume requires write=True (it re-enters an existing bundle)")
        meta = cast(dict[str, Any], load_journal_meta(resolved_out))
        corpus_root = str(meta.get("corpus_root", corpus_root))
        seed = int(meta.get("seed", seed))
        max_items = int(meta.get("max_items", max_items))
        meta_doc_limit = meta.get("doc_limit", doc_limit)
        doc_limit = int(meta_doc_limit) if meta_doc_limit is not None else None
        extract_max_chars = meta.get("extract_max_chars", extract_max_chars)
        extract_chunk_overlap = meta.get("extract_chunk_overlap", extract_chunk_overlap)
        extract_concurrency = meta.get("extract_concurrency", extract_concurrency)
        meta_index_dir = meta.get("retrieval_index_dir")
        retrieval_index_dir = meta_index_dir if meta_index_dir is not None else retrieval_index_dir
        retrieval_k = int(meta.get("retrieval_k", retrieval_k))
        drop_nonretrievable_needles = bool(
            meta.get("drop_nonretrievable_needles", drop_nonretrievable_needles)
        )
    if doc_limit is not None and doc_limit < 1:
        raise ValueError("doc_limit must be >= 1 when set")
    if extract_concurrency is not None and extract_concurrency < 1:
        raise ValueError("extract_concurrency must be >= 1 when set")
    if retrieval_k < 1:
        raise ValueError("retrieval_k must be >= 1")

    resolved_max_chars = extract_max_chars if extract_max_chars is not None else EXTRACT_MAX_CHARS
    resolved_overlap = (
        extract_chunk_overlap if extract_chunk_overlap is not None else EXTRACT_CHUNK_OVERLAP
    )
    resolved_concurrency = (
        extract_concurrency if extract_concurrency is not None else EXTRACT_CONCURRENCY
    )

    resolved_corpus_root = Path(corpus_root)
    journal: ExtractionJournal | None = None
    if write:
        resolved_out.mkdir(parents=True, exist_ok=True)
        if not resume:
            _write_journal_meta(
                resolved_out,
                {
                    "corpus_root": str(resolved_corpus_root),
                    "seed": seed,
                    "max_items": max_items,
                    "doc_limit": doc_limit,
                    "extract_max_chars": resolved_max_chars,
                    "extract_chunk_overlap": resolved_overlap,
                    "extract_concurrency": resolved_concurrency,
                    "retrieval_index_dir": str(retrieval_index_dir)
                    if retrieval_index_dir is not None
                    else None,
                    "retrieval_k": retrieval_k,
                    "drop_nonretrievable_needles": drop_nonretrievable_needles,
                },
                endpoint,
            )
        journal = ExtractionJournal(resolved_out / EXTRACTION_JOURNAL_FILENAME)
        journal.load()

    retrieval_store = _load_retrieval_store(retrieval_index_dir) if write else None
    log = ProvenanceLog()
    complete = complete if complete is not None else build_complete(endpoint, log)
    adapter = extraction_adapter or LLMExtractionAdapter(
        complete,
        max_chars=resolved_max_chars,
        chunk_overlap=resolved_overlap,
        concurrency=resolved_concurrency,
        journal=journal,
    )

    docs = inventory_corpus(resolved_corpus_root)
    if doc_limit is not None:
        docs = docs[:doc_limit]
    extractions = extract_corpus(docs, adapter)
    ontology = induce_ontology(extractions)
    seeds = sample_seeds(docs, extractions, max_items=max_items, seed=seed)
    raw_drafts = draft_items(complete, docs, seeds, ontology_constraints(ontology))
    items = refine_drafts(docs, raw_drafts)

    splits = assign_splits([it.id for it in items], seed=seed)
    for it in items:
        it.split = cast(Split, splits[it.id])

    result = PipelineResult(
        out_dir=resolved_out,
        docs=docs,
        extractions=extractions,
        ontology=ontology,
        seeds=seeds,
        items=items,
        corpus_root=resolved_corpus_root,
        elapsed_s=perf_counter() - started,
        log=log,
    )
    settings: dict[str, object] = {
        "max_items": max_items,
        "seed": seed,
        "doc_limit": doc_limit,
        "extract_max_chars": resolved_max_chars,
        "extract_chunk_overlap": resolved_overlap,
        "extract_concurrency": resolved_concurrency,
        "needle_retrieval_index_dir": str(retrieval_index_dir)
        if retrieval_index_dir is not None
        else None,
        "needle_retrieval_k": retrieval_k,
        "drop_nonretrievable_needles": drop_nonretrievable_needles,
        "resumed": resume,
    }
    if write:
        _write_bundle(
            result,
            endpoint,
            seed,
            settings,
            retrieval_store=retrieval_store,
            retrieval_k=retrieval_k,
            drop_nonretrievable_needles=drop_nonretrievable_needles,
        )
    return result
