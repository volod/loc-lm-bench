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
from typing import cast

from llb.goldset.schema import GoldItem, Split, dump_goldset
from llb.goldset.splits import assign_splits
from llb.paths import resolve_data_dir
from llb.prep.frontier import LLMComplete, ProvenanceLog
from llb.prep.ontology.artifacts import copy_pdf_citation_sidecars, write_calibration_artifacts
from llb.prep.ontology.constants import (
    CORPUS_DIRNAME,
    DEFAULT_MAX_ITEMS,
    EXTRACT_CHUNK_OVERLAP,
    EXTRACT_MAX_CHARS,
    EXTRACTION_FILENAME,
    GOLDSET_FILENAME,
    METHOD_DIR,
    ONTOLOGY_FILENAME,
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
from llb.prep.ontology.models import (
    DocExtraction,
    DocRecord,
    DraftSeed,
    OntologyCandidate,
)
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


def _write_bundle(
    result: PipelineResult, endpoint: EndpointConfig, seed: int, settings: dict[str, object]
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
    )
    _LOG.info(
        "[ontology] wrote %d drafts (verified=false) + provenance -> %s",
        len(result.items),
        out_dir,
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
    write: bool = True,
) -> PipelineResult:
    """Run stages 1-7 and (by default) write the bundle. Returns the in-memory result."""
    started = perf_counter()
    if doc_limit is not None and doc_limit < 1:
        raise ValueError("doc_limit must be >= 1 when set")
    log = ProvenanceLog()
    complete = complete if complete is not None else build_complete(endpoint, log)
    adapter = extraction_adapter or LLMExtractionAdapter(
        complete,
        max_chars=extract_max_chars if extract_max_chars is not None else EXTRACT_MAX_CHARS,
        chunk_overlap=(
            extract_chunk_overlap if extract_chunk_overlap is not None else EXTRACT_CHUNK_OVERLAP
        ),
    )

    resolved_corpus_root = Path(corpus_root)
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

    resolved_out = Path(out_dir) if out_dir is not None else default_out_dir()
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
        "extract_max_chars": extract_max_chars
        if extract_max_chars is not None
        else EXTRACT_MAX_CHARS,
        "extract_chunk_overlap": extract_chunk_overlap
        if extract_chunk_overlap is not None
        else EXTRACT_CHUNK_OVERLAP,
    }
    if write:
        _write_bundle(result, endpoint, seed, settings)
    return result
