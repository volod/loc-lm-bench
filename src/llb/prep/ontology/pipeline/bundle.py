"""Write the self-contained draft bundle and its provenance: the `verified=false` goldset, a copy
of the corpus, the induced ontology, the per-document extraction, calibration artifacts, and a
provenance record linking ontology / extraction / endpoint / prompt / model / cost / doc hashes.

`_write_bundle` is the emit stage; `_provenance` and `_prompt_fingerprints` build the traceability
record, and `_log_calibration_gates` surfaces the calibration roll-up as a WARNING (never fatal --
the human verification gate remains the real block on scoring).
"""

import hashlib
import json
import logging
from pathlib import Path

from llb.goldset.chains import dump_chains
from llb.goldset.schema import dump_goldset
from llb.prep.ontology.artifacts import (
    copy_pdf_citation_sidecars,
    required_gate_names,
    write_calibration_artifacts,
)
from llb.prep.ontology.constants import (
    CHAINS_FILENAME,
    CORPUS_DIRNAME,
    EXTRACTION_FILENAME,
    GOLDSET_FILENAME,
    ONTOLOGY_FILENAME,
    PDF_ONTOLOGY_REPORT_FILENAME,
    PROVENANCE_FILENAME,
    PROVENANCE_KIND,
)
from llb.prep.ontology.draft import draft_prompt
from llb.prep.ontology.endpoint import EndpointConfig
from llb.prep.ontology.extract import extraction_prompt
from llb.prep.ontology.models import DocRecord, DraftSeed
from llb.prep.ontology.needles import NeedleRetriever
from llb.prep.ontology.pipeline.settings import PipelineResult

_LOG = logging.getLogger(__name__)


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
