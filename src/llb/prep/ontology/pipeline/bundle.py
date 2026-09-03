"""Write the self-contained draft bundle, calibration artifacts, and traceable provenance.

Calibration failures are warnings; human verification remains the scoring gate.
"""

import json
from typing import TYPE_CHECKING
from pathlib import Path

from llb.artifacts.serialization import stated_sections
from llb.goldset.chains import dump_chains
from llb.goldset.schema import dump_goldset
from llb.goldset.span_occurrences import span_occurrence_counts, write_occurrences_sidecar
from llb.prep.ontology.artifacts.citations import copy_pdf_citation_sidecars
from llb.prep.ontology.artifacts.contracts import extraction_record, ontology_record
from llb.prep.ontology.artifacts.report import write_calibration_artifacts
from llb.core.contracts.data_prep.ontology import (
    OntologyProvenance,
)
from llb.prep.ontology.constants import (
    CHAINS_FILENAME,
    CORPUS_DIRNAME,
    EXTRACTION_FILENAME,
    GOLDSET_FILENAME,
    MULTI_HOP_PATH_STRATA_FILENAME,
    ONTOLOGY_FILENAME,
    PROVENANCE_FILENAME,
    PROVENANCE_KIND,
    PROVENANCE_SCHEMA_ID,
    PROVENANCE_SCHEMA_VERSION,
)
from llb.prep.ontology.endpoints.config import EndpointPlan, endpoint_provenance
from llb.prep.ontology.models import DocRecord
from llb.prep.ontology.drafting.needles import NeedleRetriever
from llb.prep.ontology.pipeline.bundle_provenance import provenance_payload
from llb.prep.ontology.pipeline.settings import PipelineResult

if TYPE_CHECKING:
    from llb.prep.ontology.endpoints.config import EndpointLogs
from llb.prep.ontology.pipeline.bundle_logging import _LOG, _log_calibration_gates


def write_budget_abort(
    out_dir: Path,
    endpoints: EndpointPlan,
    logs: "EndpointLogs",
    settings: dict[str, object],
    reason: str,
    *,
    elapsed_s: float,
) -> None:
    """Leave a machine-readable abort record beside the resumable extraction state."""
    record = OntologyProvenance(
        schema_id=PROVENANCE_SCHEMA_ID,
        schema_version=PROVENANCE_SCHEMA_VERSION,
        kind=PROVENANCE_KIND,
        synthetic=False,
        status="aborted",
        abort={"reason": reason, "resumable": True},
        endpoint=endpoint_provenance(endpoints, logs),
        settings=settings,
        elapsed_s=round(elapsed_s, 3),
        cost=logs.summary(),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_provenance(out_dir, record)


def _write_provenance(out_dir: Path, record: OntologyProvenance) -> None:
    """Write the bundle record, stating only what the run knows.

    An absent stage stays absent: a run that never reached the coverage pass says nothing about
    coverage rather than recording a null that reads as "no coverage". A document row's `null`
    acquisition fields are kept, because there the absence is the record.
    """
    (out_dir / PROVENANCE_FILENAME).write_text(
        json.dumps(stated_sections(record), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_corpus_copy(source_root: Path, corpus_dir: Path, docs: list[DocRecord]) -> None:
    """Copy inventoried docs verbatim so spans stay exact and the bundle self-validates."""
    for doc in docs:
        target = corpus_dir / doc.doc_id
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(doc.text, encoding="utf-8")
    copy_pdf_citation_sidecars(source_root, corpus_dir, [doc.doc_id for doc in docs])


def _load_retrieval_store(index_dir: Path | str | None) -> NeedleRetriever | None:
    if index_dir is None:
        return None
    from llb.rag.vector_store.store import RagStore

    return RagStore.load(index_dir)


def _write_bundle(
    result: PipelineResult,
    endpoints: EndpointPlan,
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
    # Draft-time ambiguous-evidence guard: flag items whose gold span repeats verbatim elsewhere in
    # the corpus, so the verification worksheet can show the count. No file when all spans are unique.
    write_occurrences_sidecar(
        out_dir, span_occurrence_counts(result.items, {doc.doc_id: doc.text for doc in result.docs})
    )
    (out_dir / ONTOLOGY_FILENAME).write_text(
        json.dumps(ontology_record(result.ontology).model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out_dir / EXTRACTION_FILENAME).open("w", encoding="utf-8") as fh:
        for extraction in result.extractions:
            row = extraction_record(extraction).model_dump()
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_provenance(out_dir, provenance_payload(result, endpoints, seed, settings))
    if result.multi_hop_path_strata is not None:
        (out_dir / MULTI_HOP_PATH_STRATA_FILENAME).write_text(
            json.dumps(
                result.multi_hop_path_strata,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
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
