"""The `draft_goldset` orchestrator: run stages 1-7 in order and (by default) write the bundle.

    1 inventory -> 2 extract -> 3 induce ontology -> 4 sample coverage
    -> 5 draft QA -> 6 ground/dedup/reject -> 7 emit bundle

`complete` and `extraction_adapter` are injectable, so the whole flow is unit-tested with a fake
endpoint and never needs a server or a provider key.
"""

from pathlib import Path
from time import perf_counter
from typing import Any, cast

from llb.goldset.schema import Split
from llb.goldset.splits import assign_splits
from llb.prep.frontier_telemetry import DraftBudgetExceeded
from llb.prep.ontology.constants import DEFAULT_MAX_ITEMS, DEFAULT_MULTI_HOP_MAX_PATHS
from llb.prep.ontology.dedup import QuestionEmbedder
from llb.prep.ontology.endpoint import build_completers
from llb.prep.ontology.endpoint_config import EndpointCompleters, EndpointLogs, EndpointPlan
from llb.prep.ontology.extract import (
    ExtractionAdapter,
    LLMExtractionAdapter,
    extract_corpus,
)
from llb.prep.ontology.induce import induce_ontology
from llb.prep.ontology.inventory import inventory_corpus
from llb.prep.ontology.journal import ExtractionJournal
from llb.prep.ontology.models import DraftSeed
from llb.prep.ontology.pipeline.bundle import (
    _load_retrieval_store,
    _write_bundle,
    write_budget_abort,
)
from llb.prep.ontology.pipeline.journaling import (
    _prepare_bundle_dir,
    default_out_dir,
    load_journal_meta,
)
from llb.prep.ontology.pipeline.settings import DraftSettings, PipelineResult
from llb.prep.ontology.pipeline.stages import _dedup_stage, _draft_stage, _graph_stages


def draft_goldset(
    corpus_root: Path | str,
    endpoints: EndpointPlan,
    *,
    completers: EndpointCompleters | None = None,
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
    multi_hop_bridge_fill: bool = False,
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
        multi_hop_bridge_fill=multi_hop_bridge_fill,
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
        journal = _prepare_bundle_dir(resolved_out, settings, endpoints, resume)
    retrieval_store = _load_retrieval_store(settings.retrieval_index_dir) if write else None

    endpoint_logs = EndpointLogs()
    active = completers if completers is not None else build_completers(endpoints, endpoint_logs)
    try:
        result = _execute_pipeline(
            settings,
            active,
            endpoint_logs,
            resolved_out,
            journal,
            extraction_adapter,
            dedup_embedder,
            started,
        )
    except DraftBudgetExceeded as exc:
        if write:
            write_budget_abort(
                resolved_out,
                endpoints,
                endpoint_logs,
                settings.provenance_settings(resumed=resume),
                exc.reason,
                elapsed_s=perf_counter() - started,
            )
        raise
    if write:
        _write_bundle(
            result,
            endpoints,
            settings.seed,
            settings.provenance_settings(resumed=resume),
            retrieval_store=retrieval_store,
            retrieval_k=settings.retrieval_k,
            drop_nonretrievable_needles=settings.drop_nonretrievable_needles,
        )
    return result


def _execute_pipeline(
    settings: DraftSettings,
    completers: EndpointCompleters,
    endpoint_logs: EndpointLogs,
    out_dir: Path,
    journal: ExtractionJournal | None,
    extraction_adapter: ExtractionAdapter | None,
    dedup_embedder: QuestionEmbedder | None,
    started: float,
) -> PipelineResult:
    """Run the model stages after resumability and budget artifacts are ready."""
    adapter = extraction_adapter or LLMExtractionAdapter(
        completers.extraction,
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
    items, item_labels, seed_info, applied_feedback = _draft_stage(
        completers.drafting, docs, extractions, ontology, settings
    )
    items, item_labels, chain_items = _graph_stages(
        completers.drafting, docs, extractions, ontology, settings, items, item_labels
    )
    dedup_report: dict[str, object] | None = None
    if settings.dedup_against:
        items, item_labels, dedup_report = _dedup_stage(
            items,
            item_labels,
            dedup_against=settings.dedup_against,
            embedder=dedup_embedder,
        )
    splits = assign_splits([item.id for item in items], seed=settings.seed)
    for item in items:
        item.split = cast(Split, splits[item.id])
    return PipelineResult(
        out_dir=out_dir,
        docs=docs,
        extractions=extractions,
        ontology=ontology,
        seeds=cast(list[DraftSeed], seed_info["seeds"]),
        items=items,
        chains=chain_items,
        corpus_root=Path(settings.corpus_root),
        elapsed_s=perf_counter() - started,
        draft_attempts=len(cast(list[DraftSeed], seed_info["seeds"])),
        draft_parsed=cast(int, seed_info["draft_parsed"]),
        item_labels=item_labels,
        coverage_report=cast("dict[str, object] | None", seed_info["coverage"]),
        dedup_report=dedup_report,
        applied_feedback=applied_feedback,
        endpoint_logs=endpoint_logs,
    )
