"""Run inventory, extraction, ontology, sampling, drafting, refinement, and bundle emission.

Model and extraction adapters are injectable so tests exercise the complete flow without a server.
"""

from pathlib import Path
from time import perf_counter
from typing import Any, cast

from llb.goldset.schema import GoldItem, Split
from llb.goldset.splits import assign_splits
from llb.prep.frontier_telemetry import DraftBudgetExceeded
from llb.prep.ontology.constants import (
    DEFAULT_MAX_ITEMS,
    DEFAULT_MULTI_HOP_DOCUMENT_MODE_TARGET,
    DEFAULT_MULTI_HOP_MAX_PATHS,
    DEFAULT_MULTI_HOP_RELATION_PAIR_TARGET,
    DEFAULT_MULTI_HOP_SOURCE_DOCUMENT_TARGET,
)
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
from llb.prep.ontology.models import DraftSeed, ItemLabels
from llb.prep.ontology.pipeline.bundle import (
    _load_retrieval_store,
    _write_bundle,
    write_budget_abort,
)
from llb.prep.ontology.pipeline.deduplication import deduplicate_drafts
from llb.prep.ontology.pipeline.journaling import (
    _prepare_bundle_dir,
    default_out_dir,
    load_journal_meta,
)
from llb.prep.ontology.pipeline.settings import DraftSettings, PipelineResult
from llb.prep.ontology.pipeline.stages import _draft_stage, _graph_stages


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
    reuse_extraction_bundle: Path | str | None = None,
    retrieval_index_dir: Path | str | None = None,
    retrieval_k: int = 10,
    drop_nonretrievable_needles: bool = False,
    coverage_target: int | None = None,
    multi_hop: bool = False,
    multi_hop_only: bool = False,
    chains: bool = False,
    multi_hop_max_paths: int = DEFAULT_MULTI_HOP_MAX_PATHS,
    multi_hop_bridge_fill: bool = False,
    multi_hop_path_stratified: bool = False,
    multi_hop_relation_pair_target: int = DEFAULT_MULTI_HOP_RELATION_PAIR_TARGET,
    multi_hop_document_mode_target: int = DEFAULT_MULTI_HOP_DOCUMENT_MODE_TARGET,
    multi_hop_source_document_target: int = DEFAULT_MULTI_HOP_SOURCE_DOCUMENT_TARGET,
    dedup_against: list[Path | str] | None = None,
    carry_forward_multi_hop: bool = False,
    graph_dir: Path | str | None = None,
    dedup_embedder: QuestionEmbedder | None = None,
    rejection_feedback: Path | str | None = None,
    write: bool = True,
    resume: bool = False,
) -> PipelineResult:
    """Run the draft stages and return the in-memory result, writing its bundle by default.

    Coverage, graph paths, prior-bundle dedup, rejection feedback, and resumable journal replay are
    configured by the keyword arguments and recorded through `DraftSettings`.
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
        reuse_extraction_bundle=reuse_extraction_bundle,
        retrieval_index_dir=retrieval_index_dir,
        retrieval_k=retrieval_k,
        drop_nonretrievable_needles=drop_nonretrievable_needles,
        coverage_target=coverage_target,
        multi_hop=multi_hop,
        multi_hop_only=multi_hop_only,
        chains=chains,
        multi_hop_max_paths=multi_hop_max_paths,
        multi_hop_bridge_fill=multi_hop_bridge_fill,
        multi_hop_path_stratified=multi_hop_path_stratified,
        multi_hop_relation_pair_target=multi_hop_relation_pair_target,
        multi_hop_document_mode_target=multi_hop_document_mode_target,
        multi_hop_source_document_target=multi_hop_source_document_target,
        dedup_against=dedup_against,
        carry_forward_multi_hop=carry_forward_multi_hop,
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
    docs = inventory_corpus(Path(settings.corpus_root))
    if settings.doc_limit is not None:
        docs = docs[: settings.doc_limit]
    if settings.reuse_extraction_bundle is not None:
        from llb.prep.ontology.pipeline.expansion import reused_extractions

        extractions = reused_extractions(settings.reuse_extraction_bundle, docs)
    else:
        adapter = extraction_adapter or LLMExtractionAdapter(
            completers.extraction,
            max_chars=settings.resolved_extract_max_chars,
            chunk_overlap=settings.resolved_extract_overlap,
            concurrency=settings.resolved_extract_concurrency,
            journal=journal,
        )
        extractions = extract_corpus(docs, adapter)
    ontology = induce_ontology(extractions)
    if settings.multi_hop_only:
        items: list[GoldItem] = []
        item_labels: dict[str, ItemLabels] = {}
        seed_info: dict[str, object] = {"seeds": [], "coverage": None, "draft_parsed": 0}
        applied_feedback = None
    else:
        items, item_labels, seed_info, applied_feedback = _draft_stage(
            completers.drafting, docs, extractions, ontology, settings
        )
    items, item_labels, chain_items, path_strata = _graph_stages(
        completers.drafting, docs, extractions, ontology, settings, items, item_labels
    )
    dedup_report: dict[str, object] | None = None
    if settings.dedup_against:
        items, item_labels, dedup_report = deduplicate_drafts(
            items,
            item_labels,
            dedup_against=settings.dedup_against,
            embedder=dedup_embedder,
        )
        if settings.multi_hop:
            from llb.prep.ontology.pipeline.expansion import prior_multihop_span_pairs

            dedup_report["excluded_prior_span_pairs"] = len(
                prior_multihop_span_pairs(settings.dedup_against)
            )
    carry_forward_report: dict[str, object] | None = None
    if settings.carry_forward_multi_hop:
        from llb.prep.ontology.pipeline.expansion import carry_forward_multi_hop

        items, item_labels, carry_forward_report = carry_forward_multi_hop(
            items, item_labels, settings.dedup_against or [], docs
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
        carry_forward_report=carry_forward_report,
        applied_feedback=applied_feedback,
        multi_hop_path_strata=path_strata,
        endpoint_logs=endpoint_logs,
    )
