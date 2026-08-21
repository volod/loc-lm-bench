"""Execution workflow for ontology-assisted gold-set drafting."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from llb.cli.helpers import cli_error
from llb.cli.prep.draft_endpoints import (
    _VllmLaunchOptions,
    _confirm_frontier_egress,
    _endpoint_plan_setup,
)
from llb.cli.prep.draft_request import DraftRequest
from llb.cli.prep.draft_resume import DraftResumeBuilder
from llb.cli.prep.draft_support import (
    _enforce_calibration_gates,
    _extraction_adapter,
    _split_dir_list,
    _validate_draft_inputs,
    _write_verification_sample,
)


@dataclass(frozen=True, slots=True)
class _ResolvedDraft:
    """The request this run executes, with the two fields everything downstream requires.

    `corpus_root` and `model` are carried separately because a resume fills them in: reading them
    off the request again would hand every downstream call an optional that was already checked.
    """

    request: DraftRequest
    resuming: bool
    corpus_root: Path
    model: str


def _resolved_request(request: DraftRequest) -> _ResolvedDraft:
    """The request this run actually executes, and whether it is a RESUME of an earlier bundle."""
    resuming = request.resume is not None
    if resuming:
        request = DraftResumeBuilder.load(request).build()
    if request.corpus_root is None or not request.model:
        cli_error("provide --corpus-root and --model, or --resume <bundle>")
    return _ResolvedDraft(
        request=request, resuming=resuming, corpus_root=request.corpus_root, model=request.model
    )


def _apply_egress_policy(resolved: _ResolvedDraft) -> None:
    """A frontier endpoint needs consent and a spend guard; a local one may not carry either."""
    request = resolved.request
    if request.endpoint == "frontier":
        if not request.egress_consent:
            _confirm_frontier_egress(resolved.corpus_root, resolved.model)
            request.egress_consent = True
        request.max_calls = request.max_calls or 100
        return
    if request.max_usd is not None or request.max_calls is not None:
        cli_error("--max-usd and --max-calls are frontier-only guards")


def _vllm_options(request: DraftRequest) -> _VllmLaunchOptions:
    """The launch settings a local vLLM endpoint is started with, if one is started at all."""
    return _VllmLaunchOptions(
        port=request.vllm_port,
        gpu_memory_utilization=request.vllm_gpu_memory_utilization,
        max_model_len=request.vllm_max_model_len or request.num_ctx,
        cpu_offload_gb=request.vllm_cpu_offload_gb,
        kv_offloading_size_gb=request.vllm_kv_offloading_size_gb,
        dtype=request.vllm_dtype,
        quantization=request.vllm_quantization,
        startup_timeout=request.vllm_startup_timeout,
    )


def _report_draft(request: DraftRequest, result: Any, endpoints: Any) -> None:
    """Write the verification sample, say what was drafted, and enforce the calibration gates."""
    if request.verification_sample_size or request.derive_verification_sample:
        _write_verification_sample(
            result.out_dir,
            request.verification_sample_size or None,
            request.seed,
            confidence=request.verification_sample_confidence,
            precision=request.verification_sample_precision,
        )
    typer.echo(
        f"[prepare-goldset-draft] {len(result.items)} drafted items (verified=false; "
        f"endpoint={request.endpoint}, egress={endpoints.egress}) -> {result.out_dir}"
    )
    if request.require_passed_gates:
        _enforce_calibration_gates(result.calibration_report, result.out_dir)


def run_draft(request: DraftRequest) -> None:
    from llb.prep.frontier.telemetry import DraftBudgetExceeded
    from llb.prep.ontology.pipeline.run import draft_goldset

    resolved = _resolved_request(request)
    request, resuming = resolved.request, resolved.resuming
    adapter = _extraction_adapter(request.extractor, request.spacy_model)
    dedup_against_dirs = _split_dir_list(request.dedup_against)
    _validate_draft_inputs(
        request.drop_nonretrievable_needles,
        request.retrieval_index_dir,
        request.graph_dir,
        request.rejection_feedback,
        request.reuse_extraction_bundle,
        request.multi_hop,
        request.multi_hop_only,
        request.carry_forward_multi_hop,
        dedup_against_dirs,
    )
    _apply_egress_policy(resolved)
    endpoints, launched_vllm, resolved_out_dir = _endpoint_plan_setup(
        resolved.model,
        request.endpoint,
        request.backend,
        request.base_url,
        request.out_dir,
        request.num_ctx,
        _vllm_options(request),
        frontier_stage=request.frontier_stage,
        local_model=request.local_model,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        timeout=request.timeout,
        no_think=request.no_think,
        egress_consent=request.egress_consent,
        max_usd=request.max_usd,
        max_calls=request.max_calls,
    )
    try:
        result = draft_goldset(
            resolved.corpus_root,
            endpoints,
            extraction_adapter=adapter,
            reuse_extraction_bundle=request.reuse_extraction_bundle,
            max_items=request.max_items,
            seed=request.seed,
            out_dir=resolved_out_dir,
            doc_limit=request.doc_limit,
            extract_max_chars=request.extract_max_chars,
            extract_chunk_overlap=request.extract_chunk_overlap,
            extract_concurrency=request.concurrency,
            retrieval_index_dir=request.retrieval_index_dir,
            retrieval_k=request.retrieval_k,
            drop_nonretrievable_needles=request.drop_nonretrievable_needles,
            coverage_target=request.coverage_target,
            multi_hop=request.multi_hop,
            multi_hop_only=request.multi_hop_only,
            chains=request.chains,
            multi_hop_max_paths=request.multi_hop_max_paths,
            multi_hop_bridge_fill=request.multi_hop_bridge_fill,
            multi_hop_path_stratified=request.multi_hop_path_stratified,
            multi_hop_relation_pair_target=request.multi_hop_relation_pair_target,
            multi_hop_document_mode_target=request.multi_hop_document_mode_target,
            multi_hop_source_document_target=request.multi_hop_source_document_target,
            dedup_against=dedup_against_dirs,
            carry_forward_multi_hop=request.carry_forward_multi_hop,
            graph_dir=request.graph_dir,
            rejection_feedback=request.rejection_feedback,
            resume=resuming,
        )
    except DraftBudgetExceeded as exc:
        target = resolved_out_dir or request.out_dir or request.resume
        cli_error(f"{exc.reason}; partial bundle and abort provenance: {target}", code=1)
    finally:
        if launched_vllm is not None:
            launched_vllm.stop()
    _report_draft(request, result, endpoints)
