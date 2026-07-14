"""Execution workflow for ontology-assisted gold-set drafting."""

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


def run_draft(request: DraftRequest) -> None:
    from llb.prep.frontier_telemetry import DraftBudgetExceeded
    from llb.prep.ontology.pipeline.run import draft_goldset

    resuming = request.resume is not None
    if request.resume is not None:
        request = DraftResumeBuilder.load(request).build()
    if request.corpus_root is None or not request.model:
        cli_error("provide --corpus-root and --model, or --resume <bundle>")

    adapter = _extraction_adapter(request.extractor, request.spacy_model)
    _validate_draft_inputs(
        request.drop_nonretrievable_needles,
        request.retrieval_index_dir,
        request.graph_dir,
        request.rejection_feedback,
    )
    dedup_against_dirs = _split_dir_list(request.dedup_against)
    if request.endpoint == "frontier":
        if not request.egress_consent:
            _confirm_frontier_egress(request.corpus_root, request.model)
            request.egress_consent = True
        request.max_calls = request.max_calls or 100
    elif request.max_usd is not None or request.max_calls is not None:
        cli_error("--max-usd and --max-calls are frontier-only guards")

    vllm_options = _VllmLaunchOptions(
        port=request.vllm_port,
        gpu_memory_utilization=request.vllm_gpu_memory_utilization,
        max_model_len=request.vllm_max_model_len or request.num_ctx,
        cpu_offload_gb=request.vllm_cpu_offload_gb,
        kv_offloading_size_gb=request.vllm_kv_offloading_size_gb,
        dtype=request.vllm_dtype,
        quantization=request.vllm_quantization,
        startup_timeout=request.vllm_startup_timeout,
    )
    endpoints, launched_vllm, resolved_out_dir = _endpoint_plan_setup(
        request.model,
        request.endpoint,
        request.backend,
        request.base_url,
        request.out_dir,
        request.num_ctx,
        vllm_options,
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
            request.corpus_root,
            endpoints,
            extraction_adapter=adapter,
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
            chains=request.chains,
            multi_hop_max_paths=request.multi_hop_max_paths,
            dedup_against=dedup_against_dirs,
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
    if request.verification_sample_size:
        _write_verification_sample(result.out_dir, request.verification_sample_size, request.seed)
    typer.echo(
        f"[prepare-goldset-draft] {len(result.items)} drafted items (verified=false; "
        f"endpoint={request.endpoint}, egress={endpoints.egress}) -> {result.out_dir}"
    )
    if request.require_passed_gates:
        _enforce_calibration_gates(result.calibration_report, result.out_dir)
