"""RAG run-eval command (retrieve -> generate -> score)."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config, resolve_registered_adapter


def _parse_query_prep(steps: Optional[str]) -> Optional[list[str]]:
    """Parse a comma-separated --query-prep list into ordered steps (None leaves the config)."""
    if steps is None:
        return None
    return [step.strip() for step in steps.split(",") if step.strip()]


@app.command("run-eval")
def run_eval_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    model: Optional[str] = typer.Option(None, help="model name (Ollama tag or HF repo id)"),
    backend: Optional[str] = typer.Option(None, help="ollama | vllm | llamacpp"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    adapter: Optional[str] = typer.Option(
        None,
        "--adapter",
        help="registered adapter id, id prefix, or label (`llb list-adapters`); the contamination "
        "guard then reads the registry's recorded digests, not the adapter directory's manifest",
    ),
    max_model_len: Optional[int] = typer.Option(
        None, help="vLLM/llama.cpp served context window (overrides the config; no YAML needed)"
    ),
    gpu_memory_utilization: Optional[float] = typer.Option(
        None, help="vLLM GPU memory fraction 0-1 (overrides the config; no YAML needed)"
    ),
    gpu_layers: Optional[int] = typer.Option(
        None,
        "--gpu-layers",
        help="llama.cpp GPU/CPU layer split (-1 == all on GPU; a smaller value forces a "
        "partial offload to system RAM)",
    ),
    split: str = typer.Option("final", help="gold split to evaluate"),
    limit: Optional[int] = typer.Option(None, help="cap the number of eval items"),
    judge_rho: Optional[float] = typer.Option(
        None, help="calibration Spearman rho; judge stays demoted below the threshold"
    ),
    judge_model: Optional[str] = typer.Option(
        None, help="local or frontier judge model id (lane selected by --scorer-policy)"
    ),
    judge_base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible judge endpoint, e.g. http://localhost:8000/v1"
    ),
    scorer_policy: Optional[str] = typer.Option(
        None,
        "--scorer-policy",
        help="judge lane: human | local (default) | frontier (budget-capped litellm)",
    ),
    scorer_egress_consent: bool = typer.Option(
        False,
        "--scorer-egress-consent",
        help="frontier lane: record explicit consent to send answers to the frontier judge",
    ),
    frontier_max_usd: Optional[float] = typer.Option(
        None, help="frontier lane: hard USD spend cap for the scorer cost ledger"
    ),
    frontier_max_calls: Optional[int] = typer.Option(
        None, help="frontier lane: hard call-count cap for the scorer cost ledger"
    ),
    retrieval_backend: Optional[str] = typer.Option(
        None,
        help="faiss (default vector store) | graph (GraphRAG backend) | fused (vector + graph)",
    ),
    retrieval_strategy: Optional[str] = typer.Option(
        None, help="graph backend strategy: local_khop | global_community"
    ),
    retrieval_mode: Optional[str] = typer.Option(
        None,
        help="flat | parent_child | hybrid (hybrid fuses dense + lexical BM25 rankings; the "
        "index must be built with `build-index --retrieval-mode hybrid`)",
    ),
    acl: Optional[str] = typer.Option(
        None,
        "--acl",
        help="restrict RAG retrieval to chunks whose governance metadata has this ACL label",
    ),
    fusion_weight: Optional[float] = typer.Option(
        None, help="hybrid mode: dense share of the weighted RRF, 0..1 (default 0.5)"
    ),
    fusion_candidates: Optional[int] = typer.Option(
        None, help="hybrid mode: per-side candidate depth fed into the fusion (default 50)"
    ),
    graph_weight: Optional[float] = typer.Option(
        None, help="fused backend: graph share of weighted RRF, 0..1 (default 0.3)"
    ),
    graph_fusion_candidates: Optional[int] = typer.Option(
        None,
        help="fused backend: per-lane candidate depth fused before the top_k cut "
        "(default: top_k, i.e. each graph candidate that enters displaces a vector one)",
    ),
    reranker: Optional[str] = typer.Option(
        None,
        help="local cross-encoder reranker (HF id, e.g. BAAI/bge-reranker-v2-m3): retrieve "
        "--rerank-candidates, rerank, keep top_k (off by default)",
    ),
    rerank_candidates: Optional[int] = typer.Option(
        None, help="candidate pool depth fed into the reranker before the top_k cut (default 30)"
    ),
    context_order: Optional[str] = typer.Option(
        None,
        help="how kept chunks are laid into the prompt: rank (best-first, default) | "
        "reverse_rank (best-last)",
    ),
    query_prep: Optional[str] = typer.Option(
        None,
        "--query-prep",
        help="opt-in query-side lane (uk-query-processing): comma-separated ordered steps "
        "normalize,typos,glossary,rewrite,hyde,decompose (last three call the local model; "
        "off by default). "
        "The raw query is always preserved; only the retrieval query is transformed",
    ),
    query_glossary: Optional[Path] = typer.Option(
        None,
        help="query_glossary.json for the query-prep 'glossary' step (build-query-glossary)",
    ),
    query_prep_typo_guard: bool = typer.Option(
        False,
        "--query-prep-typo-guard",
        help="typos step: leave an OOV token pymorphy3 knows as a valid Ukrainian word form "
        "unchanged (an inflection is not a misspelling)",
    ),
    score_semantic: Optional[bool] = typer.Option(
        None,
        "--score-semantic/--no-score-semantic",
        help="enable or disable the embedding-similarity correctness signal",
    ),
    cited_answers: Optional[bool] = typer.Option(
        None,
        "--cited-answers/--no-cited-answers",
        help="require [i] chunk citations in the generation prompt and score citation validity + "
        "hallucinated-citation rate (groundedness-citation-metrics)",
    ),
    score_groundedness: Optional[bool] = typer.Option(
        None,
        "--score-groundedness/--no-score-groundedness",
        help="record the deterministic groundedness fraction (share of answer claims supported by "
        "the retrieved context) as an additive per-case column",
    ),
    insufficient_context_probes: Optional[int] = typer.Option(
        None,
        help="re-run N sampled gold items with their gold evidence excluded from retrieval and "
        "score abstention accuracy (probe cases never enter the correctness aggregates)",
    ),
    telemetry: Optional[bool] = typer.Option(
        None,
        "--telemetry/--no-telemetry",
        help="enable or disable steady-state throughput and peak-VRAM telemetry",
    ),
    worksheet: Optional[Path] = typer.Option(
        None,
        help="emit a judge-calibration worksheet pre-filled with answers "
        "(pair with --split calibration)",
    ),
    prompt_system: Optional[str] = typer.Option(
        None,
        help="prompt-system id to prepend to the baseline RAG generation prompt",
    ),
    prompt_package: Optional[Path] = typer.Option(
        None,
        help=(
            "prompt-system run dir, candidates.json, or compact <run_dir>/<id>; "
            "defaults to searching DATA_DIR/prompt-system"
        ),
    ),
    evict: bool = typer.Option(
        False, help="vLLM contention guard: unload Ollama's resident models before launching"
    ),
    wait: bool = typer.Option(
        False, help="vLLM contention guard: wait for VRAM to free instead of derating immediately"
    ),
    resume: Optional[Path] = typer.Option(
        None,
        help="resume an interrupted run from its journal (pass the run dir); config + goldset "
        "must match the interrupted run",
    ),
    max_case_retries: int = typer.Option(
        2, help="transient per-case retries (timeout / backend_error) before giving up on a case"
    ),
    retry_backoff_s: float = typer.Option(
        1.0, help="base seconds for the capped exponential per-case retry backoff"
    ),
) -> None:
    """Run the skeleton on one model and print a ranked row + write the manifest."""
    from llb.executor.runner import run_eval
    from llb.prompt_system.selection import (
        prompt_system_id_from_package_path,
        resolve_prompt_package,
    )

    cfg = load_config(
        config,
        model=model,
        backend=backend,
        goldset_path=goldset,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        n_gpu_layers=gpu_layers,
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        scorer_policy=scorer_policy,
        scorer_egress_consent=scorer_egress_consent or None,
        frontier_max_usd=frontier_max_usd,
        frontier_max_calls=frontier_max_calls,
        retrieval_backend=retrieval_backend,
        retrieval_strategy=retrieval_strategy,
        retrieval_mode=retrieval_mode,
        acl_label=acl,
        fusion_weight=fusion_weight,
        fusion_candidates=fusion_candidates,
        graph_weight=graph_weight,
        graph_fusion_candidates=graph_fusion_candidates,
        reranker=reranker,
        rerank_candidates=rerank_candidates,
        context_order=context_order,
        query_prep=_parse_query_prep(query_prep),
        query_glossary_path=query_glossary,
        query_prep_typo_guard=query_prep_typo_guard or None,
        score_semantic=score_semantic,
        cited_answers=cited_answers,
        score_groundedness=score_groundedness,
        insufficient_context_probes=insufficient_context_probes,
        measure_telemetry=telemetry,
    )
    if adapter is not None:
        cfg = cfg.with_overrides(adapter_path=resolve_registered_adapter(cfg.data_dir, adapter))
    selected_prompt = None
    prompt_id = prompt_system or prompt_system_id_from_package_path(prompt_package)
    if prompt_id is not None:
        selected_prompt = resolve_prompt_package(cfg.data_dir, prompt_id, prompt_package)
    run_eval(
        cfg,
        split=split,
        limit=limit,
        judge_rho=judge_rho,
        worksheet=worksheet,
        evict=evict,
        wait=wait,
        resume=resume,
        max_case_retries=max_case_retries,
        retry_backoff_s=retry_backoff_s,
        prompt_package=selected_prompt.package if selected_prompt is not None else None,
        prompt_system_provenance=(
            selected_prompt.provenance if selected_prompt is not None else None
        ),
    )
