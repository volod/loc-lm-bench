"""RAG run-eval command (retrieve -> generate -> score)."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.eval.run_config import config_overrides
from llb.cli.eval.run_execution import execute_eval
from llb.cli.helpers import load_config


@app.command("run-eval")
def run_eval_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    corpus_root: Optional[Path] = typer.Option(
        None, help="corpus directory whose persisted index should be queried"
    ),
    model: Optional[str] = typer.Option(None, help="model name (Ollama tag or HF repo id)"),
    backend: Optional[str] = typer.Option(None, help="ollama | vllm | llamacpp"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    adapter: Optional[str] = typer.Option(
        None,
        "--adapter",
        help="registered adapter id, id prefix, or label (`llb list-adapters`); the contamination "
        "guard then reads the registry's recorded digests, not the adapter directory's manifest",
    ),
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        help="completion token budget per case (overrides the config). Raise it for "
        "--answer-format envelope: a typed answer is several times longer than a short free-text "
        "one, and a completion cut off at the cap is scored as a format failure",
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
    graph_fusion_span_identity: Optional[str] = typer.Option(
        None,
        help="fused backend: span identity the lanes are fused by -- 'exact' (default, identical "
        "offsets) or 'overlap' (fold a graph span into the vector chunk that contains it)",
    ),
    graph_fusion_span_merge_ratio: Optional[float] = typer.Option(
        None,
        help="fused backend: share of the shorter span a folding span identity needs covered "
        "before two spans are one candidate (default 0.5; 1.0 == containment only, dead under "
        "'exact')",
    ),
    graph_fusion_router: Optional[str] = typer.Option(
        None,
        help="fused backend: 'fixed' (default) or 'question_type' (sidecar label with a "
        "deterministic question-text fallback)",
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
    restore_table_headers: Optional[bool] = typer.Option(
        None,
        "--restore-table-headers/--no-restore-table-headers",
        help="prompt-side only: prepend a table chunk's recorded header row to it in the PROMPT "
        "when the chunk does not already carry it (needs --strategy table); stored chunks, "
        "offsets, and retrieval metrics are unchanged",
    ),
    context_strategy: Optional[str] = typer.Option(
        None,
        help="context lane (rag-vs-long-context-ablation): rag (retrieve, the default and the "
        "leaderboard row) | closed_book (no context; the model answers from its weights) "
        "| retrieved_document (retrieve, then send the whole document the top chunk came from) "
        "| long_context (the item's whole GOLD document; DIAGNOSTIC, it reads the gold label). "
        "Both document lanes skip an item whose documents do not fit",
    ),
    retrieved_document_top_n: Optional[int] = typer.Option(
        None,
        min=1,
        help="how many DISTINCT retrieved documents the retrieved_document lane lays in, walking "
        "the ranked chunk list best-first (default 1)",
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
    query_prep_dense_case: bool = typer.Option(
        False,
        "--query-prep-dense-case",
        help="normalize step: send the raw question's capitalization to the CASE-SENSITIVE dense "
        "encoder while the lexical lane keeps the casefolded text",
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
    answer_format: Optional[str] = typer.Option(
        None,
        "--answer-format",
        help="free_text (default) | envelope -- ask for the typed answer contract, validate it at "
        "the generation boundary, and read the answer-side signals from declared fields "
        "(typed-rag-answer-envelope)",
    ),
    answer_validation: Optional[str] = typer.Option(
        None,
        "--answer-validation",
        help="off (default) | ontology -- check the declared answer's triples against the SIGNED "
        "axiom set and the retrieved corpus ledger, ending a contradiction in "
        "`ontology_violation` after one bounded semantic repair (needs --answer-format envelope, "
        "--ontology-axioms, and --ontology-ledger)",
    ),
    ontology_axioms: Optional[Path] = typer.Option(
        None,
        "--ontology-axioms",
        help="SIGNED axiom file (.ttl or its .json mirror) the answer gate enables; an unsigned "
        "file is refused rather than silently enabled",
    ),
    ontology_ledger: Optional[Path] = typer.Option(
        None,
        "--ontology-ledger",
        help="the corpus extraction.jsonl (or the draft bundle dir holding it) whose facts the "
        "answer is checked against, scoped per case to the retrieved chunks",
    ),
    ontology_overlay: Optional[Path] = typer.Option(
        None,
        "--ontology-overlay",
        help="optional node overlay from `resolve-graph-entities`; a declared endpoint folds "
        "through the identity that lane PROPOSED, so an answer naming an entity the graph merged "
        "is not read as a second value",
    ),
    vllm_suppress_thinking: Optional[bool] = typer.Option(
        None,
        "--vllm-suppress-thinking/--no-vllm-suppress-thinking",
        help="send vLLM's reasoning-output controls (enable_thinking=false, and the request "
        "fields a probe confirms this server accepts) so a reasoning model served by vLLM is "
        "scored with its thinking suppressed, the way the Ollama path already is",
    ),
    suppress_reasoning_prompt: Optional[bool] = typer.Option(
        None,
        "--suppress-reasoning-prompt/--no-suppress-reasoning-prompt",
        help="append a prompt-level no-reasoning instruction on top of the backend's native "
        "thinking-suppression flag, for a tag whose chat template leaks deliberation into the "
        "answer body (thinking-suppression-and-answer-language-guard)",
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
    cfg = load_config(config, **config_overrides(locals()))
    execute_eval(
        cfg,
        adapter=adapter,
        prompt_system=prompt_system,
        prompt_package=prompt_package,
        split=split,
        limit=limit,
        judge_rho=judge_rho,
        worksheet=worksheet,
        evict=evict,
        wait=wait,
        resume=resume,
        max_case_retries=max_case_retries,
        retry_backoff_s=retry_backoff_s,
    )
