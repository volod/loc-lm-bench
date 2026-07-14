"""Ontology-assisted gold-set draft command (local/frontier drafters; resumable)."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.prep.ontology.constants import DEFAULT_MULTI_HOP_MAX_PATHS, EXTRACT_CONCURRENCY


@app.command("prepare-goldset-draft")
def prepare_goldset_draft_cmd(
    corpus_root: Optional[Path] = typer.Option(
        None, help="directory of .md/.txt source docs (read from the bundle meta with --resume)"
    ),
    model: Optional[str] = typer.Option(
        None, help="model id (local endpoint tag, or litellm route for frontier)"
    ),
    resume: Optional[Path] = typer.Option(
        None,
        help="resume an interrupted draft bundle: reuse journaled extraction windows and replay "
        "the deterministic seed/draft stages (reads settings from the bundle's journal meta)",
    ),
    endpoint: str = typer.Option(
        "local", help="local (OpenAI-compatible, no egress) | frontier (litellm, opt-in egress)"
    ),
    egress_consent: bool = typer.Option(
        False,
        "--egress-consent",
        help="parent workflow already collected corpus-and-destination-specific consent",
    ),
    frontier_stage: str = typer.Option(
        "both",
        help="frontier routing when --endpoint frontier: extraction | drafting | both",
    ),
    local_model: Optional[str] = typer.Option(
        None, help="local model for the non-frontier phase when --frontier-stage is mixed"
    ),
    backend: str = typer.Option(
        "ollama",
        help="local serving backend for --endpoint local: ollama | vllm | openai",
    ),
    base_url: Optional[str] = typer.Option(
        None, help="local endpoint base URL (default: Ollama OpenAI-compatible /v1)"
    ),
    max_items: int = typer.Option(60, min=1, help="upper bound on drafted QA items"),
    doc_limit: Optional[int] = typer.Option(
        None, min=1, help="bounded probe: only process the first N corpus documents"
    ),
    seed: int = typer.Option(13, help="deterministic sampling/split seed"),
    extractor: str = typer.Option(
        "llm", help="llm (default) | spacy (opt-in Python-native uk_core_news NER, no egress)"
    ),
    spacy_model: str = typer.Option(
        "uk_core_news_sm", help="spaCy pipeline (with --extractor spacy)"
    ),
    max_tokens: int = typer.Option(
        4096, min=1, help="per-call completion token budget for ontology drafting"
    ),
    extract_max_chars: Optional[int] = typer.Option(
        None,
        min=1,
        help="bounded probe/window size: max document characters per extraction call",
    ),
    extract_chunk_overlap: Optional[int] = typer.Option(
        None, min=0, help="overlap between extraction windows when a document is split"
    ),
    concurrency: int = typer.Option(
        EXTRACT_CONCURRENCY,
        "--concurrency",
        "--extract-concurrency",
        min=1,
        help="LLM extraction windows to run concurrently per document; merge order stays deterministic",
    ),
    temperature: float = typer.Option(
        0.0, min=0.0, help="per-call generation temperature for ontology drafting"
    ),
    timeout: float = typer.Option(
        300.0, min=1.0, help="per-call local/frontier endpoint timeout in seconds"
    ),
    max_usd: Optional[float] = typer.Option(
        None, min=0.000001, help="hard measured-spend guard for all frontier calls in this run"
    ),
    max_calls: Optional[int] = typer.Option(
        None, min=1, help="hard cap on frontier calls across extraction and drafting (default: 100)"
    ),
    no_think: bool = typer.Option(
        False,
        "--no-think",
        help="disable hidden reasoning for local JSON-producing models (Ollama native or vLLM extra_body)",
    ),
    num_ctx: Optional[int] = typer.Option(
        None,
        min=1,
        help="right-size the Ollama context window (native endpoint); avoids CPU offload from "
        "the modelfile default on VRAM-bound hosts -- keep headroom over extract-max-chars",
    ),
    vllm_port: int = typer.Option(
        8000,
        min=1,
        max=65535,
        help="port for a vLLM server launched by this command when --backend vllm and --base-url is unset",
    ),
    vllm_gpu_memory_utilization: float = typer.Option(
        0.85,
        min=0.01,
        max=1.0,
        help="vLLM --gpu-memory-utilization when this command launches the server",
    ),
    vllm_max_model_len: Optional[int] = typer.Option(
        None,
        min=1,
        help="vLLM --max-model-len when this command launches the server; defaults to --num-ctx when set",
    ),
    vllm_cpu_offload_gb: Optional[float] = typer.Option(
        None,
        min=0.0,
        help="vLLM --cpu-offload-gb when this command launches the server",
    ),
    vllm_kv_offloading_size_gb: Optional[float] = typer.Option(
        None,
        min=0.0,
        help="vLLM --kv-offloading-size when this command launches the server",
    ),
    vllm_dtype: str = typer.Option(
        "auto", help="vLLM --dtype when this command launches the server"
    ),
    vllm_quantization: Optional[str] = typer.Option(
        None, help="vLLM --quantization when this command launches the server"
    ),
    vllm_startup_timeout: float = typer.Option(
        600.0,
        min=1.0,
        help="seconds to wait for a vLLM server launched by this command to become ready",
    ),
    out_dir: Optional[Path] = typer.Option(
        None, help="output bundle dir (default: $DATA_DIR/prepare-goldset/<timestamp>/)"
    ),
    verification_sample_size: int = typer.Option(
        0,
        min=0,
        help="also write verify_sample.csv for human review (0 leaves review to make verify-sample)",
    ),
    retrieval_index_dir: Optional[Path] = typer.Option(
        None,
        help="full-corpus RAG index dir; when set, annotate citation-valid needles with retrieval_rank",
    ),
    retrieval_k: int = typer.Option(
        10, min=1, help="top-k cutoff for --retrieval-index-dir needle-rank annotation"
    ),
    drop_nonretrievable_needles: bool = typer.Option(
        False,
        "--drop-nonretrievable-needles",
        help="write only needles whose gold span is found within --retrieval-k",
    ),
    coverage_target: Optional[int] = typer.Option(
        None,
        min=1,
        help="yield-max: draft up to N seeds per stratum bucket instead of the flat --max-items cap",
    ),
    multi_hop: bool = typer.Option(
        False,
        "--multi-hop",
        help="yield-max: also draft multi-span chain questions walked from the knowledge graph",
    ),
    chains: bool = typer.Option(
        False,
        "--chains",
        help="also write chains.jsonl with ordered chain-of-questions items from graph paths",
    ),
    multi_hop_max_paths: int = typer.Option(
        DEFAULT_MULTI_HOP_MAX_PATHS,
        min=1,
        help="cap on 2-hop graph paths drafted when --multi-hop is set",
    ),
    dedup_against: Optional[str] = typer.Option(
        None,
        help="yield-max: comma-separated prior bundle dirs; drop pinned-E5 near-duplicate questions",
    ),
    graph_dir: Optional[Path] = typer.Option(
        None,
        help="persisted graph store dir for --multi-hop paths (default: build the graph in-run)",
    ),
    rejection_feedback: Optional[Path] = typer.Option(
        None,
        "--rejection-feedback",
        help="verify-gate rejection_reasons.json; dominant reject codes tighten the draft "
        "prompts and the applied hints land in provenance",
    ),
    require_passed_gates: bool = typer.Option(
        False,
        "--require-passed-gates",
        help="exit non-zero after writing the bundle when the ontology calibration gates fail",
    ),
) -> None:
    """ontology-assisted drafting: ontology-assisted DRAFT gold set from a corpus (verified=false; review before scoring)."""
    from llb.cli.prep.draft_execution import run_draft
    from llb.cli.prep.draft_request import DraftRequest

    run_draft(DraftRequest.from_mapping(locals()))
