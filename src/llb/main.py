"""loc-lm-bench CLI (Typer).

Milestone 1 commands wire the compile-free skeleton (prebuilt Ollama on the GPU; no
vLLM/flash-attn source build):
  build-index         chunk + embed the corpus into a FAISS RAG store
  validate-retrieval  recall@k / MRR of the pinned embedding over the gold set (Premise 4)
  run-eval            retrieve -> generate -> score -> ranked row + manifest

Heavy collaborators (FAISS, sentence-transformers, langgraph, a running Ollama) are only
touched by these commands at call time, so the module imports in the base install.
Config comes from a YAML file (`--config`) with CLI flags overriding individual fields.
"""

from pathlib import Path
from typing import Any, Optional

import typer

from llb.config import RunConfig
from llb.contracts import ModelSpec

app = typer.Typer(
    add_completion=False,
    rich_markup_mode=None,
    help="loc-lm-bench: local Ukrainian LLM benchmark.",
)


def _load_config(config_path: Optional[Path], **overrides: Any) -> RunConfig:
    try:
        base = RunConfig.load(config_path) if config_path else RunConfig()
        return base.with_overrides(**overrides)
    except ValueError as exc:
        typer.echo(f"[error] invalid run config: {exc}", err=True)
        raise typer.Exit(code=2) from None


def _load_models(manifest: Path) -> list[ModelSpec]:
    """Load a models manifest, reporting a YAML/schema error as a clean one-liner."""
    from llb.backends.prepare import load_manifest

    try:
        return load_manifest(manifest)
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None


@app.command("build-index")
def build_index(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    corpus_root: Optional[Path] = typer.Option(None, help="corpus directory to chunk"),
    strategy: Optional[str] = typer.Option(
        None, help="fixed | sentence | recursive | markdown | semantic"
    ),
    size: Optional[int] = typer.Option(None, help="chunk size (chars)"),
    overlap: Optional[int] = typer.Option(None, help="chunk overlap (chars)"),
    embedding_model: Optional[str] = typer.Option(None, help="pinned embedding model"),
    mode: Optional[str] = typer.Option(None, help="flat | parent_child"),
    child_size: Optional[int] = typer.Option(None, help="child chunk size (parent_child mode)"),
) -> None:
    """Chunk + embed the corpus into a FAISS RAG store under the index dir."""
    cfg = _load_config(
        config,
        corpus_root=corpus_root,
        strategy=strategy,
        chunk_size=size,
        chunk_overlap=overlap,
        embedding_model=embedding_model,
        retrieval_mode=mode,
        child_chunk_size=child_size,
    )
    from llb.rag.store import RagStore

    store = RagStore.build(
        cfg.corpus_root,
        cfg.strategy,
        cfg.chunk_size,
        cfg.chunk_overlap,
        cfg.embedding_model,
        mode=cfg.retrieval_mode,
        child_size=cfg.child_chunk_size,
    )
    store.save(cfg.index_dir())
    parents = f", {store.meta['n_parents']} parents" if store.meta["n_parents"] else ""
    typer.echo(
        f"[build-index] {store.meta['n_indexed']} indexed chunks{parents} "
        f"({cfg.strategy}/{cfg.retrieval_mode}, dim {store.meta['dim']}) -> {cfg.index_dir()}"
    )


@app.command("validate-retrieval")
def validate_retrieval(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    k: int = typer.Option(10, help="recall@k cutoff (Premise 4 gate is recall@10 >= 0.8)"),
    split: Optional[str] = typer.Option(None, help="restrict to one gold split"),
) -> None:
    """Score the pinned embedding's retrieval over the gold set (does not rank models)."""
    from llb.goldset.schema import load_goldset
    from llb.executor.cases import spans_as_dicts
    from llb.rag import retrieval
    from llb.rag.store import RagStore

    cfg = _load_config(config, goldset_path=goldset)
    store = RagStore.load(cfg.index_dir())
    items = load_goldset(cfg.goldset_path)
    if split:
        items = [it for it in items if it.split == split]
    pairs = [(store.retrieve(it.question, k), spans_as_dicts(it)) for it in items]
    report = retrieval.evaluate_retrieval(pairs, k)
    gate = "PASS" if report["recall_at_k"] >= 0.8 else "BELOW 0.8 (retrieval is the bottleneck)"
    typer.echo(
        f"[validate-retrieval] n={report['n']} recall@{k}={report['recall_at_k']:.3f} "
        f"mrr={report['mrr']:.3f} -> {gate}"
    )


@app.command("prep-models")
def prep_models_cmd(
    manifest: Path = typer.Option(
        Path("samples/models_uk.yaml"), help="candidate-models YAML manifest"
    ),
    backend: str = typer.Option("all", help="ollama | vllm | all"),
    force: bool = typer.Option(False, help="prepare even if a model looks too big for VRAM"),
    dry_run: bool = typer.Option(False, help="show the plan; pull/cache nothing"),
    cache_dir: Optional[Path] = typer.Option(None, help="HF cache dir for vLLM weights"),
) -> None:
    """Detect the GPU, pull Ollama tags, and cache vLLM (HF) weights once."""
    from llb.backends.prepare import prepare_models

    models = _load_models(manifest)
    report = prepare_models(
        models,
        backend_filter=backend,
        force=force,
        dry_run=dry_run,
        cache_dir=cache_dir,
    )

    if report["gpus"]:
        for g in report["gpus"]:
            typer.echo(
                f"[prep-models] GPU {g.index}: {g.name} "
                f"({g.total_mb} MB total, {g.free_mb} MB free, driver {g.driver})"
            )
    else:
        typer.echo("[prep-models] no GPU detected (Ollama runs on CPU; vLLM is skipped)")

    for r in report["results"]:
        typer.echo(
            f"[prep-models] {r['status']:<8} {r['backend']:<6} {r['name']:<22} "
            f"{r['source']}  -- {r['detail']}"
        )
    failed = [r for r in report["results"] if r["status"] == "failed"]
    if failed:
        raise typer.Exit(code=1)


@app.command("list-models")
def list_models_cmd(
    manifest: Path = typer.Option(
        Path("samples/models_uk.yaml"), help="candidate-models YAML manifest"
    ),
    context: Optional[int] = typer.Option(
        None, help="plan at this target context instead of the max the host can hold"
    ),
    vram_reserve: int = typer.Option(1024, help="VRAM MiB held back for CUDA/display"),
    ram_reserve: int = typer.Option(2048, help="RAM MiB held back for the OS"),
    runnable_only: bool = typer.Option(False, help="hide models that cannot run at all"),
) -> None:
    """List which candidate models can run here (GPU+RAM, KV-cache-aware, batch=1)."""
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.backends.planner import VERDICT_NO, format_plan, plan_models

    models = _load_models(manifest)
    gpus = detect_gpus()
    vram_mib = max_vram_mb(gpus)
    ram_mib = detect_ram_mb()

    if gpus:
        for g in gpus:
            typer.echo(f"[list-models] GPU {g.index}: {g.name} ({g.total_mb} MiB)")
    else:
        typer.echo("[list-models] no GPU detected -- planning against system RAM only")
    typer.echo(f"[list-models] system RAM: {ram_mib} MiB")

    rows = plan_models(
        models,
        vram_mib,
        ram_mib,
        target_ctx=context,
        vram_reserve=vram_reserve,
        ram_reserve=ram_reserve,
    )
    if runnable_only:
        rows = [r for r in rows if r["verdict"] != VERDICT_NO]
    typer.echo(format_plan(rows, max(0, vram_mib - vram_reserve), max(0, ram_mib - ram_reserve)))
    runnable = sum(1 for r in rows if r["verdict"] in ("gpu", "offload"))
    typer.echo(f"[list-models] runnable here: {runnable} of {len(rows)}")
    typer.echo(
        "[list-models] verdict is at ctx_max; ctx_gpu = max context that fits fully "
        "on GPU. gpu = no offload needed; offload = split layers GPU/CPU RAM."
    )


@app.command("mlflow-ui")
def mlflow_ui_cmd(
    host: str = typer.Option("127.0.0.1", help="network interface for the local MLflow UI"),
    port: int = typer.Option(5000, min=1, max=65535, help="port for the local MLflow UI"),
) -> None:
    """Serve the shared local MLflow experiment store."""
    from llb.tracking.server import run_mlflow_ui

    exit_code = run_mlflow_ui(host, port)
    if exit_code:
        raise typer.Exit(exit_code)


@app.command("run-eval")
def run_eval_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    model: Optional[str] = typer.Option(None, help="model name (Ollama tag or HF repo id)"),
    backend: Optional[str] = typer.Option(None, help="ollama | vllm"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    split: str = typer.Option("final", help="gold split to evaluate"),
    limit: Optional[int] = typer.Option(None, help="cap the number of eval items"),
    judge_rho: Optional[float] = typer.Option(
        None, help="calibration Spearman rho; judge stays demoted below the threshold"
    ),
    score_semantic: Optional[bool] = typer.Option(
        None,
        "--score-semantic/--no-score-semantic",
        help="enable or disable the embedding-similarity correctness signal",
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
) -> None:
    """Run the skeleton on one model and print a ranked row + write the manifest."""
    from llb.executor.runner import run_eval

    cfg = _load_config(
        config,
        model=model,
        backend=backend,
        goldset_path=goldset,
        score_semantic=score_semantic,
        measure_telemetry=telemetry,
    )
    run_eval(cfg, split=split, limit=limit, judge_rho=judge_rho, worksheet=worksheet)


def main() -> None:
    from llb.runtime import run_typer

    run_typer(app)


if __name__ == "__main__":
    main()
