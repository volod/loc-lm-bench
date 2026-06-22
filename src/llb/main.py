"""loc-lm-bench CLI (Typer).

Commands by milestone:
  build-index / validate-retrieval / run-eval        M1 skeleton (retrieve -> generate -> score)
  prep-models / list-models / build-vllm             M1/M2 model prep + feasibility + vLLM build
  detect-gpu-vram / gen-serving-config             per-GPU-tier serve + run-eval artifacts
  resolve-models                                     M3.2 pick the backend that can serve a model
  sweep                                              M3.3 isolated cell-per-model sweep (resume)
  tune                                               M3.4 two-stage Optuna (tuning -> final)
  prepare-goldset / prepare-synthetic-corpus         M3.5 frontier data-prep (litellm)
  prepare-goldset-draft                              M4.4 ontology-assisted draft (local/frontier)
  judge-experiment                                   M3.8 local DeepEval UA smoke artifact
  screen-public                                      M3.1 Tier-1 lm-eval-harness-uk screen
  board / mlflow-ui                                  M3.7 Streamlit leaderboard / MLflow UI

Heavy collaborators (FAISS, sentence-transformers, langgraph, optuna, litellm, streamlit, a
running backend) are lazy-imported at call time, so the module imports in the base install.
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


def _planning_models(manifest: Path) -> list[ModelSpec]:
    """Manifest models with missing arch fields filled from a cached config.json (M4.1).

    Offline + best-effort: it sharpens the embedding-aware VRAM estimate when weights are
    already cached, and is a no-op otherwise."""
    from llb.backends.planner import enrich_arch

    return [enrich_arch(m) for m in _load_models(manifest)]


def _best_effort_gpu_readers() -> tuple[Any, Any]:
    """Best-effort (vram_reader, pid_usage_reader) for the VRAM-reclaim + leak-attribution gate.
    Both are None when the [telemetry] extra / a GPU is absent (the gate then no-ops)."""
    try:
        from llb.executor.vram import nvml_process_reader, nvml_reader

        return nvml_reader(), nvml_process_reader()
    except (Exception, SystemExit):
        return None, None


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

    models = _planning_models(manifest)
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


@app.command("resolve-models")
def resolve_models_cmd(
    manifest: Path = typer.Option(
        Path("samples/models_uk.yaml"), help="candidate-models YAML manifest"
    ),
    context: Optional[int] = typer.Option(
        None, help="resolve fit at this target context instead of the max the host can hold"
    ),
    vram_reserve: int = typer.Option(1024, help="VRAM MiB held back for CUDA/display"),
    ram_reserve: int = typer.Option(2048, help="RAM MiB held back for the OS"),
    offline: bool = typer.Option(
        False, help="skip availability probes (assume every declared source exists)"
    ),
) -> None:
    """Pick the backend that can actually serve each model (discovery + vLLM>Ollama priority)."""
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.backends.resolver import ResolverProbes, format_resolution, resolve_all

    models = _planning_models(manifest)
    gpus = detect_gpus()
    vram_mib = max_vram_mb(gpus)
    ram_mib = detect_ram_mb()
    if gpus:
        for g in gpus:
            typer.echo(f"[resolve-models] GPU {g.index}: {g.name} ({g.total_mb} MiB)")
    else:
        typer.echo("[resolve-models] no GPU detected -- planning against system RAM only")

    probes = (
        ResolverProbes(hf_repo=lambda _s: True, gguf=lambda _s: True, ollama_tag=lambda _s: True)
        if offline
        else ResolverProbes()
    )
    rows = resolve_all(
        models,
        vram_mib,
        ram_mib,
        probes=probes,
        target_ctx=context,
        vram_reserve=vram_reserve,
        ram_reserve=ram_reserve,
    )
    typer.echo(format_resolution(rows))
    resolved = sum(1 for r in rows if r["chosen_backend"] is not None)
    typer.echo(f"[resolve-models] resolved {resolved} of {len(rows)} to a runnable backend")


@app.command("sweep")
def sweep_cmd(
    manifest: Path = typer.Option(
        Path("samples/models_uk.yaml"), help="candidate-models YAML manifest"
    ),
    split: str = typer.Option("final", help="gold split to evaluate each cell on"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL for every cell"),
    sweep_id: Optional[str] = typer.Option(None, help="resume id (default: a UTC timestamp)"),
    max_model_len: int = typer.Option(8192, help="vLLM context cap per cell (KV cache fit)"),
    telemetry: bool = typer.Option(True, help="measure steady-state throughput + peak VRAM"),
    resume: bool = typer.Option(True, help="skip cells already completed under this sweep id"),
    offline: bool = typer.Option(False, help="skip availability probes (assume sources exist)"),
) -> None:
    """Run one isolated cell per runnable model (process-per-cell, VRAM gate, thermal cooldown)."""
    from datetime import datetime, timezone

    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.backends.resolver import ResolverProbes, llamacpp_offload_split, resolve_all
    from llb.executor.isolation import run_sweep

    models = _load_models(manifest)
    gpus = detect_gpus()
    probes = (
        ResolverProbes(hf_repo=lambda _s: True, gguf=lambda _s: True, ollama_tag=lambda _s: True)
        if offline
        else ResolverProbes()
    )
    resolved = resolve_all(
        models, max_vram_mb(gpus), detect_ram_mb(), probes=probes, target_ctx=None
    )
    base = _load_config(None, goldset_path=goldset)
    cells: list[RunConfig] = []
    for r in resolved:
        if not r["chosen_backend"]:
            typer.echo(f"[sweep] skip {r['name']}: {r['note']}")
            continue
        overrides: dict[str, Any] = {
            "model": r["chosen_source"],
            "backend": r["chosen_backend"],
            "measure_telemetry": telemetry,
            "run_name": f"sweep-{r['name']}",
        }
        if r["chosen_backend"] == "vllm":
            overrides["max_model_len"] = max_model_len
        elif r["chosen_backend"] == "llamacpp":
            # Offload split from the planner so an oversized GGUF spills extra layers to CPU RAM
            # instead of the launcher default (-1 == every layer on GPU) OOMing the card.
            ngl = llamacpp_offload_split(r)
            if ngl is not None:
                overrides["n_gpu_layers"] = ngl
                typer.echo(f"[sweep] {r['name']}: llama.cpp offload split -ngl={ngl}")
        cells.append(base.with_overrides(**overrides))

    if not cells:
        typer.echo("[sweep] no runnable models resolved; nothing to do")
        raise typer.Exit(code=1)

    vram_reader, pid_reader = _best_effort_gpu_readers()

    sid = sweep_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    typer.echo(f"[sweep] id={sid} cells={len(cells)} split={split} resume={resume}")
    report = run_sweep(
        cells,
        sweep_id=sid,
        split=split,
        data_dir=base.data_dir,
        telemetry=telemetry,
        resume=resume,
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
    )
    for cell in report["results"]:
        hot = next((g["temp_c"] for g in cell["gpu"] if g["temp_c"] is not None), None)
        typer.echo(
            f"[sweep] {cell['status']:<7} {cell['backend']:<6} {cell['model']:<34} "
            f"residual={cell['vram_residual_mb']} cooldown={cell['cooldown_s']}s temp={hot}C"
        )
    typer.echo(
        f"[sweep] done id={sid}: {report['completed']} run, {report['skipped']} skipped, "
        f"{report['failed']} failed"
    )
    if report["failed"]:
        raise typer.Exit(code=1)


@app.command("tune")
def tune_cmd(
    model: str = typer.Option(..., help="model name (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm"),
    trials: int = typer.Option(30, min=1, help="stage-1 Optuna trials on the tuning split"),
    study: Optional[str] = typer.Option(None, help="study name (persistent SQLite; resumes)"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL"),
    max_model_len: Optional[int] = typer.Option(None, help="vLLM context cap"),
    manifest: Path = typer.Option(
        Path("samples/models_uk.yaml"), help="manifest to read the model's arch from (VRAM prune)"
    ),
    seed: int = typer.Option(13, help="Optuna sampler seed"),
    isolate: bool = typer.Option(
        False, help="run each trial under the executor VRAM-reclaim + thermal-cooldown gate"
    ),
) -> None:
    """Two-stage tune: search RAG params on the tuning split, score the winner on final."""
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.optimize.tuner import two_stage

    cfg = _load_config(
        None, model=model, backend=backend, goldset_path=goldset, max_model_len=max_model_len
    )
    spec = None
    for m in _load_models(manifest):
        if m["source"] == model or m.get("name") == model:
            spec = m
            break
    gpus = detect_gpus()
    vram_reader, pid_reader = _best_effort_gpu_readers() if isolate else (None, None)
    study_name = study or f"tune-{model.replace('/', '_').replace(':', '_')}"
    typer.echo(f"[tune] study={study_name} model={model} backend={backend} trials={trials}")
    out = two_stage(
        cfg,
        n_trials=trials,
        study_name=study_name,
        model_spec=spec,
        vram_mib=max_vram_mb(gpus),
        ram_mib=detect_ram_mb(),
        seed=seed,
        isolate=isolate,
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
    )
    t = out.tune
    typer.echo(
        f"[tune] stage-1 best quality={t.best_value:.4f} "
        f"({t.n_complete} complete, {t.n_pruned} pruned of {t.n_trials})"
    )
    typer.echo(
        f"[tune] winning config: strategy={t.best_config.strategy} "
        f"size={t.best_config.chunk_size} overlap={t.best_config.chunk_overlap} "
        f"top_k={t.best_config.top_k} mode={t.best_config.retrieval_mode}"
    )
    typer.echo("[tune] stage-2 (final split) is the leaderboard entry:")
    typer.echo(out.final["table"])


@app.command("prepare-goldset")
def prepare_goldset_cmd(
    corpus_root: Path = typer.Option(..., help="directory of .md/.txt source docs"),
    model: str = typer.Option(..., help="litellm model id (needs a provider key in .env)"),
    n_per_doc: int = typer.Option(3, min=1, help="draft this many QA pairs per document"),
    out: Path = typer.Option(..., help="output gold set JSONL (items are verified=false)"),
) -> None:
    """Draft review-ready (question, answer, exact span) gold items from a corpus via a frontier LLM."""
    from llb.prep.frontier import prepare_goldset

    items = prepare_goldset(corpus_root, model=model, n_per_doc=n_per_doc, out_path=out)
    typer.echo(
        f"[prepare-goldset] {len(items)} drafted items (verified=false; review before scoring) -> {out}"
    )


@app.command("prepare-synthetic-corpus")
def prepare_synthetic_corpus_cmd(
    topics_file: Path = typer.Option(..., help="text file: one synthetic-doc topic per line"),
    planter: str = typer.Option(..., help="litellm model that PLANTS the labels"),
    judge: str = typer.Option(..., help="the eval judge model (must differ from the planter)"),
    out_dir: Path = typer.Option(..., help="output dir for docs + planted_labels.jsonl"),
    n_labels: int = typer.Option(3, min=1, help="planted QA pairs per document"),
) -> None:
    """Generate synthetic docs with structured planted labels (planter must differ from judge)."""
    from llb.prep.frontier import prepare_synthetic_corpus

    topics = [t.strip() for t in topics_file.read_text(encoding="utf-8").splitlines() if t.strip()]
    if not topics:
        typer.echo(f"[error] no topics found in {topics_file}", err=True)
        raise typer.Exit(code=2)
    docs, items = prepare_synthetic_corpus(
        topics, planter_model=planter, judge_model=judge, n_labels=n_labels, out_dir=out_dir
    )
    typer.echo(
        f"[prepare-synthetic-corpus] {len(docs)} docs, {len(items)} planted items "
        f"(planter={planter} != judge={judge}) -> {out_dir}"
    )


@app.command("prepare-goldset-draft")
def prepare_goldset_draft_cmd(
    corpus_root: Path = typer.Option(..., help="directory of .md/.txt source docs"),
    model: str = typer.Option(
        ..., help="model id (local endpoint tag, or litellm route for frontier)"
    ),
    endpoint: str = typer.Option(
        "local", help="local (OpenAI-compatible, no egress) | frontier (litellm, opt-in egress)"
    ),
    base_url: Optional[str] = typer.Option(
        None, help="local endpoint base URL (default: Ollama OpenAI-compatible /v1)"
    ),
    max_items: int = typer.Option(60, min=1, help="upper bound on drafted QA items"),
    seed: int = typer.Option(13, help="deterministic sampling/split seed"),
    out_dir: Optional[Path] = typer.Option(
        None, help="output bundle dir (default: $DATA_DIR/prepare-goldset/<timestamp>/)"
    ),
) -> None:
    """M4.4: ontology-assisted DRAFT gold set from a corpus (verified=false; review before scoring)."""
    from llb.prep.ontology import EndpointConfig, draft_goldset

    try:
        cfg = (
            EndpointConfig(kind=endpoint, model=model, base_url=base_url)
            if base_url
            else EndpointConfig(kind=endpoint, model=model)
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)
    result = draft_goldset(corpus_root, cfg, max_items=max_items, seed=seed, out_dir=out_dir)
    typer.echo(
        f"[prepare-goldset-draft] {len(result.items)} drafted items (verified=false; "
        f"endpoint={endpoint}, egress={cfg.egress}) -> {result.out_dir}"
    )


@app.command("screen-public")
def screen_public_cmd(
    model: str = typer.Option(..., help="model name (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama (generation track) | vllm (logprob track)"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint (skips launching)"
    ),
    tasks: Optional[str] = typer.Option(None, help="extra lm-eval task ids (comma-separated)"),
    limit: Optional[int] = typer.Option(None, help="cap examples per task (smoke runs)"),
    out_dir: Optional[Path] = typer.Option(None, help="output dir for lm-eval results JSON"),
    max_model_len: int = typer.Option(
        8192, help="vLLM context cap (the native window OOMs the KV cache on 16 GB)"
    ),
    isolated: bool = typer.Option(
        False, help="run under the Tier-2 VRAM-reclaim + thermal-cooldown isolation contract"
    ),
) -> None:
    """Tier-1 public screen via lm-eval-harness-uk (logprob vs generation track; never mixed)."""
    import json

    from llb.screen.public import ScreenReport, run_screen, run_screen_isolated

    cfg = _load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    extra = [t.strip() for t in (tasks or "").split(",") if t.strip()]
    out = out_dir or (cfg.data_dir / "screen")

    def do_screen(url: str) -> ScreenReport:
        return run_screen(model, backend, url, extra_tasks=extra, output_dir=out, limit=limit)

    def screen_fn() -> ScreenReport:
        """Launch the backend (or use the running endpoint), run the screen, kill the backend."""
        if base_url:
            return do_screen(base_url)
        if backend == "ollama":
            return do_screen(f"{cfg.ollama_host.rstrip('/')}/v1")
        if backend == "vllm":
            from llb.backends.vllm import VllmLauncher

            launcher = VllmLauncher(
                model,
                host=cfg.vllm_host,
                port=cfg.vllm_port,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
                max_model_len=cfg.max_model_len,
            )
            with launcher:
                return do_screen(f"{cfg.vllm_host.rstrip('/')}/v1")
        typer.echo(f"[error] backend '{backend}' not supported for the screen", err=True)
        raise typer.Exit(code=2)

    if isolated:
        vram_reader, pid_reader = _best_effort_gpu_readers()
        report, iso = run_screen_isolated(
            backend, screen_fn, vram_reader=vram_reader, pid_usage_reader=pid_reader
        )
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{model.replace('/', '_').replace(':', '_')}.isolation.json").write_text(
            json.dumps(iso), encoding="utf-8"
        )
        typer.echo(
            f"[screen-public] isolation: vram_residual={iso['vram_residual_mb']} "
            f"cooldown={iso['cooldown']['waited_s']}s capped={iso['cooldown']['capped']}"
        )
    else:
        report = screen_fn()

    cov = f"{len(report['covered'])}/{len(report['requested_tasks'])}"
    status = "complete" if report["complete"] else f"PARTIAL (missing {report['missing']})"
    typer.echo(f"[screen-public] {model} track={report['track']} coverage={cov} -- {status}")
    for r in report["results"]:
        typer.echo(f"[screen-public]   {r['task']:<22} {r['metric']}={r['score']:.3f}")


@app.command("board")
def board_cmd(
    host: str = typer.Option("127.0.0.1", help="network interface for the Streamlit board"),
    port: int = typer.Option(8501, min=1, max=65535, help="port for the Streamlit board"),
) -> None:
    """Serve the thin Streamlit leaderboard (rank + best-config-per-model + CIs)."""
    import subprocess
    import sys

    try:
        import streamlit  # noqa: F401
    except ImportError:
        typer.echo(
            '[error] the board needs the [board] extra. uv pip install -e ".[board]"', err=True
        )
        raise typer.Exit(code=2) from None
    from llb.board import app as board_app

    app_path = Path(board_app.__file__)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    raise typer.Exit(subprocess.call(cmd))


@app.command("pipeline")
def pipeline_cmd(
    manifest: Path = typer.Option(
        Path("samples/models_uk.yaml"), help="candidate-models YAML manifest"
    ),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL for the Tier-2 tuning"),
    top_n: int = typer.Option(2, min=1, help="finalists to keep per screen track"),
    trials: int = typer.Option(20, min=1, help="stage-1 Optuna trials per finalist"),
    offline: bool = typer.Option(False, help="resolver: assume declared sources exist"),
) -> None:
    """Tier handoff: screen reports -> per-track finalists -> tuned private eval -> final board.

    Run `screen-public` per candidate first to produce the Tier-1 reports; this command then
    selects finalists, runs the two-stage tune for each, and prints the final-only board.
    """
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.backends.resolver import ResolverProbes, resolve_all
    from llb.board.data import best_per_model, load_run_records
    from llb.optimize.tuner import two_stage
    from llb.scoring.aggregate import format_board, rank_board, ranking_policy_note
    from llb.screen.public import select_finalists

    from llb.board.data import load_screen_reports

    cfg = _load_config(None, goldset_path=goldset)
    reports = load_screen_reports(cfg.data_dir / "screen")
    if not reports:
        typer.echo(
            "[pipeline] no screen reports found; run `screen-public` per candidate first", err=True
        )
        raise typer.Exit(code=2)
    finalists = set(select_finalists(reports, top_n))
    typer.echo(f"[pipeline] finalists (top {top_n}/track): {sorted(finalists)}")

    gpus = detect_gpus()
    probes = (
        ResolverProbes(hf_repo=lambda _s: True, gguf=lambda _s: True, ollama_tag=lambda _s: True)
        if offline
        else ResolverProbes()
    )
    resolved = {
        r["name"]: r
        for r in resolve_all(
            _load_models(manifest), max_vram_mb(gpus), detect_ram_mb(), probes=probes
        )
    }
    for name in sorted(finalists):
        info = resolved.get(name)
        if not info or not info["chosen_backend"]:
            typer.echo(f"[pipeline] skip {name}: not resolvable on this host")
            continue
        base = cfg.with_overrides(model=info["chosen_source"], backend=info["chosen_backend"])
        typer.echo(f"[pipeline] tuning finalist {name} ({info['chosen_backend']})")
        two_stage(base, n_trials=trials, study_name=f"pipeline-{name.replace('/', '_')}")

    records = best_per_model(load_run_records(cfg.data_dir / "run-eval"))
    if records:
        results = [r.result for r in records]
        typer.echo("[pipeline] final-only board:")
        typer.echo(format_board(rank_board(results), policy=ranking_policy_note(results, False)))


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
    backend: Optional[str] = typer.Option(None, help="ollama | vllm | llamacpp"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    max_model_len: Optional[int] = typer.Option(
        None, help="vLLM/llama.cpp served context window (overrides the config; no YAML needed)"
    ),
    gpu_memory_utilization: Optional[float] = typer.Option(
        None, help="vLLM GPU memory fraction 0-1 (overrides the config; no YAML needed)"
    ),
    split: str = typer.Option("final", help="gold split to evaluate"),
    limit: Optional[int] = typer.Option(None, help="cap the number of eval items"),
    judge_rho: Optional[float] = typer.Option(
        None, help="calibration Spearman rho; judge stays demoted below the threshold"
    ),
    judge_model: Optional[str] = typer.Option(
        None, help="local judge model id; enables the DeepEval judge (gated by --judge-rho)"
    ),
    judge_base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible judge endpoint, e.g. http://localhost:8000/v1"
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
    evict: bool = typer.Option(
        False, help="vLLM contention guard: unload Ollama's resident models before launching"
    ),
    wait: bool = typer.Option(
        False, help="vLLM contention guard: wait for VRAM to free instead of derating immediately"
    ),
) -> None:
    """Run the skeleton on one model and print a ranked row + write the manifest."""
    from llb.executor.runner import run_eval

    cfg = _load_config(
        config,
        model=model,
        backend=backend,
        goldset_path=goldset,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        score_semantic=score_semantic,
        measure_telemetry=telemetry,
    )
    run_eval(
        cfg,
        split=split,
        limit=limit,
        judge_rho=judge_rho,
        worksheet=worksheet,
        evict=evict,
        wait=wait,
    )


@app.command("judge-experiment")
def judge_experiment_cmd(
    judge_model: str = typer.Option(..., help="served local judge model id"),
    judge_base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1"
    ),
    data_dir: Optional[Path] = typer.Option(None, help="artifact root (default: DATA_DIR)"),
) -> None:
    """Run fixed Ukrainian judge sanity cases and record prompts plus scores."""
    from llb.judge.experiment import run_judge_experiment

    report, out_path = run_judge_experiment(
        judge_model,
        base_url=judge_base_url,
        data_dir=data_dir,
    )
    typer.echo(
        f"[judge-experiment] model={report['judge']['model']} "
        f"cases={len(report['cases'])} -> {out_path}"
    )


@app.command("detect-gpu-vram")
def detect_gpu_vram_cmd() -> None:
    """Print the supported GPU VRAM tier (12/16/24/32 GiB) for this host."""
    from llb.inference.generate import detect_gpu_tier, format_detect_line

    typer.echo(format_detect_line(detect_gpu_tier()))


@app.command("gen-serving-config")
def gen_serving_config_cmd(
    gpu_gb: Optional[int] = typer.Option(
        None, help="GPU VRAM tier in GiB (12, 16, 24, 32); default: detect from nvidia-smi"
    ),
    manifest: Path = typer.Option(
        Path("samples/config-example/manifest.yaml"),
        help="tier manifest with model + vLLM knobs",
    ),
    output: Optional[Path] = typer.Option(
        None, help="output directory (default: .data/llb/serving/gpu-<tier>gb/)"
    ),
) -> None:
    """Generate serve scripts and run-eval YAML for the largest models on this GPU tier."""
    from llb.inference.generate import generate_serving_configs, resolve_tier
    from llb.paths import PROJECT_ROOT

    manifest_path = manifest.resolve()
    out = generate_serving_configs(
        gpu_gb=gpu_gb,
        output_root=output.resolve() if output else None,
        manifest_path=manifest_path,
    )
    info = resolve_tier(gpu_gb)
    rel = out.resolve().relative_to(PROJECT_ROOT.resolve())
    typer.echo(f"[gen-serving-config] tier={info.tier_gb} GiB gpu_mb={info.total_mb} -> {rel}/")
    typer.echo(f"[gen-serving-config] see {rel / 'tier.json'} for serve/run script names")


def main() -> None:
    from llb.runtime import run_typer

    run_typer(app)


if __name__ == "__main__":
    main()
