"""Eval, screen, pipeline, and judge experiment commands."""

import json
from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import (
    best_effort_gpu_readers,
    load_config,
    load_models,
    resolver_probes,
)
from llb.core.config import RunConfig
from llb.screen.public import ScreenReport


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
        None, help="local judge model id; enables the DeepEval judge (gated by --judge-rho)"
    ),
    judge_base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible judge endpoint, e.g. http://localhost:8000/v1"
    ),
    retrieval_backend: Optional[str] = typer.Option(
        None, help="faiss (default vector store) | graph (GraphRAG knowledge-graph backend)"
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
        "normalize,typos,glossary,rewrite (rewrite calls the local model; off by default). "
        "The raw query is always preserved; only the retrieval query is transformed",
    ),
    query_glossary: Optional[Path] = typer.Option(
        None,
        help="query_glossary.json for the query-prep 'glossary' step (build-query-glossary)",
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
        retrieval_backend=retrieval_backend,
        retrieval_strategy=retrieval_strategy,
        retrieval_mode=retrieval_mode,
        acl_label=acl,
        fusion_weight=fusion_weight,
        fusion_candidates=fusion_candidates,
        reranker=reranker,
        rerank_candidates=rerank_candidates,
        context_order=context_order,
        query_prep=_parse_query_prep(query_prep),
        query_glossary_path=query_glossary,
        score_semantic=score_semantic,
        cited_answers=cited_answers,
        score_groundedness=score_groundedness,
        insufficient_context_probes=insufficient_context_probes,
        measure_telemetry=telemetry,
    )
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


@app.command("probe-context-position")
def probe_context_position_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    model: Optional[str] = typer.Option(None, help="model name (Ollama tag or HF repo id)"),
    backend: Optional[str] = typer.Option(None, help="ollama | vllm | llamacpp"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL (overrides the config)"),
    k: int = typer.Option(5, min=3, help="fixed context size (gold at head/middle/tail)"),
    split: str = typer.Option("final", help="gold split to probe"),
    limit: Optional[int] = typer.Option(None, help="cap the number of probed items"),
    candidate_depth: Optional[int] = typer.Option(
        None, help="retrieval depth scanned for the gold chunk + distractors (default 50)"
    ),
    max_model_len: Optional[int] = typer.Option(
        None, help="vLLM/llama.cpp served context window (overrides the config)"
    ),
    out_dir: Optional[Path] = typer.Option(
        None, help="probe output dir (default: $DATA_DIR/context-position/<timestamp>)"
    ),
) -> None:
    """Lost-in-the-middle probe (rerank-context-order): place each item's gold chunk at the
    head, middle, and tail of a fixed-k context of real retrieved distractors, score every
    position, and recommend a per-model `context_order` with bootstrap CIs."""
    from llb.bench.common import new_run_timestamp
    from llb.core.contracts import ChatMessage
    from llb.eval.position_probe import (
        DEFAULT_CANDIDATE_DEPTH,
        render_report,
        run_probe,
        write_probe,
    )
    from llb.executor.runner import _load_store, _make_launcher
    from llb.goldset.schema import load_goldset

    cfg = load_config(
        config, model=model, backend=backend, goldset_path=goldset, max_model_len=max_model_len
    )
    items = [it for it in load_goldset(cfg.goldset_path) if it.verified and it.split == split]
    items.sort(key=lambda it: it.id)
    if limit is not None:
        items = items[:limit]
    if not items:
        typer.echo(f"[probe-context-position] no verified '{split}' items to probe", err=True)
        raise typer.Exit(code=2)
    store = _load_store(cfg)
    launcher = _make_launcher(cfg)
    with launcher as served:

        def chat(messages: list[ChatMessage]) -> tuple[str, str | None]:
            result = served.chat(
                messages,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                timeout=cfg.request_timeout_s,
            )
            return result.text or "", result.error

        report = run_probe(
            items,
            store,
            chat,
            model=cfg.model,
            backend=cfg.backend,
            k=k,
            candidate_depth=candidate_depth or DEFAULT_CANDIDATE_DEPTH,
        )
    _, run_ts = new_run_timestamp()
    out = out_dir or (cfg.data_dir / "context-position" / run_ts)
    paths = write_probe(report, out)
    typer.echo(render_report(report))
    typer.echo(f"[probe-context-position] report -> {paths['report']}")
    typer.echo(f"[probe-context-position] cases -> {paths['cases']}")


@app.command("analyze-misses")
def analyze_misses_cmd(
    run_dir: Path = typer.Option(
        ..., "--run-dir", help="finalized run-eval bundle whose misses to explain"
    ),
    goldset: Optional[Path] = typer.Option(
        None, help="goldset JSONL the run scored (default: the bundle manifest's goldset_path)"
    ),
    miss_threshold: Optional[float] = typer.Option(
        None,
        "--miss-threshold",
        help="objective score below which a scoreable (status=ok) case counts as a miss "
        "(default 0.5)",
    ),
    probe_top_k: Optional[str] = typer.Option(
        None,
        "--probe-top-k",
        help="comma-separated retrieval depths (e.g. 3,8): re-run ONLY the miss subset at each "
        "depth to confirm or reject the retrieval hypothesis (launches the model backend)",
    ),
    out_dir: Optional[Path] = typer.Option(
        None, help="analysis output dir (default: $DATA_DIR/miss-analysis/<timestamp>)"
    ),
) -> None:
    """Explain one run's wrong answers: classify every miss (retrieval / generation / refusal /
    format artifact / judge disagreement), cluster by document, topic, and question type, and
    write ranked, evidence-backed recommendations that `llb recommend` folds into its summary."""
    from llb.board.miss_analysis import (
        DEFAULT_MISS_THRESHOLD,
        analysis_out_dir,
        analyze_run,
        load_item_provenance,
        refresh_recommendations,
        write_analysis,
    )
    from llb.board.miss_probe import parse_probe_depths, run_probes
    from llb.board.runs import load_run_records
    from llb.core.paths import resolve_data_dir
    from llb.goldset.schema import load_goldset

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        typer.echo(f"[analyze-misses] no manifest.json in {run_dir}", err=True)
        raise typer.Exit(code=2)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config") or {}
    goldset_path = goldset or Path(str(config.get("goldset_path", "")))
    if not str(goldset_path) or not goldset_path.is_file():
        typer.echo(
            f"[analyze-misses] goldset not found: '{goldset_path}' "
            "(the bundle's recorded goldset moved? pass --goldset)",
            err=True,
        )
        raise typer.Exit(code=2)
    items = load_goldset(goldset_path)
    provenance = load_item_provenance(goldset_path)

    # Comparable sibling runs (same split + case count, board dedup rules) back the
    # "try the named alternative model" recommendation with measured numbers.
    alternatives = [
        (record.result.model, record.result.objective_score)
        for record in load_run_records(run_dir.parent)
        if record.split == manifest.get("split")
        and record.result.n_cases == int(manifest.get("n_cases", -1))
    ]
    threshold = miss_threshold if miss_threshold is not None else DEFAULT_MISS_THRESHOLD
    analysis = analyze_run(
        run_dir, items, threshold=threshold, provenance=provenance, alternatives=alternatives
    )

    if probe_top_k and analysis.misses:
        depths = parse_probe_depths(probe_top_k)
        analysis.probes = run_probes(manifest, analysis.misses, items, depths)
        refresh_recommendations(analysis, alternatives=alternatives)
    elif probe_top_k:
        typer.echo("[analyze-misses] no misses to probe; skipping --probe-top-k")

    out = out_dir or analysis_out_dir(resolve_data_dir())
    paths = write_analysis(analysis, out)
    counted = ", ".join(f"{cls}={n}" for cls, n in analysis.class_counts.items() if n) or "none"
    typer.echo(
        f"[analyze-misses] {len(analysis.misses)} of {analysis.n_cases} cases missed ({counted})"
    )
    for rank, rec in enumerate(analysis.recommendations, 1):
        typer.echo(f"[analyze-misses] {rank}. {rec['line']}")
    typer.echo(f"[analyze-misses] report -> {paths['report']}")
    typer.echo(f"[analyze-misses] misses -> {paths['misses']}")


@app.command("score-external-rag")
def score_external_rag_cmd(
    answers: Path = typer.Option(
        ..., "--answers", help="answered goldset JSONL from an external or closed RAG system"
    ),
    csv_out: Optional[Path] = typer.Option(
        None, help="detailed per-row CSV path (default: <answers>.csv)"
    ),
    report_out: Optional[Path] = typer.Option(
        None, help="Markdown report path (default: <answers>.report.md)"
    ),
    answer_field: Optional[str] = typer.Option(
        None, help="answer field to score (default: auto: llm_answer, predicted_answer, ...)"
    ),
    sources_field: Optional[str] = typer.Option(
        None, help="source-list field to flatten (default: auto: llm_sources, sources, ...)"
    ),
    error_field: Optional[str] = typer.Option(
        None, help="error field (default: auto: llm_error, error)"
    ),
    source_limit: int = typer.Option(3, min=0, help="number of top sources to flatten into CSV"),
    label: Optional[str] = typer.Option(None, help="system label in the report"),
    strip_source_footer: bool = typer.Option(
        True,
        "--strip-source-footer/--keep-source-footer",
        help="strip a trailing Source:/Dzherelo: footer before objective scoring",
    ),
    start: Optional[int] = typer.Option(
        None, "--start", min=1, help="start review at 1-based row number"
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help="confirmation-gated restart: clear JSONL human fields before reviewing",
    ),
) -> None:
    """Interactively score an external RAG JSONL; finalize CSV + report when complete."""
    from llb.scoring.external_rag_session import run_external_rag_session

    try:
        run_external_rag_session(
            answers,
            csv_out=csv_out,
            report_out=report_out,
            answer_field=answer_field,
            sources_field=sources_field,
            error_field=error_field,
            source_limit=source_limit,
            strip_source_footer=strip_source_footer,
            label=label,
            start=start,
            clear=clear,
        )
    except ValueError as exc:
        typer.echo(f"[score-external-rag] {exc}", err=True)
        raise typer.Exit(code=2)


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


@app.command("judge-smoke")
def judge_smoke_cmd(
    judge_model: str = typer.Option(..., help="served local judge model id"),
    judge_base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1"
    ),
) -> None:
    """judge diagnostics: strict-JSON judge precheck. Run ONE grounded case and confirm the local judge returns
    a well-formed, non-zero score BEFORE a long judged run; exits non-zero (naming the reason) when
    the judge cannot emit strict JSON or its endpoint is unreachable."""
    from llb.judge.experiment import judge_smoke_check

    result = judge_smoke_check(judge_model, base_url=judge_base_url)
    if result.ok and result.score is not None:
        typer.echo(
            f"[judge-smoke] ok model={judge_model} "
            f"faithfulness={result.score['faithfulness']:.3f} "
            f"answer_relevancy={result.score['answer_relevancy']:.3f}"
        )
        return
    typer.echo(f"[judge-smoke] FAILED model={judge_model}: {result.reason}", err=True)
    raise typer.Exit(code=2)


def _run_screen_with_backend(
    model: str,
    backend: str,
    base_url: str | None,
    cfg: RunConfig,
    extra_tasks: list[str],
    out: Path,
    limit: int | None,
) -> ScreenReport:
    """Launch or reuse a backend endpoint, run the Tier-1 screen, return the report."""
    from llb.screen.public import run_screen

    def do_screen(url: str) -> ScreenReport:
        return run_screen(model, backend, url, extra_tasks=extra_tasks, output_dir=out, limit=limit)

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
            cpu_offload_gb=cfg.cpu_offload_gb,
            kv_offloading_size_gb=cfg.kv_offloading_size_gb,
        )
        with launcher:
            return do_screen(f"{cfg.vllm_host.rstrip('/')}/v1")
    typer.echo(f"[error] backend '{backend}' not supported for the screen", err=True)
    raise typer.Exit(code=2)


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
    from llb.screen.public import run_screen_isolated

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    extra = [t.strip() for t in (tasks or "").split(",") if t.strip()]
    out = out_dir or (cfg.data_dir / "screen")

    def screen_fn() -> ScreenReport:
        return _run_screen_with_backend(model, backend, base_url, cfg, extra, out, limit)

    if isolated:
        vram_reader, pid_reader = best_effort_gpu_readers()
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
    from llb.backends.resolver import resolve_all
    from llb.board.runs import best_per_model, load_run_records, load_screen_reports
    from llb.optimize.tuner import two_stage
    from llb.scoring.aggregate import format_board, rank_board, ranking_policy_note
    from llb.screen.public import select_finalists

    cfg = load_config(None, goldset_path=goldset)
    reports = load_screen_reports(cfg.data_dir / "screen")
    if not reports:
        typer.echo(
            "[pipeline] no screen reports found; run `screen-public` per candidate first", err=True
        )
        raise typer.Exit(code=2)
    finalists = set(select_finalists(reports, top_n))
    typer.echo(f"[pipeline] finalists (top {top_n}/track): {sorted(finalists)}")

    gpus = detect_gpus()
    resolved = {
        r["name"]: r
        for r in resolve_all(
            load_models(manifest),
            max_vram_mb(gpus),
            detect_ram_mb(),
            probes=resolver_probes(offline),
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
