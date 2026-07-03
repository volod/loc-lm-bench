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
from llb.config import RunConfig
from llb.screen.public import ScreenReport


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
        score_semantic=score_semantic,
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
