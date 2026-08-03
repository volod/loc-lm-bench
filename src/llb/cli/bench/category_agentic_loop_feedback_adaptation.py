"""CLI orchestration for the seeded family-adapted repeat-feedback study."""

import json
from pathlib import Path
from typing import cast

import typer

from llb.cli.app import app


@app.command("bench-agentic-loop-repeat-feedback-family-adaptation")
def bench_agentic_loop_repeat_feedback_family_adaptation_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_loop_feedback_family_adaptation_design.json"),
        "--design",
        help="predeclared family candidates, wording, seeds, and adoption gates",
    ),
    tasks_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_loop_repeat_power_uk.json"),
        "--tasks",
        help="fixed powered repetition-prone task ledger",
    ),
    data_verified: bool = typer.Option(False, help="stamp human-verified task data"),
    verification_ref: str | None = typer.Option(
        None, help="verification worksheet, sample manifest, or accepted ledger"
    ),
) -> None:
    """Run each predeclared family candidate on two seeds and persist stable routes."""
    from llb.backends.ollama import list_models
    from llb.bench.agentic.run import load_tasks_file
    from llb.bench.agentic_loop_feedback_adaptation import (
        FeedbackAdaptationRun,
        analyze_feedback_adaptation,
        validate_feedback_adaptation_design,
    )
    from llb.bench.agentic_loop_feedback_adaptation_report import (
        format_feedback_adaptation_table,
        persist_feedback_adaptation,
    )
    from llb.bench.agentic_loop_policy import run_agentic_loop_policy
    from llb.bench.agentic_loop_policy_power import load_repeat_power_design
    from llb.bench.agentic_loop_policy_report import AgenticLoopPolicyRun
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.bench._agent_context import resolve_agent_context_budget
    from llb.cli.helpers import best_effort_gpu_readers, cli_error, load_config

    design = load_repeat_power_design(design_path)
    tasks = load_tasks_file(tasks_path)
    try:
        validate_feedback_adaptation_design(design, tasks)
    except ValueError as exc:
        cli_error(str(exc))
    roster = cast(list[dict[str, object]], design["roster"])
    seeds = cast(list[int], design["run_seeds"])
    fixed = cast(dict[str, object], design["fixed_policy"])
    sampling = cast(dict[str, object], design["sampling"])
    temperature = float(cast(float, sampling["temperature"]))
    max_model_len = int(cast(int, design["max_model_len"]))
    available = set(list_models())
    missing = [cast(str, row["model"]) for row in roster if row["model"] not in available]
    if missing:
        cli_error(f"family-adaptation roster models are not installed: {missing}")

    vram_reader, pid_reader = best_effort_gpu_readers()
    seed_runs: list[FeedbackAdaptationRun] = []
    for row in roster:
        family = cast(str, row["model_family"])
        model = cast(str, row["model"])
        backend = cast(str, row["backend"])
        candidate = cast(str, row["candidate_feedback_variant"])
        variants = ["current", candidate]
        for seed in seeds:
            typer.echo(
                f"[feedback-adaptation] family={family} model={model} candidate={candidate} "
                f"seed={seed} temperature={temperature}"
            )
            cfg = load_config(
                None,
                model=model,
                backend=backend,
                max_model_len=max_model_len,
                seed=seed,
                temperature=temperature,
            )
            budget = resolve_agent_context_budget(cfg, base_url=None, max_prompt_chars=None)
            meter = ThroughputMeter()

            def run(complete: LLMComplete) -> AgenticLoopPolicyRun:
                return run_agentic_loop_policy(
                    tasks,
                    model=model,
                    backend=backend,
                    complete=complete,
                    max_steps=[int(cast(int, fixed["max_steps"]))],
                    malformed_policies=[cast(str, fixed["malformed_call"])],
                    repeated_call_policies=cast(list[str], fixed["repeated_call"]),
                    repeated_feedback_variants=variants,
                    budget=budget,
                    data_dir=cfg.data_dir,
                    run_name=f"{design['study_id']}-{family}-seed={seed}",
                    data_verified=data_verified,
                    verification_ref=verification_ref,
                    meter=meter,
                    repeat_feedback_design=design,
                    model_family=family,
                    run_seed=seed,
                )

            result = drive_with_backend(
                cfg,
                run,
                vram_reader=vram_reader,
                pid_usage_reader=pid_reader,
                meter=meter,
            )
            if meter.calls == 0:
                raise RuntimeError(f"backend returned no generations for {family}/{seed}")
            analysis = result.repeat_feedback_analysis
            if analysis is None:
                raise RuntimeError("family-adaptation run produced no feedback analysis")
            manifests = {
                report.cell.cell_id: report.paths["manifest"]
                for report in result.reports
                if report.paths is not None
            }
            seed_runs.append(
                FeedbackAdaptationRun(family, model, seed, candidate, analysis, manifests)
            )
            variant = cast(dict[str, dict[str, object]], analysis["variants"])[candidate]
            typer.echo(
                f"[feedback-adaptation] family={family} seed={seed} "
                f"completion={cast(float, variant['completion_rate']):.3f} "
                f"supports={str(variant['supports_variant']).lower()} "
                f"throughput={meter.tokens_per_s:.1f} tok/s"
            )

    analysis = analyze_feedback_adaptation(design, seed_runs)
    table = format_feedback_adaptation_table(analysis)
    paths = persist_feedback_adaptation(
        design,
        analysis,
        data_dir=load_config(None).data_dir,
        task_digest=cast(str, cast(dict[str, object], design["reference"])["task_set_digest"]),
        table=table,
    )
    typer.echo(table)
    typer.echo(json.dumps(analysis, indent=2, sort_keys=True))
    typer.echo(f"[feedback-adaptation] aggregate -> {paths['manifest']}")
