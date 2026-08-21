"""CLI orchestration for the seeded cross-family repeat-feedback study."""

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

from llb.cli.app import app

if TYPE_CHECKING:
    from llb.bench.loop_feedback.generalization import FeedbackSeedRun
    from llb.bench.loop_policy.report import AgenticLoopPolicyRun
    from llb.bench.common import LLMComplete


@dataclass(frozen=True, slots=True)
class _GeneralizationPlan:
    """What the design fixes for every family/seed cell of this study.

    Resolved once from the design so each cell is driven from one record rather than from eight
    locals captured by a closure -- and so a cell cannot silently read a different sampling
    temperature or policy than the cell beside it.
    """

    design: dict[str, object]
    roster: list[dict[str, object]]
    seeds: list[int]
    variants: list[str]
    fixed: dict[str, object]
    temperature: float
    max_model_len: int

    @classmethod
    def of(cls, design: dict[str, object]) -> "_GeneralizationPlan":
        """Read the predeclared roster, seeds, arms, policy, and sampling out of the design."""
        from llb.bench.agentic.design_fields import as_int, as_ints, as_mapping, as_rows, as_strs

        return cls(
            design=design,
            roster=as_rows(design, "roster"),
            seeds=as_ints(design, "run_seeds"),
            variants=as_strs(design, "repeat_feedback_variants"),
            fixed=as_mapping(design, "fixed_policy"),
            temperature=float(as_mapping(design, "sampling")["temperature"]),  # type: ignore[arg-type]
            max_model_len=as_int(design, "max_model_len"),
        )


def _run_generalization_cell(
    complete: "LLMComplete",
    *,
    plan: _GeneralizationPlan,
    tasks: list[Any],
    family: str,
    model: str,
    backend: str,
    seed: int,
    budget: Any,
    data_dir: Path,
    meter: Any,
    data_verified: bool,
    verification_ref: str | None,
) -> "AgenticLoopPolicyRun":
    """Run one family/seed cell of the study through the shared loop-policy lane."""
    from llb.bench.agentic.design_fields import as_int, as_str, as_strs
    from llb.bench.loop_policy.run import run_agentic_loop_policy

    return run_agentic_loop_policy(
        tasks,
        model=model,
        backend=backend,
        complete=complete,
        max_steps=[as_int(plan.fixed, "max_steps")],
        malformed_policies=[as_str(plan.fixed, "malformed_call")],
        repeated_call_policies=as_strs(plan.fixed, "repeated_call"),
        repeated_feedback_variants=plan.variants,
        budget=budget,
        data_dir=data_dir,
        run_name=f"{plan.design['study_id']}-{family}-seed={seed}",
        data_verified=data_verified,
        verification_ref=verification_ref,
        meter=meter,
        repeat_feedback_design=plan.design,
        model_family=family,
        run_seed=seed,
    )


def _seed_run(
    plan: _GeneralizationPlan,
    tasks: list[Any],
    *,
    family: str,
    model: str,
    backend: str,
    seed: int,
    data_verified: bool,
    verification_ref: str | None,
    readers: tuple[Any, Any],
) -> "FeedbackSeedRun":
    """Drive one cell on a real backend and read the analysis it produced."""
    from llb.bench.loop_feedback.generalization import FeedbackSeedRun
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.bench._agent_context import resolve_agent_context_budget
    from llb.cli.helpers import load_config

    typer.echo(
        f"[feedback-generalization] family={family} model={model} seed={seed} "
        f"temperature={plan.temperature}"
    )
    cfg = load_config(
        None,
        model=model,
        backend=backend,
        max_model_len=plan.max_model_len,
        seed=seed,
        temperature=plan.temperature,
    )
    meter = ThroughputMeter()
    vram_reader, pid_reader = readers
    result = drive_with_backend(
        cfg,
        partial(
            _run_generalization_cell,
            plan=plan,
            tasks=tasks,
            family=family,
            model=model,
            backend=backend,
            seed=seed,
            budget=resolve_agent_context_budget(cfg, base_url=None, max_prompt_chars=None),
            data_dir=cfg.data_dir,
            meter=meter,
            data_verified=data_verified,
            verification_ref=verification_ref,
        ),
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
        meter=meter,
    )
    if meter.calls == 0:
        raise RuntimeError(f"backend returned no successful generations for {family}/{seed}")
    analysis = result.repeat_feedback_analysis
    if analysis is None:
        raise RuntimeError("generalization run produced no repeat-feedback analysis")
    manifests = {
        report.cell.cell_id: report.paths["manifest"]
        for report in result.reports
        if report.paths is not None
    }
    variant = cast(dict[str, dict[str, object]], analysis["variants"])[plan.variants[1]]
    typer.echo(
        f"[feedback-generalization] family={family} seed={seed} "
        f"completion={cast(float, variant['completion_rate']):.3f} "
        f"supports={str(variant['supports_variant']).lower()} "
        f"throughput={meter.tokens_per_s:.1f} tok/s"
    )
    return FeedbackSeedRun(family, model, seed, analysis, manifests)


def _check_roster_installed(roster: list[dict[str, object]]) -> None:
    """Refuse before any run when a declared model is not on this host."""
    from llb.backends.ollama import list_models
    from llb.cli.helpers import cli_error

    available = set(list_models())
    missing = [cast(str, row["model"]) for row in roster if row["model"] not in available]
    if missing:
        cli_error(f"generalization roster models are not installed: {missing}")


@app.command("bench-agentic-loop-repeat-feedback-generalization")
def bench_agentic_loop_repeat_feedback_generalization_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_loop_feedback_generalization_design.json"),
        "--design",
        help="predeclared roster, seeds, sampling, and cross-family adoption rule",
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
    """Run every predeclared model-family/seed cell and persist one aggregate decision."""
    from llb.bench.agentic.run import load_tasks_file
    from llb.bench.loop_feedback.generalization import (
        analyze_feedback_generalization,
        validate_feedback_generalization_design,
    )
    from llb.bench.loop_feedback.generalization_report import (
        format_feedback_generalization_table,
        persist_feedback_generalization,
    )
    from llb.bench.loop_policy.power import load_repeat_power_design
    from llb.cli.helpers import best_effort_gpu_readers, cli_error, load_config

    design = load_repeat_power_design(design_path)
    tasks = load_tasks_file(tasks_path)
    try:
        validate_feedback_generalization_design(design, tasks)
    except ValueError as exc:
        cli_error(str(exc))
    plan = _GeneralizationPlan.of(design)
    _check_roster_installed(plan.roster)
    readers = best_effort_gpu_readers()
    seed_runs = [
        _seed_run(
            plan,
            tasks,
            family=cast(str, row["model_family"]),
            model=cast(str, row["model"]),
            backend=cast(str, row["backend"]),
            seed=seed,
            data_verified=data_verified,
            verification_ref=verification_ref,
            readers=readers,
        )
        for row in plan.roster
        for seed in plan.seeds
    ]
    analysis = analyze_feedback_generalization(design, seed_runs)
    table = format_feedback_generalization_table(analysis)
    paths = persist_feedback_generalization(
        design,
        analysis,
        data_dir=load_config(None).data_dir,
        task_digest=cast(str, cast(dict[str, object], design["reference"])["task_set_digest"]),
        table=table,
    )
    typer.echo(table)
    typer.echo(json.dumps(analysis, indent=2, sort_keys=True))
    typer.echo(f"[feedback-generalization] aggregate -> {paths['manifest']}")
