"""CLI orchestration for restating published crossovers under the shipped summarize-input cap."""

from pathlib import Path
from typing import cast

import typer

from llb.cli.app import app


def _newest_surface_analysis(data_dir: Path) -> dict[str, object] | None:
    """The most recent boundary-surface aggregate on this host, or None when it never ran here."""
    import json

    manifests = sorted(
        (data_dir / "agentic-compact-memory-boundary-surface").glob("*/manifest.json")
    )
    if not manifests:
        return None
    payload = json.loads(manifests[-1].read_text(encoding="utf-8"))
    analysis = cast(dict[str, object], payload["config"]["analysis"])
    analysis["manifest_path"] = str(manifests[-1])
    return analysis


@app.command("bench-agentic-context-compact-crossover-restatement")
def bench_agentic_context_compact_crossover_restatement_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_compact_crossover_restatement_design.json"),
        "--design",
        help="the audited studies and the crossovers they published",
    ),
    surface_aggregate: Path | None = typer.Option(
        None,
        "--surface-aggregate",
        help="boundary-surface manifest to restate (default: the newest under DATA_DIR)",
    ),
    audit_only: bool = typer.Option(
        False,
        "--audit-only",
        help="report the model-free bound audit and stop, without running any cell",
    ),
) -> None:
    """Audit which published cells the shipped cap can move, re-measure only those, and restate."""
    import json

    from llb.backends.ollama import list_models
    from llb.bench.agentic_memory_crossover_restatement import (
        analyze_restatement,
        audit_published_cells,
        run_sensitive_surface_cells,
    )
    from llb.bench.agentic_memory_crossover_restatement_design import (
        load_restatement_design,
        validate_restatement_design,
    )
    from llb.bench.agentic_memory_crossover_restatement_report import (
        format_restatement_table,
        persist_restatement,
    )
    from llb.bench.agentic_memory_transfer import run_control_pilot
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.helpers import best_effort_gpu_readers, cli_error, load_config
    from llb.core.paths import PROJECT_ROOT

    root = PROJECT_ROOT
    try:
        design = load_restatement_design(design_path)
        validate_restatement_design(design, root=root)
        audit = audit_published_cells(design, root=root)
    except ValueError as exc:
        cli_error(str(exc))
    summary = cast(dict[str, object], audit["summary"])
    typer.echo(
        f"[restatement] audit: {summary['n_bound_invariant']}/{summary['n_cells']} published cells "
        f"are bit-identical under both bounds, {summary['n_bound_sensitive']} are bound-sensitive"
    )
    for cell in cast(list[dict[str, object]], summary["sensitive"]):
        typer.echo(
            f"[restatement] bound-sensitive: {cell['study_kind']} {cell['cell_id']} "
            f"(depth {cell['depth']}, guard {cell['max_prompt_chars']}, "
            f"elided {cell['trigger_elided_chars']} chars)"
        )
    if audit_only:
        typer.echo(json.dumps(summary, indent=2, sort_keys=True))
        return

    held = cast(dict[str, object], design["held_fixed"])
    model = cast(str, held["model"])
    backend = cast(str, held["backend"])
    if model not in set(list_models()):
        cli_error(f"the pinned restatement model is not installed: {model}")

    cfg = load_config(
        None,
        model=model,
        backend=backend,
        max_model_len=int(cast(int, held["max_model_len"])),
        seed=int(cast(int, design["seed"])),
        temperature=0.0,
    )
    published_surface = (
        json.loads(surface_aggregate.read_text(encoding="utf-8"))["config"]["analysis"]
        if surface_aggregate is not None
        else _newest_surface_analysis(cfg.data_dir)
    )
    if published_surface is None:
        cli_error(
            "no boundary-surface aggregate is available to restate; pass --surface-aggregate "
            "or run make bench-agentic-context-compact-memory-boundary-surface first"
        )
    vram_reader, pid_reader = best_effort_gpu_readers()
    meter = ThroughputMeter()

    def run(complete: LLMComplete) -> tuple[dict[str, object], list[dict[str, object]]]:
        _report, control = run_control_pilot(
            cast(dict[str, object], design["control_recheck"]),
            model=model,
            backend=backend,
            complete=complete,
        )
        control["model_family"] = held["model_family"]
        typer.echo(
            f"[restatement] control completion={cast(float, control['completion']):.3f} "
            f"eligible={str(control['eligible']).lower()}"
        )
        if not control["eligible"]:
            return control, []
        return control, run_sensitive_surface_cells(
            design,
            audit,
            root=root,
            model=model,
            backend=backend,
            complete=complete,
            data_dir=cfg.data_dir,
        )

    control_row, restated_rows = drive_with_backend(
        cfg,
        run,
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
        meter=meter,
    )
    control_row["tokens_per_s"] = meter.tokens_per_s
    analysis = analyze_restatement(
        design, audit, control_row, published_surface, restated_rows, root=root
    )
    table = format_restatement_table(analysis)
    paths = persist_restatement(
        design,
        analysis,
        data_dir=cfg.data_dir,
        table=table,
        tokens_per_s=meter.tokens_per_s,
    )
    typer.echo(table)
    typer.echo(f"[restatement] aggregate -> {paths['manifest']}")
    # A moved crossover is a measured outcome; only an ineligible pinned family is a failure.
    if not control_row["eligible"]:
        raise typer.Exit(code=2)
