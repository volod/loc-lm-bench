"""CLI orchestration for restating published crossovers under the shipped summarize-input cap."""

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import typer

from llb.cli.app import app

if TYPE_CHECKING:
    from llb.bench.published_value.registry import PublishedValueReport
    from llb.bench.common import LLMComplete

# The refresh WROTE its evidence and the designs no longer state their published numbers out of it.
# Distinct from the usage refusals, which exit 2 having written nothing: here the repair is a design
# edit rather than a corrected command, and the operator's next step is to restate the named values.
EXIT_UNRESOLVED_PUBLISHED_VALUES = 3


def _echo_refresh_report(report: "PublishedValueReport") -> None:
    """Say whether the evidence just committed still states the numbers the designs publish.

    Reported with a non-zero exit rather than refused: the write is the first half of the repair
    flow, so the operator keeps the new evidence and learns here which published numbers to restate,
    instead of meeting the same answer as a `make ci` failure once the run context is gone.
    """
    for unresolved in report.unresolved:
        typer.echo(f"[restatement] unresolved: {unresolved.named()}")
    if report.unresolved:
        typer.echo(
            f"[restatement] {len(report.unresolved)}/{len(report.walked)} registered designs "
            "publish values the evidence just committed does not state -- the write stands, so "
            "restate the values named above rather than regenerating again"
        )
        raise typer.Exit(code=EXIT_UNRESOLVED_PUBLISHED_VALUES)
    # The kinds are named rather than counted: a walk that checked nothing reports exactly like one
    # that checked everything, and the operator is being told a clean answer they will act on.
    typer.echo(
        "[restatement] every published value resolves out of the evidence just committed "
        f"(walked: {', '.join(report.walked)})"
    )


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


def _refresh_provenance(design_path: Path, root: Path, host_data_dir: Path) -> None:
    """Regenerate the committed evidence of EVERY registered design, then report and stop.

    The refresh walks the REGISTRY, not this command's `--design`: the committed evidence tree is
    shared, so regenerating it from one design would prune every other registered design's
    aggregates. A `--design` the registry does not know is refused rather than walked alone.
    """
    from llb.bench.published_value.registry import (
        PUBLISHED_VALUE_DESIGNS,
        refresh_committed_evidence_and_report,
    )
    from llb.cli.helpers import cli_error

    registered = {
        (root / design.design_path).resolve() for design in PUBLISHED_VALUE_DESIGNS.values()
    }
    if (root / design_path).resolve() not in registered:
        cli_error(
            f"--refresh-provenance regenerates the committed evidence of every registered "
            f"published-value design, so it cannot be pointed at {design_path} alone; register "
            f"that design in PUBLISHED_VALUE_DESIGNS to have the refresh carry its evidence too"
        )
    try:
        written, report = refresh_committed_evidence_and_report(root=root, data_dir=host_data_dir)
    except (KeyError, ValueError) as exc:
        cli_error(str(exc))
    typer.echo(f"[restatement] provenance manifest -> {written}")
    _echo_refresh_report(report)


def _echo_audit(summary: dict[str, object]) -> None:
    """Report which published cells the shipped bound can move, before anything is re-measured."""
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


def _run_restatement(
    complete: "LLMComplete",
    *,
    design: dict[str, object],
    audit: dict[str, object],
    root: Path,
    model: str,
    backend: str,
    data_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Re-check the pinned family, then re-measure only the cells the audit found sensitive."""
    from llb.bench.memory.crossover_restatement.run import run_sensitive_surface_cells
    from llb.bench.memory.transfer.run import run_control_pilot

    held = cast(dict[str, object], design["held_fixed"])
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
        data_dir=data_dir,
    )


def _published_surface(surface_aggregate: Path | None, data_dir: Path) -> dict[str, object] | None:
    """The boundary-surface analysis being restated: the named aggregate, else the newest one."""
    import json

    from llb.cli.helpers import cli_error

    surface = (
        json.loads(surface_aggregate.read_text(encoding="utf-8"))["config"]["analysis"]
        if surface_aggregate is not None
        else _newest_surface_analysis(data_dir)
    )
    if surface is None:
        cli_error(
            "no boundary-surface aggregate is available to restate; pass --surface-aggregate "
            "or run make bench-agentic-context-compact-memory-boundary-surface first"
        )
    return cast(dict[str, object], surface)


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
    refresh_provenance: bool = typer.Option(
        False,
        "--refresh-provenance",
        help="re-commit the run aggregates EVERY registered design's published values are resolved "
        "against, and their content pins, from the artifacts under DATA_DIR, then report which of "
        "those values the evidence just committed no longer states, and stop",
    ),
) -> None:
    """Audit which published cells the shipped cap can move, re-measure only those, and restate."""
    import json

    from llb.backends.ollama import list_models
    from llb.bench.memory.crossover_restatement.run import (
        analyze_restatement,
        audit_published_cells,
    )
    from llb.bench.memory.crossover_restatement.design import (
        load_restatement_design,
        validate_restatement_design,
    )
    from llb.bench.memory.crossover_restatement.report import (
        format_restatement_table,
        persist_restatement,
    )
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.helpers import best_effort_gpu_readers, cli_error, load_config
    from llb.core.paths import PROJECT_ROOT, resolve_data_dir

    root = PROJECT_ROOT
    host_data_dir = resolve_data_dir()
    if refresh_provenance:
        _refresh_provenance(design_path, root, host_data_dir)
        return
    try:
        design = load_restatement_design(design_path)
        validate_restatement_design(design, root=root, data_dir=host_data_dir)
        audit = audit_published_cells(design, root=root)
    except ValueError as exc:
        cli_error(str(exc))
    summary = cast(dict[str, object], audit["summary"])
    _echo_audit(summary)
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
    published_surface = _published_surface(surface_aggregate, cfg.data_dir)
    vram_reader, pid_reader = best_effort_gpu_readers()
    meter = ThroughputMeter()
    control_row, restated_rows = drive_with_backend(
        cfg,
        partial(
            _run_restatement,
            design=design,
            audit=audit,
            root=root,
            model=model,
            backend=backend,
            data_dir=cfg.data_dir,
        ),
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
