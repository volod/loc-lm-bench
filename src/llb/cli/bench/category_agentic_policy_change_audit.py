"""CLI for the agent policy-change evidence audit: what does changing this constant invalidate?"""

from pathlib import Path

import typer

from llb.cli.app import app

# The committed studies whose published numbers an agent policy change can invalidate.
_AUDITED_DESIGNS = {
    "compact_memory_boundary_surface": (
        "samples/benchmarks/agentic_compact_memory_boundary_surface_design.json"
    ),
    "compact_trigger_guard_collapse": (
        "samples/benchmarks/agentic_compact_trigger_guard_collapse_design.json"
    ),
    "compact_fold_step_crossover": (
        "samples/benchmarks/agentic_compact_fold_step_crossover_design.json"
    ),
}


@app.command("bench-agentic-policy-change-audit")
def bench_agentic_policy_change_audit_cmd(
    field: str = typer.Option(
        ...,
        "--field",
        help="the ContextPolicy constant being changed (e.g. observation_cap_chars)",
    ),
    baseline: str = typer.Option(
        ..., "--baseline", help="the value the published evidence was measured under"
    ),
    candidate: str = typer.Option(..., "--candidate", help="the value being considered"),
    persist: bool = typer.Option(
        True, "--persist/--no-persist", help="write the audit under DATA_DIR"
    ),
) -> None:
    """Report which published agentic numbers a policy-constant change invalidates. No GPU."""
    from llb.bench.agentic_policy_change_audit import (
        AUDITABLE_FIELDS,
        audit_policy_change,
        coerce_policy_value,
        load_audited_design,
    )
    from llb.bench.agentic_policy_change_audit_report import (
        format_policy_change_table,
        persist_policy_change_audit,
        policy_change_summary,
    )
    from llb.cli.helpers import cli_error, load_config
    from llb.core.paths import PROJECT_ROOT

    if field not in AUDITABLE_FIELDS:
        cli_error(f"{field!r} is not an auditable policy field; choose from {AUDITABLE_FIELDS}")
    try:
        values = (coerce_policy_value(field, baseline), coerce_policy_value(field, candidate))
        designs = {
            kind: load_audited_design(PROJECT_ROOT / Path(path))
            for kind, path in _AUDITED_DESIGNS.items()
        }
        audits = audit_policy_change(designs, field=field, baseline=values[0], candidate=values[1])
    except ValueError as exc:
        cli_error(str(exc))
    summary = policy_change_summary(audits, field=field, baseline=values[0], candidate=values[1])
    table = format_policy_change_table(audits, summary)
    typer.echo(table)
    if persist:
        cfg = load_config(None)
        paths = persist_policy_change_audit(audits, summary, data_dir=cfg.data_dir, table=table)
        typer.echo(f"[policy-change-audit] aggregate -> {paths['manifest']}")
