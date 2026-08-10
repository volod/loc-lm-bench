"""CLI for the agent policy-change evidence audit: what does changing this constant invalidate?"""

import typer

from llb.cli.app import app


@app.command("bench-agentic-policy-change-audit")
def bench_agentic_policy_change_audit_cmd(
    field: list[str] = typer.Option(
        ...,
        "--field",
        help=(
            "the ContextPolicy constant being changed (e.g. observation_cap_chars); repeat it, "
            "with --baseline/--candidate, to audit a compound change as ONE change"
        ),
    ),
    baseline: list[str] = typer.Option(
        ..., "--baseline", help="the value the published evidence was measured under, per --field"
    ),
    candidate: list[str] = typer.Option(
        ..., "--candidate", help="the value being considered, per --field"
    ),
    persist: bool = typer.Option(
        True, "--persist/--no-persist", help="write the audit under DATA_DIR"
    ),
) -> None:
    """Report which published agentic numbers a policy-constant change invalidates. No GPU."""
    from llb.bench.agentic_policy_change_audit import (
        PolicyChange,
        audit_policy_change,
        coerce_policy_value,
    )
    from llb.bench.agentic_policy_change_geometry import load_audited_designs
    from llb.bench.agentic_policy_change_audit_report import (
        format_policy_change_table,
        persist_policy_change_audit,
        policy_change_summary,
    )
    from llb.cli.helpers import cli_error, load_config

    if not len(field) == len(baseline) == len(candidate):
        cli_error(
            "--field, --baseline and --candidate must be repeated the same number of times, got "
            f"{len(field)}/{len(baseline)}/{len(candidate)}"
        )
    if len(set(field)) != len(field):
        cli_error(f"each field can move only once in one change, got {field}")
    try:
        change = PolicyChange(
            baseline={
                name: coerce_policy_value(name, value) for name, value in zip(field, baseline)
            },
            candidate={
                name: coerce_policy_value(name, value) for name, value in zip(field, candidate)
            },
        )
        audits = audit_policy_change(load_audited_designs(), change)
    except ValueError as exc:
        cli_error(str(exc))
    summary = policy_change_summary(audits, change)
    table = format_policy_change_table(audits, summary)
    typer.echo(table)
    if persist:
        cfg = load_config(None)
        paths = persist_policy_change_audit(audits, summary, data_dir=cfg.data_dir, table=table)
        typer.echo(f"[policy-change-audit] aggregate -> {paths['manifest']}")
