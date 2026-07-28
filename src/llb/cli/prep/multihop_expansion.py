"""Audit commands for widening a drafted multi-hop review slice."""

import json
from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.prep.ontology.constants import DEFAULT_MULTI_HOP_REVIEW_HEADROOM_FRACTION


@app.command("audit-multihop-draft")
def audit_multihop_draft_cmd(
    bundle: Path = typer.Option(..., help="widened draft bundle to audit"),
    minimum_headroom_fraction: float = typer.Option(
        DEFAULT_MULTI_HOP_REVIEW_HEADROOM_FRACTION,
        min=0.0,
        max=1.0,
        help="minimum additions as a fraction of the carried review ledger",
    ),
    out: Optional[Path] = typer.Option(
        None, help="audit JSON (default: <bundle>/multihop_expansion_report.json)"
    ),
) -> None:
    """Check size, labels, exact spans, language, and dedup provenance of a widened slice."""
    from llb.prep.ontology.pipeline.expansion_audit import audit_multihop_expansion

    report = audit_multihop_expansion(bundle, minimum_headroom_fraction=minimum_headroom_fraction)
    target = out or bundle / "multihop_expansion_report.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    typer.echo(
        "[audit-multihop-draft] "
        f"drafted={report['drafted_multi_hop_items']} "
        f"headroom={report['review_headroom_fraction']:.3f} "
        f"required={minimum_headroom_fraction:.3f} "
        f"ready={report['ready_for_human_review']} -> {target}"
    )
    errors = report["errors"]
    if isinstance(errors, list) and errors:
        for error in errors:
            typer.echo(f"[audit-multihop-draft] ERROR: {error}", err=True)
        raise typer.Exit(code=1)
