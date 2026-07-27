"""Audit commands for widening a drafted multi-hop review slice."""

import json
from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app


@app.command("audit-multihop-draft")
def audit_multihop_draft_cmd(
    bundle: Path = typer.Option(..., help="widened draft bundle to audit"),
    decision_floor: int = typer.Option(53, min=1, help="accepted-item floor the review must reach"),
    minimum_items: int = typer.Option(
        60, min=1, help="minimum drafted rows required before human-review attrition"
    ),
    out: Optional[Path] = typer.Option(
        None, help="audit JSON (default: <bundle>/multihop_expansion_report.json)"
    ),
) -> None:
    """Check size, labels, exact spans, language, and dedup provenance of a widened slice."""
    from llb.prep.ontology.pipeline.expansion_audit import audit_multihop_expansion

    report = audit_multihop_expansion(
        bundle, decision_floor=decision_floor, minimum_items=minimum_items
    )
    target = out or bundle / "multihop_expansion_report.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    typer.echo(
        "[audit-multihop-draft] "
        f"drafted={report['drafted_multi_hop_items']} floor={decision_floor} "
        f"headroom={report['review_headroom']} ready={report['ready_for_human_review']} -> {target}"
    )
    errors = report["errors"]
    if isinstance(errors, list) and errors:
        for error in errors:
            typer.echo(f"[audit-multihop-draft] ERROR: {error}", err=True)
        raise typer.Exit(code=1)
