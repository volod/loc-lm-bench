"""Check extraction ledgers against the ontology axiom set."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app


@app.command("validate-ontology-axioms")
def validate_ontology_axioms_cmd(
    extraction: list[Path] = typer.Option(
        ..., "--extraction", help="extraction.jsonl or a draft bundle dir; repeat for more ledgers"
    ),
    axioms: Path = typer.Option(..., help="axiom set to check against (.ttl, or its .json mirror)"),
    crosscheck: bool = typer.Option(
        False,
        "--crosscheck",
        help="hold the checker to an OWL 2 RL reasoner (needs the [ontology] extra)",
    ),
    data_dir: Optional[Path] = typer.Option(None, help="artifact root (default: $DATA_DIR)"),
    method: Optional[str] = typer.Option(None, help="artifact method directory name"),
    run: Optional[str] = typer.Option(None, help="run directory name (default: a fresh stamp)"),
    fail_on_violations: bool = typer.Option(
        False,
        "--fail-on-violations",
        help="exit non-zero when any axiom is broken (for a gated pipeline step)",
    ),
) -> None:
    """Report every logical inconsistency an axiom set finds in an extraction ledger.

    An axiom is a DOMAIN CLAIM over the closed vocabulary and the induced relations -- at most one
    object per subject, an allowed subject/object type, a disjoint type pair, a direction rule, a
    cardinality bound. The induced ontology cannot express any of them: it is a type inventory,
    and nothing in it can be violated. This command is what makes a ledger's contradictions
    visible BEFORE the drafting pipeline turns them into gold questions and the graph lane
    retrieves them as evidence.

    Nothing here deletes a fact or changes a graph. Every violation is reported with both
    offending facts' exact evidence spans so a reviewer can adjudicate it; `llb build-graph
    --refuse-violations` is the separate, opt-in way to refuse a build over a broken ledger.
    """
    from llb.core.paths import resolve_data_dir
    from llb.prep.ontology.axioms.constants import METHOD_DIR
    from llb.prep.ontology.axioms.run import (
        bundle_dir,
        format_summary,
        publish,
        validate_axioms,
    )

    report = validate_axioms(list(extraction), axioms, crosscheck=crosscheck)
    out_dir = bundle_dir(resolve_data_dir(data_dir), method or METHOD_DIR, run)
    paths = publish(report, out_dir)
    for line in format_summary(report):
        typer.echo(line)
    typer.echo(f"[axioms] report -> {paths['report']}")
    if report.crosscheck is not None and report.crosscheck.ran and not report.crosscheck.agrees:
        typer.echo("[error] the reasoner and the in-repo checker disagree", err=True)
        raise typer.Exit(code=1)
    if fail_on_violations and report.n_violations:
        typer.echo(f"[error] {report.n_violations} axiom violations", err=True)
        raise typer.Exit(code=1)
