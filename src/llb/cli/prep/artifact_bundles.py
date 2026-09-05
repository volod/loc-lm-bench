"""Check or upgrade a data-prep bundle against its registered artifact contracts."""

from pathlib import Path
from typing import TYPE_CHECKING

import typer

from llb.cli.app import app

if TYPE_CHECKING:
    from llb.artifacts.bundles import MemberReading

KIND_CORPUS = "corpus"
KIND_DRAFT = "draft"
KINDS = (KIND_CORPUS, KIND_DRAFT)


@app.command("check-bundle")
def check_bundle(
    bundle: Path = typer.Argument(..., help="staged corpus directory or draft bundle directory"),
    kind: str = typer.Option(
        KIND_DRAFT, help=f"which bundle this is: {' | '.join(KINDS)}", metavar="|".join(KINDS)
    ),
    upgrade: bool = typer.Option(
        False,
        "--upgrade",
        help="rewrite every member written at an older contract version at the current one",
    ),
) -> None:
    """Validate every project-owned member of a bundle, member by member.

    Run it before a store build, a review session, or an external handoff reads the bundle: each
    member is bound to its registered contract, its content digest is checked, and its records are
    read at the current version. `--upgrade` then rewrites the members an older writer produced.
    """
    from llb.artifacts.bundles import (
        corpus_bundle_manifest,
        draft_bundle_manifest,
        survey_bundle,
        upgrade_bundle,
    )

    if kind not in KINDS:
        typer.echo(f"[error] unknown --kind '{kind}'; choose one of {', '.join(KINDS)}", err=True)
        raise typer.Exit(code=2)
    try:
        manifest = (
            corpus_bundle_manifest(bundle) if kind == KIND_CORPUS else draft_bundle_manifest(bundle)
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"[check-bundle] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if upgrade:
        rewritten = upgrade_bundle(bundle, manifest)
        typer.echo(f"[check-bundle] upgraded {len(rewritten)} member(s): {', '.join(rewritten)}")
        manifest = (
            corpus_bundle_manifest(bundle) if kind == KIND_CORPUS else draft_bundle_manifest(bundle)
        )

    readings = survey_bundle(bundle, manifest)
    for reading in readings:
        typer.echo(_line(reading))
    refused = [reading for reading in readings if reading.refusal]
    typer.echo(
        f"[check-bundle] {len(readings) - len(refused)}/{len(readings)} member(s) readable "
        f"in {manifest.dataset_id} at {bundle}"
    )
    if refused:
        raise typer.Exit(code=1)


def _line(reading: "MemberReading") -> str:
    if reading.refusal:
        return f"  [refused] {reading.member_id} ({reading.path}): {reading.refusal}"
    version = (
        f"{reading.source_version} -> {reading.current_version}"
        if reading.needs_upgrade
        else reading.current_version
    )
    return (
        f"  [ok] {reading.member_id} ({reading.path}): {reading.records} record(s) of "
        f"{reading.schema_id}@{version}"
    )
