"""Check or upgrade a data-prep bundle against its registered artifact contracts."""

from pathlib import Path

import typer

from llb.cli.app import app
from llb.cli.artifact_survey import survey_command

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
    from llb.artifacts.bundles import corpus_bundle_manifest, draft_bundle_manifest

    if kind not in KINDS:
        typer.echo(f"[error] unknown --kind '{kind}'; choose one of {', '.join(KINDS)}", err=True)
        raise typer.Exit(code=2)
    manifest_for = corpus_bundle_manifest if kind == KIND_CORPUS else draft_bundle_manifest
    survey_command("check-bundle", bundle, manifest_for, upgrade)
