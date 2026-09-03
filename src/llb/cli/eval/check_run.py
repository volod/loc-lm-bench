"""Check or upgrade a published run bundle against its registered artifact contracts."""

from pathlib import Path

import typer

from llb.artifacts.runs.datasets import KIND_BENCHMARK, KIND_RUN
from llb.cli.app import app
from llb.cli.artifact_survey import ManifestFor, survey_command
from llb.core.contracts.artifacts import DatasetManifest

KINDS = (KIND_RUN, KIND_BENCHMARK)


@app.command("check-run")
def check_run(
    run_dir: Path = typer.Argument(..., help="a published run bundle directory"),
    kind: str = typer.Option(
        "",
        help=("override the kind read off the bundle's own manifest: " + " | ".join(KINDS)),
        metavar="|".join(KINDS),
    ),
    upgrade: bool = typer.Option(
        False,
        "--upgrade",
        help="rewrite every member written at an older contract version at the current one",
    ),
) -> None:
    """Report what every member of a run bundle is, without loading a model or a store.

    Run it before a board, a paired comparison, or an external handoff reads the bundle: the
    manifest, the score rows, the retrieval and probe rows, and each sidecar the lane added are
    bound to their registered contracts and read at the current version. A bundle this build
    published carries its own description and is checked against exactly that; a bundle written
    before descriptions existed is described at the kind its manifest states, which `--kind`
    overrides.
    """
    if kind and kind not in KINDS:
        typer.echo(f"[error] unknown --kind '{kind}'; choose one of {', '.join(KINDS)}", err=True)
        raise typer.Exit(code=2)
    survey_command("check-run", run_dir, _manifest_for(kind), upgrade)


def _manifest_for(kind: str) -> ManifestFor:
    """The published description when the bundle has one, otherwise one of the stated kind."""

    def build(root: Path) -> DatasetManifest:
        from llb.artifacts.datasets import load_dataset_manifest
        from llb.artifacts.runs.bundle import run_bundle_kind
        from llb.artifacts.runs.datasets import run_bundle_manifest

        published = load_dataset_manifest(root)
        if published is not None:
            return published
        return run_bundle_manifest(root, kind=kind or run_bundle_kind(root))

    return build
