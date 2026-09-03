"""Check or upgrade a vector store, knowledge graph, or prompt-system package."""

from pathlib import Path

import typer

from llb.cli.app import app
from llb.cli.artifact_survey import ManifestFor, survey_command

KIND_STORE = "store"
KIND_GRAPH = "graph"
KIND_PROMPT_SYSTEM = "prompt-system"
KINDS = (KIND_STORE, KIND_GRAPH, KIND_PROMPT_SYSTEM)


@app.command("check-store")
def check_store(
    target: Path = typer.Argument(..., help="vector store, graph store, or prompt-system run dir"),
    kind: str = typer.Option(
        KIND_STORE, help=f"which directory this is: {' | '.join(KINDS)}", metavar="|".join(KINDS)
    ),
    upgrade: bool = typer.Option(
        False,
        "--upgrade",
        help="rewrite every member written at an older contract version at the current one",
    ),
) -> None:
    """Report what every member of a retrieval artifact directory is, without loading a model.

    The chunk rows, graph rows, and package members are read through their registered contracts;
    the vector index, the lexical postings, and the graph database are checked by digest and
    reported by the owner of their format. Nothing here imports FAISS, DuckDB, or an encoder, so
    a store can be inspected on a machine that could not query it.
    """
    from llb.artifacts.retrieval.datasets import (
        graph_dataset_manifest,
        prompt_system_dataset_manifest,
        store_dataset_manifest,
    )

    builders: dict[str, ManifestFor] = {
        KIND_STORE: lambda root: store_dataset_manifest(root),
        KIND_GRAPH: lambda root: graph_dataset_manifest(root),
        KIND_PROMPT_SYSTEM: lambda root: prompt_system_dataset_manifest(root),
    }
    if kind not in builders:
        typer.echo(f"[error] unknown --kind '{kind}'; choose one of {', '.join(KINDS)}", err=True)
        raise typer.Exit(code=2)
    survey_command("check-store", target, builders[kind], upgrade)
