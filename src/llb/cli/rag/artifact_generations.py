"""Check a built store, graph, or prompt-system package against its artifact contracts."""

from pathlib import Path

import typer

from llb.cli.app import app
from llb.cli.artifact_readings import member_line

KIND_STORE = "store"
KIND_GRAPH = "graph"
KIND_PROMPT_SYSTEM = "prompt-system"
KINDS = (KIND_STORE, KIND_GRAPH, KIND_PROMPT_SYSTEM)


@app.command("check-generation")
def check_generation(
    generation: Path = typer.Argument(..., help="store, graph, or prompt-system package directory"),
    kind: str = typer.Option(
        KIND_STORE, help=f"which generation this is: {' | '.join(KINDS)}", metavar="|".join(KINDS)
    ),
) -> None:
    """Validate every member of one generation, member by member.

    Run it before retrieval, a refresh, or an external handoff reads the generation: each
    project-owned member is bound to its registered contract and read at the current version,
    each opaque member names its owner and is checked against the digest recorded when the
    generation was published, and every refusal is reported rather than only the first.
    """
    from llb.artifacts.retrieval_graph.datasets import (
        graph_store_manifest,
        prompt_system_manifest,
        vector_store_manifest,
    )
    from llb.artifacts.retrieval_graph.survey import survey_generation
    from llb.core.store_generations import resolve_store_dir
    from llb.graph.constants import META_FILE as GRAPH_META_FILE
    from llb.rag.vector_store.layout import META_FILE as STORE_META_FILE

    if kind not in KINDS:
        typer.echo(f"[error] unknown --kind '{kind}'; choose one of {', '.join(KINDS)}", err=True)
        raise typer.Exit(code=2)
    builders = {
        KIND_STORE: (vector_store_manifest, STORE_META_FILE),
        KIND_GRAPH: (graph_store_manifest, GRAPH_META_FILE),
        KIND_PROMPT_SYSTEM: (prompt_system_manifest, ""),
    }
    builder, meta_file = builders[kind]
    # A refresh publishes immutable `generations/<ts>/` children; check the live one.
    target = resolve_store_dir(generation, meta_file) if meta_file else generation
    try:
        manifest = builder(target)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"[check-generation] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    readings = survey_generation(target, manifest)
    for reading in readings:
        typer.echo(member_line(reading))
    refused = [reading for reading in readings if reading.refusal]
    typer.echo(
        f"[check-generation] {len(readings) - len(refused)}/{len(readings)} member(s) readable "
        f"in {manifest.dataset_id} at {target}"
    )
    if refused:
        raise typer.Exit(code=1)
