"""CLI entry points for the held-out paired robotics RAG benchmark."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import cli_error
from llb.core.paths import PROJECT_ROOT

DEFAULT_DESIGN = PROJECT_ROOT / "samples" / "robotics" / "benchmark" / "design.json"
DEFAULT_EMULATOR = PROJECT_ROOT / "samples" / "robotics" / "emulator" / "scenarios.json"
DEFAULT_HFLOW = PROJECT_ROOT / "samples" / "robotics" / "hflow"
DEFAULT_MANUALS = PROJECT_ROOT / "samples" / "robotics" / "benchmark" / "corpus"


@app.command("robotics-rag-design-check")
def robotics_rag_design_check(
    design: Path = typer.Option(DEFAULT_DESIGN, help="Frozen final-split benchmark design."),
) -> None:
    """Validate the prospective design, digest, evidence floor, and fault coverage."""
    from llb.robotics.benchmark.run import validate_benchmark_design

    try:
        report = validate_benchmark_design(design)
    except ValueError as exc:
        cli_error(str(exc), code=1)
    fault_classes = report["fault_classes"]
    if not isinstance(fault_classes, list):
        cli_error("robotics design report has invalid fault classes", code=1)
    typer.echo(
        f"[ok] robotics RAG design {report['task_count']} tasks, "
        f"{len(fault_classes)} mandatory fault classes"
    )


@app.command("robotics-rag-benchmark")
def robotics_rag_benchmark(
    model: str = typer.Option(..., help="Pinned local model name."),
    backend: str = typer.Option("ollama", help="Local backend (currently ollama)."),
    design: Path = typer.Option(DEFAULT_DESIGN, help="Frozen final-split benchmark design."),
    emulator: Path = typer.Option(DEFAULT_EMULATOR, help="Pinned emulator fixture."),
    hflow_fixture: Path = typer.Option(DEFAULT_HFLOW, help="Pinned HFlow evidence fixture."),
    manual_corpus: Path = typer.Option(DEFAULT_MANUALS, help="Committed operation manuals."),
    agent_profile: Optional[Path] = typer.Option(
        None, help="Composed profile JSON (default: newest under DATA_DIR)."
    ),
    data_dir: Optional[Path] = typer.Option(None, help="Artifact root (default: DATA_DIR)."),
) -> None:
    """Run retrieval-on/off model lanes and the deterministic reference controller."""
    from llb.robotics.benchmark.run import run_benchmark

    try:
        output, report = run_benchmark(
            design_path=design,
            emulator_path=emulator,
            hflow_fixture=hflow_fixture,
            manual_corpus=manual_corpus,
            model=model,
            backend=backend,
            agent_profile=agent_profile,
            data_dir=data_dir,
        )
    except (RuntimeError, ValueError) as exc:
        cli_error(str(exc), code=1)
    typer.echo(f"[ok] robotics RAG {report['paired_verdict']['decision']} -> {output}")
