"""Roster conformance study for the declared answer contract (typed-rag-answer-envelope)."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app

STUDY_METHOD = "answer-envelope"


@app.command("analyze-answer-envelope")
def analyze_answer_envelope_cmd(
    run_dir: list[Path] = typer.Option(
        ...,
        "--run-dir",
        help="run-eval bundle recorded with --answer-format envelope; repeat once per model "
        "over the same item set",
    ),
    out_dir: Optional[Path] = typer.Option(
        None, help="output dir (default: $DATA_DIR/answer-envelope/<timestamp>)"
    ),
) -> None:
    """Report per-model envelope conformance, failure split, and repair rate beside correctness."""
    from llb.bench.common import new_run_timestamp
    from llb.core.paths import resolve_data_dir
    from llb.eval.answer_envelope.study import analyze, render, write

    try:
        report = analyze(run_dir)
        if out_dir is None:
            _, stamp = new_run_timestamp()
            out_dir = resolve_data_dir() / STUDY_METHOD / stamp
        paths = write(report, out_dir)
    except (OSError, ValueError) as exc:
        typer.echo(f"[analyze-answer-envelope] error: {exc}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(render(report))
    typer.echo(f"[analyze-answer-envelope] report -> {paths['report']}")
    typer.echo(f"[analyze-answer-envelope] JSON   -> {paths['json']}")
