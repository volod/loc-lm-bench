"""Recompute the lost-pair stage attribution over audit runs already on disk.

The stage rule reads a bundle's `summary.json` record and its own `findings.jsonl` rows, so asking
what a finished audit would say under today's rule costs no model call, no store, and no corpus:
the run wrote down what it read (`stage_replay.py`), and a store rebuilt since would answer a
different question. A bundle written before that record says so instead of guessing.
"""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.core.contracts.common import JsonObject

STAGE_METHOD = "corpus-conflict-stage"
STAGE_REPORT_FILE = "stage.md"
STAGE_DATA_FILE = "stage.json"


@app.command("recompute-conflict-stage")
def recompute_conflict_stage_cmd(
    run: list[Path] = typer.Option(
        ...,
        "--run",
        help="audit run directory (or its summary.json); repeat to score a whole archive",
    ),
    budget: Optional[int] = typer.Option(
        None,
        "--budget",
        help="also re-read each bundle as if its candidate budget had been this rank cutoff",
    ),
    out: Optional[Path] = typer.Option(
        None, help=f"report directory (default: $DATA_DIR/{STAGE_METHOD}/<run>/)"
    ),
) -> None:
    """Re-read which stage each audited run lost an orderable document pair at."""
    import json

    from llb.conflicts.report_stage_replay import budget_line, replay_line, replay_report
    from llb.core.paths import resolve_data_dir
    from llb.core.store_generations import generation_timestamp

    entries = [_entry(path, budget) for path in run]
    out_dir = out if out is not None else resolve_data_dir() / STAGE_METHOD / generation_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / STAGE_REPORT_FILE
    report_path.write_text(replay_report(entries), encoding="utf-8")
    data_path = out_dir / STAGE_DATA_FILE
    data_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    for entry in entries:
        typer.echo(replay_line(entry))
        if "at_budget" in entry:
            typer.echo(budget_line(entry))
    typer.echo(f"[stage] report: {report_path}")


def _entry(path: Path, budget: int | None = None) -> JsonObject:
    """One bundle re-read from the two files it wrote, labelled by its run directory."""
    import json

    from llb.conflicts.constants import FINDINGS_FILE, SUMMARY_FILE
    from llb.conflicts.stage_replay import replay_entry

    summary_path = path / SUMMARY_FILE if path.is_dir() else path
    if not summary_path.is_file():
        raise typer.BadParameter(f"no {SUMMARY_FILE} at {summary_path}")
    findings_path = summary_path.parent / FINDINGS_FILE
    if not findings_path.is_file():
        raise typer.BadParameter(f"no {FINDINGS_FILE} beside {summary_path}")
    rows = [
        json.loads(line) for line in findings_path.read_text(encoding="utf-8").splitlines() if line
    ]
    return replay_entry(
        summary_path.parent.name,
        str(summary_path),
        json.loads(summary_path.read_text(encoding="utf-8")),
        rows,
        budget=budget,
    )
