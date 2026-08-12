"""Recompute both decision-grouping rules over audit runs already on disk.

The rules read `findings.jsonl` and nothing else, so comparing them across corpora costs no model
call and no store: a run measured on another host months ago answers the question as well as a run
made today. That is the whole reason this is a separate command rather than a flag on the audit --
the comparison is a re-reading of committed artifacts, not a new measurement.
"""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.core.contracts.common import JsonObject

GRANULARITY_METHOD = "corpus-conflict-granularity"
GRANULARITY_REPORT_FILE = "granularity.md"
GRANULARITY_DATA_FILE = "granularity.json"


@app.command("compare-conflict-granularity")
def compare_conflict_granularity_cmd(
    run: list[Path] = typer.Option(
        ...,
        "--run",
        help="audit run directory (or its findings.jsonl); repeat for a cross-corpus comparison",
    ),
    out: Optional[Path] = typer.Option(
        None, help=f"report directory (default: $DATA_DIR/{GRANULARITY_METHOD}/<run>/)"
    ),
) -> None:
    """Compare the transitive and shared-unit decision groupings over audited runs."""
    import json

    from llb.conflicts.granularity import RULE_SHARED_UNIT, RULE_TRANSITIVE
    from llb.conflicts.report_granularity import comparison_report
    from llb.core.paths import resolve_data_dir
    from llb.core.store_generations import generation_timestamp

    entries = [_entry(path) for path in run]
    default_out = resolve_data_dir() / GRANULARITY_METHOD / generation_timestamp()
    out_dir = out if out is not None else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / GRANULARITY_REPORT_FILE
    report_path.write_text(comparison_report(entries), encoding="utf-8")
    data_path = out_dir / GRANULARITY_DATA_FILE
    data_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    for entry in entries:
        rules = entry["granularity"]["rules"]
        low, high = entry["granularity"]["decision_range"]
        typer.echo(
            f"[granularity] {entry['label']}: {entry['granularity']['rows']} rows -> "
            f"{rules[RULE_TRANSITIVE]['groups']} {RULE_TRANSITIVE} / "
            f"{rules[RULE_SHARED_UNIT]['groups']} {RULE_SHARED_UNIT} groups "
            f"(decisions {low} - {high}; "
            f"{rules[RULE_SHARED_UNIT]['rows_in_multiple_groups']} rows in two shared-unit groups)"
        )
    typer.echo(f"[granularity] report: {report_path}")


def _entry(path: Path) -> JsonObject:
    """One run's rows, recomputed under both rules and labelled by its run directory."""
    import json

    from llb.conflicts.constants import FINDINGS_FILE
    from llb.conflicts.granularity import rows_granularity

    findings = path / FINDINGS_FILE if path.is_dir() else path
    if not findings.is_file():
        raise typer.BadParameter(f"no {FINDINGS_FILE} at {findings}")
    rows = [json.loads(line) for line in findings.read_text(encoding="utf-8").splitlines() if line]
    return {
        "label": findings.parent.name,
        "source": str(findings),
        "granularity": rows_granularity(rows),
    }
