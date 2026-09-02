"""Score one adjudicator against the frozen calibration probe, with no corpus in the way.

`audit-corpus-conflicts --effort claim` already runs the probe, but it runs it behind a store, a
candidate budget, and an adjudication bill that dwarfs the probe itself. Choosing BETWEEN model
families is the opposite shape: the probe is the whole measurement, and it must be repeatable at
the cost of its own pairs. That is why this is a separate command rather than a flag on the audit.
"""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.prep.conflicts_output import parsed_probe_tiers
from llb.conflicts.claim.adjudicator import (
    DEFAULT_ADJUDICATOR_SEED,
    DEFAULT_ADJUDICATOR_TEMPERATURE,
)
from llb.conflicts.claim.probe import DEFAULT_CALIBRATION_PROBE, PROBE_TIERS

CALIBRATION_METHOD = "corpus-conflict-calibration"
CALIBRATION_REPORT_FILE = "calibration.md"
CALIBRATION_DATA_FILE = "calibration.json"


@app.command("calibrate-conflict-adjudicator")
def calibrate_conflict_adjudicator_cmd(
    conflict_model: str = typer.Option(..., help="local model to score against the frozen probe"),
    conflict_backend: str = typer.Option("ollama", help="local backend: ollama | vllm | openai"),
    conflict_base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL for the adjudicator"
    ),
    conflict_temperature: float = typer.Option(
        DEFAULT_ADJUDICATOR_TEMPERATURE,
        min=0.0,
        max=2.0,
        help="adjudicator sampling temperature; zero makes the comparison repeatable",
    ),
    null_seed: int = typer.Option(DEFAULT_ADJUDICATOR_SEED, help="adjudicator sampling seed"),
    calibration_probe: Optional[Path] = typer.Option(
        None, help=f"frozen-label probe (default {DEFAULT_CALIBRATION_PROBE})"
    ),
    probe_tiers: Optional[str] = typer.Option(
        None,
        help="comma-separated probe tiers to adjudicate (default: every tier the probe declares, "
        f"currently {','.join(PROBE_TIERS)})",
    ),
    out: Optional[Path] = typer.Option(
        None, help=f"report directory (default: $DATA_DIR/{CALIBRATION_METHOD}/<run>/)"
    ),
) -> None:
    """Adjudicate the frozen probe with one model and report the agreement tier by tier."""
    import json
    import time

    from llb.conflicts.claim.adjudicator import build_adjudicator
    from llb.conflicts.claim.calibration import calibrate_adjudicator, log_calibration
    from llb.conflicts.claim.probe import load_calibration_probe
    from llb.conflicts.report.calibration import calibration_report
    from llb.core.paths import resolve_data_dir, resolve_project_path
    from llb.core.store_generations import generation_timestamp

    probe = load_calibration_probe(calibration_probe, parsed_probe_tiers(probe_tiers))
    complete = build_adjudicator(
        conflict_model,
        conflict_backend,
        conflict_base_url,
        temperature=conflict_temperature,
        seed=null_seed,
    )
    assert complete is not None  # a model name is required, so the endpoint always resolves
    started = time.monotonic()
    calibration = calibrate_adjudicator(probe, complete)
    payload = {
        "method": CALIBRATION_METHOD,
        "model": conflict_model,
        "backend": conflict_backend,
        "temperature": conflict_temperature,
        "seed": null_seed,
        "probe": str(
            resolve_project_path(
                calibration_probe if calibration_probe is not None else DEFAULT_CALIBRATION_PROBE
            )
        ),
        "requested_tiers": list(probe.tiers),
        "seconds": round(time.monotonic() - started, 3),
        "calibration": calibration,
    }
    log_calibration(calibration)
    out_dir = (
        out if out is not None else resolve_data_dir() / CALIBRATION_METHOD / generation_timestamp()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / CALIBRATION_REPORT_FILE
    report_path.write_text(calibration_report(payload) + "\n", encoding="utf-8")
    (out_dir / CALIBRATION_DATA_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(
        f"[calibration] {conflict_model}: {calibration['agreements']}/"
        f"{calibration['parsed_pairs']} pairs agree, calibrated="
        f"{'yes' if calibration['calibrated'] else 'no'}"
    )
    typer.echo(f"[calibration] report: {report_path}")
