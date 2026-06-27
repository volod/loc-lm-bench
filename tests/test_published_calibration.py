"""The committed canonical calibration worksheet still clears the judge-trust gate.

This is what makes the judge calibration gate decision reproducible after a fresh clone: the worksheet (86 human +
judge ratings for the committed `ua_squad_postedited_v1` goldset) is tracked under `calibration/`,
and this test re-derives rho from it on every run -- no model, endpoint, or GPU needed. If someone
edits the worksheet and the calibration stops clearing the gate, CI fails here.
"""

from pathlib import Path

from llb.judge.calibration import DEFAULT_THRESHOLD, _load_ratings, calibrate

REPO = Path(__file__).resolve().parents[1]
WORKSHEET = REPO / "calibration" / "ua_squad_postedited_v1.csv"


def test_committed_calibration_clears_the_gate():
    human, judge = _load_ratings(WORKSHEET)
    assert len(human) == len(judge) == 86  # every calibration item rated + judge-scored
    result = calibrate(human, judge)
    assert result["n"] == 86
    assert result["rho"] >= DEFAULT_THRESHOLD  # the canonical judge is trusted
    assert result["trusted"] is True
    assert round(result["rho"], 3) == 0.628  # pin the committed calibration (catches drift)
