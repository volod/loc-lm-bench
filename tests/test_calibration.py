from llb.judge.calibration import calibrate, emit_worksheet, spearman_rho


def test_perfect_positive():
    assert round(spearman_rho([1, 2, 3, 4], [1, 2, 3, 4]), 6) == 1.0


def test_perfect_negative():
    assert round(spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]), 6) == -1.0


def test_calibrate_trusted():
    result = calibrate([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert result["trusted"] is True and result["rho"] >= 0.6


def test_calibrate_decision_matches_rho():
    result = calibrate([1, 2, 3, 4, 5], [3, 1, 4, 1, 5])
    assert result["trusted"] == (result["rho"] >= result["threshold"])


def test_worksheet_only_calibration_rows(tmp_path):
    items = [
        {"id": "a", "split": "calibration", "question": "q", "reference_answer": "r"},
        {"id": "b", "split": "final", "question": "q", "reference_answer": "r"},
    ]
    out = tmp_path / "ws.csv"
    assert emit_worksheet(items, out) == 1
    assert "human_rating" in out.read_text(encoding="utf-8")
