from llb.judge.calibration import (
    calibrate,
    emit_worksheet,
    load_worksheet,
    spearman_rho,
    write_filled_worksheet,
    write_worksheet_rows,
)


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


class _Item:
    def __init__(self, id, split, question, reference_answer):
        self.id = id
        self.split = split
        self.question = question
        self.reference_answer = reference_answer


def test_write_filled_worksheet_prefills_model_answer(tmp_path):
    answers = [
        (_Item("a", "calibration", "q1", "r1"), "Київ - столиця"),
        (_Item("b", "calibration", "q2", "r2"), ""),
    ]
    out = tmp_path / "ws.csv"
    assert write_filled_worksheet(answers, out) == 2
    text = out.read_text(encoding="utf-8")
    assert "Київ - столиця" in text  # model_answer pre-filled
    assert text.strip().endswith(",,")  # human_rating + judge_rating still blank


def test_write_filled_worksheet_prefills_judge_rating(tmp_path):
    import csv

    answers = [
        (_Item("a", "calibration", "q1", "r1"), "ans1"),
        (_Item("b", "calibration", "q2", "r2"), "ans2"),
    ]
    out = tmp_path / "ws.csv"
    assert write_filled_worksheet(answers, out, judge_ratings=[0.81234, 0.4]) == 2
    rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
    assert [r["judge_rating"] for r in rows] == ["0.8123", "0.4"]  # judge pre-filled (rounded)
    assert [r["human_rating"] for r in rows] == ["", ""]  # human column still blank


def test_load_worksheet_fills_missing_columns(tmp_path):
    # A worksheet whose header lacks some columns loads with them added blank, so callers can
    # rely on every WORKSHEET_COLS column being present.
    partial = "item_id,split,question,reference_answer,model_answer,human_rating,judge_rating\n"
    partial += "a,calibration,q,r,ans,,0.5\n"
    out = tmp_path / "partial.csv"
    out.write_text(partial, encoding="utf-8")
    rows, fieldnames = load_worksheet(out)
    assert "provenance" in fieldnames and "human_answer" in fieldnames
    assert rows[0]["judge_rating"] == "0.5" and rows[0]["human_rating"] == ""


def test_write_filled_worksheet_merges_human_columns_on_rerun(tmp_path):
    out = tmp_path / "ws.csv"
    answers = [(_Item("a", "calibration", "q1", "r1"), "ans1")]
    write_filled_worksheet(answers, out)

    # Human fills their columns, then the worksheet is regenerated with the SAME candidate.
    rows, fieldnames = load_worksheet(out)
    rows[0]["human_answer"] = "my answer"
    rows[0]["human_rating"] = "4"
    rows[0]["human_status"] = "rated"
    write_worksheet_rows(out, rows, fieldnames)

    write_filled_worksheet(answers, out)  # same model_answer -> human work survives
    rows, _ = load_worksheet(out)
    assert rows[0]["human_answer"] == "my answer"
    assert rows[0]["human_rating"] == "4" and rows[0]["human_status"] == "rated"


def test_write_filled_worksheet_clears_stale_rating_on_changed_answer(tmp_path):
    out = tmp_path / "ws.csv"
    write_filled_worksheet([(_Item("a", "calibration", "q1", "r1"), "ans1")], out)
    rows, fieldnames = load_worksheet(out)
    rows[0]["human_answer"] = "my answer"
    rows[0]["human_rating"] = "4"
    rows[0]["human_status"] = "rated"
    write_worksheet_rows(out, rows, fieldnames)

    # A DIFFERENT candidate answer invalidates the rating but keeps the human's own answer.
    write_filled_worksheet([(_Item("a", "calibration", "q1", "r1"), "ans2-different")], out)
    rows, _ = load_worksheet(out)
    assert rows[0]["human_rating"] == ""  # stale rating cleared
    assert rows[0]["human_status"] == "pending"
    assert rows[0]["human_answer"] == "my answer"  # human's authored answer preserved
