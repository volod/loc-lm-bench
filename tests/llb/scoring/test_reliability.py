"""category expansion reliability -- typed failure taxonomy aggregation."""

import json

import pytest

from llb.scoring import reliability


def test_reliability_report_counts():
    statuses = ["ok", "ok", "empty", "refusal", "timeout", "ok"]
    report = reliability.reliability_report(statuses)
    assert report["n"] == 6 and report["n_ok"] == 3
    assert report["reliability"] == 0.5
    assert report["failures"] == {"empty": 1, "refusal": 1, "timeout": 1}


def test_reliability_report_empty():
    report = reliability.reliability_report([])
    assert report["reliability"] == 0.0 and report["failures"] == {}


def test_read_case_statuses_jsonl(tmp_path):
    rows = [{"item_id": "1", "status": "ok"}, {"item_id": "2", "status": "malformed"}]
    (tmp_path / "scores.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    assert reliability.read_case_statuses(tmp_path) == ["ok", "malformed"]


def test_read_case_statuses_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        reliability.read_case_statuses(tmp_path)
