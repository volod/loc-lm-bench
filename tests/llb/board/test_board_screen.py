"""Tests for board screen."""

import json
from llb.board.runs import (
    load_screen_reports,
)


def test_load_screen_reports_separates_tier1(tmp_path):
    screens = tmp_path / "screen"
    screens.mkdir()
    (screens / "m.json").write_text(
        json.dumps({"model": "m", "track": "logprob", "results": [{"task": "t", "score": 0.5}]}),
        encoding="utf-8",
    )
    (screens / "junk.json").write_text(json.dumps({"not": "a report"}), encoding="utf-8")
    reports = load_screen_reports(screens)
    assert len(reports) == 1 and reports[0]["track"] == "logprob"
