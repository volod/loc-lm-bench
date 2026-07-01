"""Quickstart model-selection bridge: recommendation JSON -> drafter model pick.

`scripts/quickstart.sh` shells out to `llb.quickstart.model_choice` to turn a recommendation
bundle into the local drafter model that drives the whole PDF goldset run. These tests pin that
contract against the REAL `recommendation_payload` shape, so a schema drift can't silently pick
the wrong model or abort a multi-hour draft.
"""

import json
from pathlib import Path

import pytest

from llb.board.recommend import HostInfo, build_recommendation, recommendation_payload
from llb.quickstart import model_choice
from tests.test_recommend import COHORT, MAMAYLM_V2_12B, MAMAYLM_V2_27B


@pytest.fixture
def recommend_json(tmp_path: Path) -> Path:
    rec = build_recommendation(COHORT, HostInfo(16, 16380, "RTX 4060 Ti", True))
    path = tmp_path / "pdf_model_choice.json"
    path.write_text(json.dumps(recommendation_payload(rec)), encoding="utf-8")
    return path


def test_selection_and_count_match_payload(recommend_json, capsys):
    model_choice.print_selection(recommend_json, "best_quality")
    model_choice.print_count(recommend_json)
    out = capsys.readouterr().out.splitlines()
    assert out[0] == MAMAYLM_V2_27B  # highest objective in the cohort
    assert out[1] == str(len(COHORT))


def test_candidate_and_speed_lookup(recommend_json, capsys):
    # candidates are ranked; #1 is the best-quality model and its tok/s round-trips exactly
    model_choice.print_candidate(recommend_json, 1)
    model_choice.print_speed(recommend_json, MAMAYLM_V2_12B)
    out = capsys.readouterr().out.splitlines()
    assert out[0] == MAMAYLM_V2_27B
    assert out[1] == "30.100"  # MAMAYLM_V2_12B tok/s from the cohort


def test_table_surfaces_host_and_named_picks(recommend_json, capsys):
    model_choice.print_table(recommend_json)
    out = capsys.readouterr().out
    assert "gpu=RTX 4060 Ti" in out
    assert f"best_quality={MAMAYLM_V2_27B}" in out
    assert "recommended_for_host=lapa" in out  # vram-adaptive pick at 16 GiB


def test_out_of_range_and_missing_selection_fail_loudly(recommend_json):
    with pytest.raises(SystemExit):
        model_choice.print_candidate(recommend_json, 99)
    with pytest.raises(SystemExit):
        model_choice.print_selection(recommend_json, "no_such_key")


def test_speed_is_zero_for_unknown_model(recommend_json, capsys):
    model_choice.print_speed(recommend_json, "not-a-model")
    assert capsys.readouterr().out.strip() == "0"
