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
from tests.test_recommend import COHORT, MAMAYLM_V2_12B, MAMAYLM_V2_27B, _summary


@pytest.fixture
def recommend_json(tmp_path: Path) -> Path:
    rec = build_recommendation(COHORT, HostInfo(16, 16380, "RTX 4060 Ti", True))
    path = tmp_path / "pdf_model_choice.json"
    path.write_text(json.dumps(recommendation_payload(rec)), encoding="utf-8")
    return path


def _write_recommendation(tmp_path: Path, cohort) -> Path:
    rec = build_recommendation(cohort, HostInfo(16, 16380, "RTX 4060 Ti", True))
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


def test_drafter_prefers_host_recommendation_when_ollama(recommend_json, capsys):
    # the whole COHORT is ollama-backed, so the drafter matches recommended_for_host
    model_choice.print_drafter(recommend_json)
    model_choice.print_drafter_backend(recommend_json)
    model_choice.print_selection(recommend_json, "recommended_for_host")
    drafter, backend, recommended = capsys.readouterr().out.splitlines()
    assert drafter == recommended
    assert backend == "ollama"


def test_drafter_accepts_vllm_candidate_when_backend_qualifies(tmp_path, capsys):
    cohort = [
        _summary("vllm-fast-e4b", 0.60, 58.9, 14782, 0.237, backend="vllm"),
        _summary("ollama-best", 0.44, 22.9, 15868, 0.179),
        _summary("ollama-slower", 0.31, 26.8, 15167, 0.078),
    ]
    path = _write_recommendation(tmp_path, cohort)
    model_choice.print_drafter(path)
    model_choice.print_drafter_backend(path)
    assert capsys.readouterr().out.splitlines() == ["vllm-fast-e4b", "vllm"]


def test_drafter_can_still_be_restricted_to_ollama(tmp_path, capsys):
    cohort = [
        _summary("vllm-fast-e4b", 0.60, 58.9, 14782, 0.237, backend="vllm"),
        _summary("ollama-best", 0.44, 22.9, 15868, 0.179),
    ]
    path = _write_recommendation(tmp_path, cohort)
    model_choice.print_drafter(path, ["ollama"])
    model_choice.print_drafter_backend(path, ["ollama"])
    assert capsys.readouterr().out.splitlines() == ["ollama-best", "ollama"]


def test_candidate_backend_reports_selected_backend(recommend_json, capsys):
    model_choice.print_candidate_backend(recommend_json, 1)
    assert capsys.readouterr().out.strip() == "ollama"


def test_host_gemma4_field_reports_tiered_cuda_target(capsys):
    model_choice.print_host_gemma4("model", gpu_gb=16)
    model_choice.print_host_gemma4("backend", gpu_gb=16)
    model_choice.print_host_gemma4("max-model-len", gpu_gb=16)
    model_choice.print_host_gemma4("cpu-offload-gb", gpu_gb=16)
    model_choice.print_host_gemma4("kv-offloading-size-gb", gpu_gb=16)
    assert capsys.readouterr().out.splitlines() == [
        "google/gemma-4-12B-it-qat-w4a16-ct",
        "vllm",
        "16384",
        "16",
        "32",
    ]


def test_drafter_fails_loudly_when_no_backend_qualifies(tmp_path):
    cohort = [_summary("vllm-only", 0.60, 58.9, 14782, 0.237, backend="vllm")]
    path = _write_recommendation(tmp_path, cohort)
    with pytest.raises(SystemExit):
        model_choice.print_drafter(path, ["ollama"])
