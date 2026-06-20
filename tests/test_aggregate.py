import pytest

from llb.scoring.aggregate import (
    ModelResult,
    format_table,
    headline_quality,
    rank_results,
)


def make(model, objective, judge=None, tok=10.0, vram=None, feasible=True):
    return ModelResult(
        model=model,
        backend="ollama",
        objective_score=objective,
        n_cases=10,
        reliability=1.0,
        tokens_per_s=tok,
        peak_vram_mb=vram,
        judge_score=judge,
        feasible=feasible,
    )


def test_headline_objective_only_when_judge_demoted():
    r = make("m", objective=0.4, judge=0.9)
    assert headline_quality(r, judge_trusted=False) == 0.4


def test_headline_blends_when_trusted():
    r = make("m", objective=0.4, judge=0.8)
    assert headline_quality(r, judge_trusted=True, weight_judge=0.5) == pytest.approx(0.6)


def test_rank_orders_by_quality_then_speed():
    rows = rank_results([make("slow", 0.5, tok=5.0), make("fast", 0.5, tok=50.0)])
    assert [r["model"] for r in rows] == ["fast", "slow"]
    assert rows[0]["rank"] == 1


def test_vram_tiebreak_when_quality_and_speed_equal():
    rows = rank_results(
        [make("big", 0.5, tok=10.0, vram=9000), make("small", 0.5, tok=10.0, vram=3000)]
    )
    assert [r["model"] for r in rows] == ["small", "big"]


def test_infeasible_listed_without_rank():
    rows = rank_results([make("ok", 0.5), make("oom", 0.9, feasible=False)])
    by_model = {r["model"]: r for r in rows}
    assert by_model["ok"]["rank"] == 1
    assert by_model["oom"]["rank"] is None


def test_format_table_is_ascii():
    rows = rank_results([make("m", 0.5)])
    table = format_table(rows)
    assert "rank" in table and table.isascii()
