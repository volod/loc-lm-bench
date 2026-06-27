import pytest

from llb.scoring.aggregate import (
    TIER_SCREEN,
    ModelResult,
    average_ranks,
    bootstrap_mean_ci,
    format_board,
    format_table,
    headline_quality,
    pareto_front,
    per_case_quality,
    rank_board,
    ranking_policy_note,
    rank_results,
)


def make(
    model,
    objective,
    judge=None,
    tok=10.0,
    vram=None,
    feasible=True,
    semantic=None,
    cases=None,
    case_judge=None,
    case_semantic=None,
):
    return ModelResult(
        model=model,
        backend="ollama",
        objective_score=objective,
        n_cases=10,
        reliability=1.0,
        tokens_per_s=tok,
        peak_vram_mb=vram,
        judge_score=judge,
        semantic_score=semantic,
        feasible=feasible,
        case_objectives=cases or [],
        case_judge=case_judge or [],
        case_semantic=case_semantic or [],
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
    assert all(line == line.rstrip() for line in table.splitlines())


# --- ranking rigor N-model rigor -------------------------------------------------------------------


def test_bootstrap_mean_ci_brackets_mean_and_none_for_singleton():
    lo, hi = bootstrap_mean_ci([1.0, 1.0, 1.0, 0.0, 0.0], seed=1)
    assert lo <= 0.6 <= hi  # mean is 0.6
    assert bootstrap_mean_ci([0.5]) is None


def test_average_rank_breaks_blend_ties():
    # objective ranks: C,A,B ; judge ranks: B,A,C -> every model averages to rank 2.
    results = [make("A", 0.8, judge=0.6), make("B", 0.7, judge=0.9), make("C", 0.9, judge=0.5)]
    avg = average_ranks(results, judge_trusted=True)
    assert avg == {"A": 2.0, "B": 2.0, "C": 2.0}


def test_average_rank_ignores_judge_when_demoted():
    results = [make("A", 0.8, judge=0.1), make("B", 0.7, judge=0.9)]
    # judge demoted -> only objective counts: A (0.8) ranks 1, B ranks 2.
    assert average_ranks(results, judge_trusted=False) == {"A": 1.0, "B": 2.0}


def test_pareto_front_excludes_dominated():
    # "mid" is dominated by "best" (higher quality, faster, less VRAM).
    best = make("best", 0.9, tok=50.0, vram=3000)
    mid = make("mid", 0.6, tok=20.0, vram=9000)
    cheap = make("cheap", 0.5, tok=80.0, vram=2000)  # not dominated: fastest + least VRAM
    front = pareto_front([best, mid, cheap], judge_trusted=False, weight_judge=0.5)
    assert "best" in front and "cheap" in front and "mid" not in front


def test_rank_board_orders_and_flags_unresolved_overlap():
    # a/b have overlapping objective CIs (near-identical case scores) -> b flagged unresolved.
    a = make("a", 0.70, tok=10, cases=[1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    b = make("b", 0.60, tok=10, cases=[1, 0, 1, 0, 0, 0, 1, 0, 1, 0])
    rows = rank_board([a, b])
    assert [r["model"] for r in rows] == ["a", "b"]
    assert rows[0]["rank"] == 1 and rows[0]["unresolved"] is False
    assert rows[1]["unresolved"] is True  # CI overlaps the model above
    assert "quality_ci" in rows[0]


def test_rank_board_refuses_to_mix_tiers():
    private = make("p", 0.5)
    screen = make("s", 0.5)
    screen.tier = TIER_SCREEN
    with pytest.raises(ValueError, match="across tiers"):
        rank_board([private, screen])


def test_rank_board_refuses_duplicate_model_configs():
    with pytest.raises(ValueError, match="one selected config per model"):
        rank_board([make("same", 0.5), make("same", 0.7)])


def test_format_board_ascii_with_markers():
    rows = rank_board([make("m", 0.5, vram=3000, cases=[1, 0, 1, 0])])
    table = format_board(rows)
    assert table.isascii() and "avg_rank" in table and "Pareto" in table
    assert all(line == line.rstrip() for line in table.splitlines())


# --- ranking rigor statistical completeness --------------------------------------------------------


def test_per_case_quality_blends_only_when_judge_trusted():
    r = make("m", 0.6, cases=[1.0, 0.0], case_judge=[0.5, 0.5])
    assert per_case_quality(r, judge_trusted=False) == [1.0, 0.0]  # objective only
    assert per_case_quality(r, judge_trusted=True, weight_judge=0.5) == [0.75, 0.25]  # blended


def test_rank_board_records_per_signal_cis_when_present():
    a = make("a", 0.7, cases=[1, 1, 0, 1], case_semantic=[0.8, 0.9, 0.7, 0.85])
    rows = rank_board([a])
    assert "objective_ci" in rows[0] and "semantic_ci" in rows[0]
    assert "judge_ci" not in rows[0]  # no judge series / not trusted -> no fabricated CI


def test_rank_board_unresolved_uses_blended_headline_ci_when_judge_trusted():
    # objective is identical; the judge per-case series separates them, and the CI overlap
    # test is computed on the BLENDED headline (the rank-uncertainty comparison method).
    a = make("a", 0.5, judge=0.9, cases=[0.5] * 8, case_judge=[1.0] * 8)
    b = make("b", 0.5, judge=0.1, cases=[0.5] * 8, case_judge=[0.0] * 8)
    rows = rank_board([a, b], judge_trusted=True)
    assert [r["model"] for r in rows] == ["a", "b"]
    assert rows[1]["unresolved"] is False  # disjoint blended CIs -> resolved


def test_rank_board_rejects_incompatible_judge_cohort():
    # judge trusted, but only one model has a judge score -> blending would be apples-to-oranges.
    with pytest.raises(ValueError, match="incompatible judge cohort"):
        rank_board([make("a", 0.7, judge=0.8), make("b", 0.6)], judge_trusted=True)


def test_ranking_policy_note_states_signals_and_judge_state():
    results = [make("a", 0.7, semantic=0.8), make("b", 0.6, semantic=0.7)]
    demoted = ranking_policy_note(results, judge_trusted=False)
    assert "objective" in demoted and "semantic" in demoted and "DEMOTED" in demoted
    trusted = ranking_policy_note(
        [make("a", 0.7, judge=0.8), make("b", 0.6, judge=0.7)], judge_trusted=True, weight_judge=0.5
    )
    assert "judge trusted" in trusted and "0.5" in trusted
