"""multihop-budget-answer-conversion -- the `(lane x top_k)` sweep of the answer-quality lane.

The retrieval-budget probe can show a multi-hop coverage ceiling is a property of `top_k`; whether
that coverage reaches the ANSWERS is a different measurement, and this file drives it end to end
with an injected lane runner -- no backend, no store, no GPU.
"""

import json
from pathlib import Path

import pytest
from tests.llb.eval._answer_quality_helpers import FUSED, VECTOR, _gold_item, _row

from llb.core.config import RunConfig
from llb.eval.answer_quality import (
    budget_label,
    conversion_baselines,
    expand_budget_lanes,
    lane_config,
    parse_lane_label,
    parse_lanes,
    run_answer_quality,
    split_budget_label,
)
from llb.eval.answer_quality.models import (
    CONVERSION_CONVERTED,
    CONVERSION_STALLED,
    METRIC_CONTEXT_CHARS,
    VERDICT_ANSWER_GAIN,
    VERDICT_NO_GAIN,
    VERDICT_RETRIEVAL_ONLY,
)

SHIPPED, RAISED = 10, 50
CHUNK_CHARS = 100


def test_a_budget_cell_label_round_trips_into_the_row_and_its_top_k():
    cells = expand_budget_lanes(parse_lanes(f"{VECTOR},{FUSED}"), [SHIPPED, RAISED])
    assert [cell.label for cell in cells] == [
        f"{VECTOR}#k10",
        f"{VECTOR}#k50",
        f"{FUSED}#k10",
        f"{FUSED}#k50",
    ]
    assert [cell.top_k for cell in cells] == [10, 50, 10, 50]
    parsed = parse_lane_label(f"{FUSED}#k50")
    assert parsed.top_k == RAISED
    assert parsed.graph_weight == pytest.approx(0.1)
    assert split_budget_label(parsed.label) == (FUSED, RAISED)
    assert budget_label(FUSED, RAISED) == f"{FUSED}#k50"


def test_a_label_without_a_budget_leaves_the_configs_own_top_k_alone():
    assert parse_lane_label(VECTOR).top_k is None
    assert (
        lane_config(RunConfig(top_k=7), parse_lane_label(VECTOR), run_name_prefix="aq").top_k == 7
    )
    cell = parse_lane_label(f"{VECTOR}#k50")
    assert lane_config(RunConfig(top_k=7), cell, run_name_prefix="aq").top_k == RAISED


@pytest.mark.parametrize("label", [f"{VECTOR}#k0", f"{VECTOR}#kx"])
def test_an_unusable_budget_suffix_is_rejected(label: str):
    with pytest.raises(ValueError, match="retrieval budget"):
        parse_lane_label(label)


def test_every_raised_cell_is_paired_against_the_same_row_at_the_smallest_budget():
    assert conversion_baselines(parse_lanes(f"{VECTOR},{FUSED}"), [SHIPPED, RAISED]) == {
        f"{VECTOR}#k50": f"{VECTOR}#k10",
        f"{FUSED}#k50": f"{FUSED}#k10",
    }


def _bundle(path: Path, question_types: dict[str, str]) -> None:
    """A gold set whose question types live in the needle sidecar beside it."""
    path.write_text(
        "".join(
            _gold_item(item_id).model_dump_json(exclude_none=True) + "\n"
            for item_id in question_types
        ),
        encoding="utf-8",
    )
    (path.parent / "needle_items.jsonl").write_text(
        "".join(
            json.dumps({"id": item_id, "question_type": kind}) + "\n"
            for item_id, kind in question_types.items()
        ),
        encoding="utf-8",
    )


def _retrieval_line(item_id: str, covered: int, served: int) -> str:
    """A two-hop item whose served context carries `covered` of its hops in `served` records."""
    gold = [{"doc_id": f"d{i}", "char_start": 0, "char_end": 10, "text": "a"} for i in range(1, 3)]
    retrieved = [dict(span, rank=rank) for rank, span in enumerate(gold[:covered], 1)]
    filler = {"doc_id": "pad", "char_start": 0, "char_end": CHUNK_CHARS}
    retrieved += [dict(filler, rank=rank) for rank in range(covered + 1, served + 1)]
    return json.dumps({"item_id": item_id, "retrieved": retrieved, "gold_spans": gold}) + "\n"


def _lane_runner(tmp_path: Path, objective, covered):
    """Persist one fake bundle per (lane, split); both callbacks see the cell's own `top_k`."""

    def run_lane(config: RunConfig, items: list, split: str) -> Path:
        run_dir = tmp_path / "run-eval" / f"{config.run_name}-{split}"
        run_dir.mkdir(parents=True, exist_ok=True)
        scores = run_dir / "scores.jsonl"
        scores.write_text(
            "".join(
                json.dumps(_row(item.id, objective(config.top_k, item.id))) + "\n" for item in items
            ),
            encoding="utf-8",
        )
        (run_dir / "retrieval.jsonl").write_text(
            "".join(
                _retrieval_line(item.id, covered(config.top_k, item.id), config.top_k)
                for item in items
            ),
            encoding="utf-8",
        )
        return scores

    return run_lane


def _run(tmp_path: Path, question_types: dict[str, str], objective, covered):
    goldset = tmp_path / "goldset.jsonl"
    _bundle(goldset, question_types)
    return run_answer_quality(
        RunConfig(data_dir=tmp_path, goldset_path=goldset, top_k=SHIPPED),
        parse_lanes(f"{VECTOR},{FUSED}"),
        budgets=[SHIPPED, RAISED],
        out_dir=tmp_path / "answer-quality",
        resamples=200,
        run_lane=_lane_runner(tmp_path, objective, covered),
    )


def _multi_hop(count: int) -> dict[str, str]:
    return {f"q{i}": "multi-hop" for i in range(count)}


def test_a_raised_budget_that_answers_better_is_recorded_as_a_conversion(tmp_path: Path):
    run = _run(
        tmp_path,
        _multi_hop(12),
        objective=lambda k, _: 1.0 if k == RAISED else 0.0,
        covered=lambda k, _: 2 if k == RAISED else 1,
    )
    conversion = run.report["budget_conversion"]
    assert conversion["decision"] == CONVERSION_CONVERTED
    assert conversion["budgets"] == [SHIPPED, RAISED]
    assert {row["decision"] for row in conversion["rows"]} == {VERDICT_ANSWER_GAIN}
    assert [row["row"] for row in conversion["rows"]] == sorted([VECTOR, FUSED])
    assert all(row["base_budget"] == SHIPPED for row in conversion["rows"])


def test_more_evidence_and_the_same_answers_is_recorded_as_a_stall(tmp_path: Path):
    """The negative result the sweep exists to be able to record."""
    run = _run(
        tmp_path,
        _multi_hop(12),
        objective=lambda _k, _id: 0.5,
        covered=lambda k, _: 2 if k == RAISED else 1,
    )
    conversion = run.report["budget_conversion"]
    assert conversion["decision"] == CONVERSION_STALLED
    assert {row["decision"] for row in conversion["rows"]} == {VERDICT_RETRIEVAL_ONLY}
    assert "retrieval-only effect" in conversion["reason"]


def test_a_slice_the_extra_context_costs_is_named_beside_the_conversion(tmp_path: Path):
    types = {**_multi_hop(12), **{f"f{i}": "factoid" for i in range(8)}}
    run = _run(
        tmp_path,
        types,
        objective=lambda k, item_id: (
            (1.0 if k == RAISED else 0.0)
            if item_id.startswith("q")
            else (0.0 if k == RAISED else 1.0)
        ),
        covered=lambda k, item_id: 2 if (k == RAISED and item_id.startswith("q")) else 1,
    )
    conversion = run.report["budget_conversion"]
    assert conversion["decision"] == CONVERSION_CONVERTED
    assert all(row["cost_slices"] == ["factoid"] for row in conversion["rows"])
    assert "COST" in conversion["rows"][0]["reason"]


def test_a_budget_that_buys_nothing_at_all_is_no_gain(tmp_path: Path):
    run = _run(
        tmp_path,
        _multi_hop(12),
        objective=lambda _k, _id: 0.5,
        covered=lambda _k, _id: 1,
    )
    assert run.report["budget_conversion"]["decision"] == VERDICT_NO_GAIN


def test_each_cell_reads_its_coverage_and_its_context_bill_at_its_own_budget(tmp_path: Path):
    """Reading every cell at the base config's `top_k` would erase the thing the sweep measures."""
    run = _run(
        tmp_path,
        _multi_hop(12),
        objective=lambda _k, _id: 0.5,
        covered=lambda k, _: 2 if k == RAISED else 1,
    )
    served = {
        label: lane["overall"]["metrics"][METRIC_CONTEXT_CHARS]["mean"]
        for label, lane in run.report["lanes"].items()
    }
    # One 10-char gold span per covered hop, the rest filler chunks -- so the bill scales with the
    # budget, not with the coverage.
    assert served[f"{VECTOR}#k10"] == pytest.approx(10 + 9 * CHUNK_CHARS)
    assert served[f"{VECTOR}#k50"] == pytest.approx(20 + 48 * CHUNK_CHARS)


def test_the_report_and_the_persisted_artifact_carry_both_budgets(tmp_path: Path):
    run = _run(
        tmp_path,
        _multi_hop(12),
        objective=lambda k, _: 1.0 if k == RAISED else 0.0,
        covered=lambda k, _: 2 if k == RAISED else 1,
    )
    rendered = Path(run.paths["report"]).read_text(encoding="utf-8")
    assert "### Budget conversion (k = 10, 50)" in rendered
    assert "context chars delta" in rendered
    for label in (f"{VECTOR}#k10", f"{VECTOR}#k50", f"{FUSED}#k10", f"{FUSED}#k50"):
        assert label in rendered
    # The conversion readings decide a verdict, so the artifact's own audit of how settled its
    # readings are has to see them too.
    assert f"`{VECTOR}#k50 vs {VECTOR}#k10` objective" in rendered
    persisted = json.loads(Path(run.paths["comparison"]).read_text(encoding="utf-8"))
    assert persisted["metadata"]["top_k"] == "10,50"
    assert persisted["budget_conversion"]["decision"] == CONVERSION_CONVERTED
    assert persisted["cross_readings"][f"{VECTOR}#k50"]["base_lane"] == f"{VECTOR}#k10"


def test_a_case_the_model_never_answered_is_counted_not_read_as_a_bad_answer(tmp_path: Path):
    """A k=50 prompt can outrun the request timeout; that zero is a missing answer, not a wrong
    one, and every metric column would otherwise hide the difference."""
    goldset = tmp_path / "goldset.jsonl"
    _bundle(goldset, _multi_hop(4))

    def run_lane(config: RunConfig, items: list, split: str) -> Path:
        run_dir = tmp_path / "run-eval" / f"{config.run_name}-{split}"
        run_dir.mkdir(parents=True, exist_ok=True)
        scores = run_dir / "scores.jsonl"
        rows = [_row(item.id, 1.0) for item in items]
        if config.top_k == RAISED:
            rows[0] = {**rows[0], "status": "timeout", "objective_score": 0.0}
        scores.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return scores

    run = run_answer_quality(
        RunConfig(data_dir=tmp_path, goldset_path=goldset, top_k=SHIPPED),
        parse_lanes(f"{VECTOR},{FUSED}"),
        budgets=[SHIPPED, RAISED],
        out_dir=tmp_path / "answer-quality",
        resamples=0,
        run_lane=run_lane,
    )
    assert run.report["lanes"][f"{VECTOR}#k10"]["not_ok"] == 0
    assert run.report["lanes"][f"{VECTOR}#k50"]["not_ok"] == 1
    rendered = Path(run.paths["report"]).read_text(encoding="utf-8")
    assert "### Cases the model never answered" in rendered
    assert f"- `{VECTOR}#k50`: 1 of 4 not `ok`" in rendered


def test_a_single_budget_run_carries_no_conversion_block(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _bundle(goldset, _multi_hop(4))
    run = run_answer_quality(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        parse_lanes(f"{VECTOR},{FUSED}"),
        out_dir=tmp_path / "answer-quality",
        resamples=0,
        run_lane=_lane_runner(tmp_path, lambda _k, _id: 1.0, lambda _k, _id: 1),
    )
    assert "budget_conversion" not in run.report
    assert "### Budget conversion" not in Path(run.paths["report"]).read_text(encoding="utf-8")
