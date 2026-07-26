"""Power declaration and resolution for the long-context context-ablation row."""

import json
from pathlib import Path

import pytest

from llb.core.config import RunConfig
from llb.eval.context_ablation.compare import compare_context_strategies
from llb.eval.context_ablation.models import (
    LANE_CLOSED_BOOK,
    LANE_LONG_CONTEXT,
    LANE_RAG,
    POWER_RESOLUTION_SEPARATED,
    POWER_RESOLUTION_UNDECIDABLE,
)
from llb.eval.context_ablation.power import (
    plan_from_artifact,
    required_sample_size,
    resolve_power_analysis,
)
from llb.eval.context_ablation.run import run_context_ablation
from llb.goldset.schema import GoldItem


def _row(item_id: str, objective: float) -> dict[str, object]:
    return {
        "item_id": item_id,
        "split": "final",
        "status": "ok",
        "objective_score": objective,
        "token_f1": objective,
        "exact": 0.0,
        "contains": 0.0,
        "retrieval_hit": 1.0,
    }


def _reference(path: Path, deltas: list[float]) -> None:
    payload = {
        "items": [
            {
                "item_id": f"q{i}",
                "lanes": {
                    LANE_RAG: {"objective_score": 0.5},
                    LANE_LONG_CONTEXT: {"objective_score": 0.5 + delta},
                },
            }
            for i, delta in enumerate(deltas)
        ],
        "lanes": {LANE_LONG_CONTEXT: {"skipped_item_ids": []}},
        "derived": [{"label": "long_context_delta"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_qwen_reference_variance_prices_the_predeclared_target_at_207_items():
    assert (
        required_sample_size(
            0.307817693144,
            0.060,
            alpha=0.05,
            target_power=0.80,
        )
        == 207
    )


def test_plan_reads_paired_variance_and_records_whether_the_planned_set_reaches_it(
    tmp_path: Path,
):
    reference = tmp_path / "comparison.json"
    _reference(reference, [-0.2, 0.0, 0.2, 0.4])
    plan = plan_from_artifact(
        reference,
        minimum_detectable_delta=0.2,
        target_power=0.8,
        confidence=0.95,
        planned_n=20,
    )
    assert plan["reference_n"] == 4
    assert plan["reference_mean"] == pytest.approx(0.1)
    assert plan["required_n"] <= plan["planned_n"]
    assert plan["target_reached"]


def test_resolution_is_separated_only_when_the_new_interval_clears_zero():
    ids = [f"q{i}" for i in range(20)]
    lanes = {
        LANE_CLOSED_BOOK: [_row(item_id, 0.0) for item_id in ids],
        LANE_RAG: [_row(item_id, 0.2) for item_id in ids],
        LANE_LONG_CONTEXT: [_row(item_id, 0.6) for item_id in ids],
    }
    report = compare_context_strategies(lanes, {}, resamples=200)
    plan = {
        "method": "paired-normal-approximation",
        "reference_artifact": "prior.json",
        "reference_n": 20,
        "reference_mean": 0.1,
        "reference_sample_sd": 0.2,
        "minimum_detectable_delta": 0.06,
        "target_power": 0.8,
        "alpha": 0.05,
        "required_n": 20,
        "planned_n": 20,
        "target_reached": True,
    }
    resolved = resolve_power_analysis(report, plan)
    assert resolved["resolution"] == POWER_RESOLUTION_SEPARATED
    assert resolved["direction"] == LANE_LONG_CONTEXT

    lanes[LANE_LONG_CONTEXT] = [
        _row(item_id, 0.4 if i % 2 else 0.0) for i, item_id in enumerate(ids)
    ]
    unresolved_report = compare_context_strategies(lanes, {}, resamples=200)
    unresolved = resolve_power_analysis(unresolved_report, plan)
    assert unresolved["resolution"] == POWER_RESOLUTION_UNDECIDABLE


def test_power_plan_is_on_disk_before_the_first_lane_is_scored(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    items = [
        GoldItem(
            id=f"q{i}",
            lang="uk",
            question=f"q{i}",
            reference_answer="a",
            source_doc_id="doc.txt",
            source_spans=[{"doc_id": "doc.txt", "char_start": 0, "char_end": 1, "text": "a"}],
            provenance="human-authored",
            verified=True,
            split="final",
        )
        for i in range(4)
    ]
    goldset.write_text(
        "".join(item.model_dump_json(exclude_none=True) + "\n" for item in items),
        encoding="utf-8",
    )
    reference = tmp_path / "prior.json"
    _reference(reference, [-0.1, 0.0, 0.1, 0.2])
    out_dir = tmp_path / "context-ablation"

    def lane_runner(config: RunConfig, selected: list[GoldItem], split: str) -> Path:
        assert (out_dir / "power-plan.json").is_file()
        scores = tmp_path / f"{config.context_strategy}-{split}" / "scores.jsonl"
        scores.parent.mkdir(parents=True, exist_ok=True)
        objective = {LANE_CLOSED_BOOK: 0.0, LANE_RAG: 0.5, LANE_LONG_CONTEXT: 0.6}
        scores.write_text(
            "".join(
                json.dumps(_row(item.id, objective[config.context_strategy])) + "\n"
                for item in selected
            ),
            encoding="utf-8",
        )
        return scores

    run = run_context_ablation(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        out_dir=out_dir,
        resamples=50,
        run_lane=lane_runner,
        power_reference=reference,
        minimum_detectable_delta=0.1,
    )
    assert run.paths["power_plan"] == str(out_dir / "power-plan.json")
    assert "power_analysis" in run.report
