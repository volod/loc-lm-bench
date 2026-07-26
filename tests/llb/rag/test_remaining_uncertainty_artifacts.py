"""Read-only re-render checks for the three recorded remaining uncertainty lanes."""

import copy
import json
from pathlib import Path

import pytest

from llb.eval.query_robustness import summarize_query_robustness
from llb.eval.query_robustness_report import render_report as render_robustness
from llb.goldset.schema import load_goldset
from llb.rag.fusion_calibration import format_report as render_calibration
from llb.rag.fusion_evidence.stats import bootstrap_index_sets, bootstrap_ratio
from llb.rag.fusion_routing import HeuristicPolicy, QuestionTypeRouter, ROUTE_GRAPH
from llb.rag.noise_floor import floor_margin
from llb.rag.noise_floor_report import format_margin

ROOT = Path(__file__).resolve().parents[3]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_recorded_routing_ratios_keep_every_number_when_the_qualifier_is_added():
    run = ROOT / ".data/graph-vector-fusion-multihop/20260722T180211Z-routing-calibration"
    artifact = run / "calibration.json"
    goldset = ROOT / ".data/graph-vector-fusion-multihop/goods-draft/goldset.jsonl"
    if not artifact.is_file() or not goldset.is_file():
        pytest.skip("the host's recorded routing-calibration evidence is not present")
    recorded = _load_json(artifact)
    rerendered = copy.deepcopy(recorded)
    gold = load_goldset(goldset)
    results = [
        *[(recorded["tuning_split"], result) for result in rerendered["tuning"].values()],
        (recorded["final_split"], rerendered["final"]),
    ]
    for split, result in results:
        items = [item for item in gold if item.split == split]
        spec = result["policy"]
        router = QuestionTypeRouter(
            recorded["graph_weight"],
            {},
            HeuristicPolicy(spec["long_question_words"], spec["min_linked_entities"]),
        )
        predicted = [router.decide(item.question).route == ROUTE_GRAPH for item in items]
        actual = [len(item.source_spans) > 1 for item in items]
        true_positive = [guess and truth for guess, truth in zip(predicted, actual)]
        index_sets = bootstrap_index_sets(len(items), recorded["resamples"], recorded["seed"])
        for metric, denominator in (("precision", predicted), ("recall", actual)):
            previous = dict(result["route"][metric])
            estimate = bootstrap_ratio(
                true_positive,
                denominator,
                index_sets,
                recorded["confidence"],
            )
            assert {key: estimate[key] for key in ("mean", "lo", "hi")} == previous
            result["route"][metric] = estimate
    text = render_calibration(rerendered)
    assert "Route-quality threshold stability" in text
    assert all(
        "stability" in result["route"]["precision"] and "stability" in result["route"]["recall"]
        for _, result in results
    )


@pytest.mark.parametrize(
    ("run_name", "clean_name"),
    [
        (
            "20260724T121701.652129Z-6feeb0cd727e",
            "20260724T113233.376054Z-7f2b659f138a",
        ),
        (
            "20260724T124802.064874Z-526a1af2007d",
            "20260724T121702.998145Z-b09c3af6a01f",
        ),
    ],
)
def test_recorded_query_robustness_tables_reproduce_before_uncertainty_is_appended(
    run_name: str, clean_name: str
):
    run = ROOT / ".data/query-robustness-evidence/query-robustness" / run_name
    rows_path = run / "robustness.jsonl"
    clean_path = ROOT / ".data/query-robustness-evidence/run-eval" / clean_name / "scores.jsonl"
    if not rows_path.is_file() or not clean_path.is_file():
        pytest.skip("the host's recorded query-robustness evidence is not present")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    clean = [json.loads(line) for line in clean_path.read_text(encoding="utf-8").splitlines()]
    classes = tuple(dict.fromkeys(row["variant_class"] for row in rows))
    result = summarize_query_robustness(rows, clean, classes, seed=13)
    rendered = render_robustness(
        result,
        {
            "model": "recorded",
            "backend": "recorded",
            "split": "final",
            "seed": 13,
            "typo_rate": 0.08,
            "clean_run_dir": "<recorded-clean-run>",
        },
    )
    old_rows = [
        line
        for line in (run / "report.md").read_text(encoding="utf-8").splitlines()
        if any(line.startswith(f"| {variant_class} |") for variant_class in classes)
    ]
    assert old_rows and all(line in rendered for line in old_rows)
    assert "## Paired uncertainty by noise class" in rendered


def test_recorded_noise_floor_margin_keeps_its_cut_and_adds_distance():
    artifact = ROOT / ".data/graph-vector-fusion-multihop/20260724T-noise-floor/comparison.json"
    if not artifact.is_file():
        pytest.skip("the host's recorded noise-floor evidence is not present")
    floor = _load_json(artifact)["noise_floor"]
    previous = floor["margin"]
    margin = floor_margin(floor["lanes"], floor["floor_recall_at_k"])
    assert margin is not None
    assert {key: margin[key] for key in previous} == previous
    assert margin["clearance"] == margin["delta"] - margin["floor"]
    assert "clearance" in format_margin(margin)
