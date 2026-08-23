"""answer-quality-recompare-from-bundles -- re-render a recorded comparison with no model call.

The comparison is pure over the per-case rows its lanes recorded, so a finished run can be re-read
under an improved report instead of re-generating every answer. These tests pin the three things
that makes true: the recorded bundles resolve back into the same lanes, a bundle set that no longer
matches them is refused, and a re-render under an UNCHANGED report reproduces the recorded artifact
byte for byte apart from the two keys that say it was re-rendered.
"""

import json
from pathlib import Path

import pytest
from tests.llb.eval._answer_quality_helpers import (
    FUSED,
    VECTOR,
    _bundle_lane,
    _write_bundle,
)

from llb.core.config import RunConfig
from llb.eval.answer_quality import (
    BundleMismatch,
    parse_lanes,
    read_recorded,
    rerender_from_bundles,
    run_answer_quality,
)
from llb.eval.answer_quality.rerender import RERENDER_SOURCE_KEY, RERENDER_TIMESTAMP_KEY

RERENDER_KEYS = (RERENDER_SOURCE_KEY, RERENDER_TIMESTAMP_KEY)


def _recorded(tmp_path: Path, **kwargs) -> Path:
    """Score a two-lane comparison over full run bundles and return its `comparison.json`."""
    goldset = tmp_path / "goldset.jsonl"
    _write_bundle(goldset)
    run = run_answer_quality(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        parse_lanes(f"{VECTOR},{FUSED}"),
        out_dir=tmp_path / "answer-quality",
        resamples=50,
        run_lane=_bundle_lane(tmp_path),
        **kwargs,
    )
    return Path(run.paths["comparison"])


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_a_recorded_comparison_re_renders_byte_identically_from_its_own_bundles(tmp_path: Path):
    """The acceptance gate: an unchanged report reproduces the recorded artifact exactly."""
    recorded = _recorded(tmp_path)
    original = recorded.read_text(encoding="utf-8")

    run = rerender_from_bundles(recorded, out_dir=tmp_path / "rerendered", timestamp="20260101T")

    assert run.report["lanes"][FUSED]["run_dirs"] == _payload(recorded)["lanes"][FUSED]["run_dirs"]
    rebuilt = _payload(Path(run.paths["comparison"]))
    assert rebuilt["metadata"][RERENDER_SOURCE_KEY] == str(recorded)
    assert rebuilt["metadata"][RERENDER_TIMESTAMP_KEY] == "20260101T"
    for key in RERENDER_KEYS:
        rebuilt["metadata"].pop(key)
    assert json.dumps(rebuilt, ensure_ascii=False, indent=2) == original
    assert recorded.read_text(encoding="utf-8") == original, "the recorded artifact was rewritten"


def test_the_recorded_coverage_columns_come_back_at_each_lanes_own_budget(tmp_path: Path):
    """A re-render recomputes coverage from the sidecar, not from the score rows alone."""
    recorded = _recorded(tmp_path)

    run = rerender_from_bundles(recorded, out_dir=tmp_path / "rerendered")

    assert "span_coverage" in run.report["metrics"]
    fused = run.report["lanes"][FUSED]["overall"]["metrics"]["span_coverage"]["mean"]
    vector = run.report["lanes"][VECTOR]["overall"]["metrics"]["span_coverage"]["mean"]
    assert fused > vector


def test_a_bundle_whose_retrieval_config_drifted_is_refused(tmp_path: Path):
    recorded = _recorded(tmp_path)
    run_dir = Path(_payload(recorded)["lanes"][FUSED]["run_dirs"][0])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["config"]["graph_weight"] = 0.99
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BundleMismatch, match="graph_weight"):
        rerender_from_bundles(recorded, out_dir=tmp_path / "rerendered")


def test_a_lane_repointed_at_another_lanes_bundle_is_refused(tmp_path: Path):
    """The bundle's own run name is what ties it to the lane label that claims it."""
    recorded = _recorded(tmp_path)
    payload = _payload(recorded)
    payload["lanes"][FUSED]["run_dirs"] = payload["lanes"][VECTOR]["run_dirs"]
    _rewrite(recorded, payload)

    with pytest.raises(BundleMismatch, match="run_name"):
        rerender_from_bundles(recorded, out_dir=tmp_path / "rerendered")


def test_a_bundle_scored_on_a_different_model_is_refused(tmp_path: Path):
    recorded = _recorded(tmp_path)
    run_dir = Path(_payload(recorded)["lanes"][VECTOR]["run_dirs"][0])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["config"]["model"] = "some-other-model"
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BundleMismatch, match="model"):
        rerender_from_bundles(recorded, out_dir=tmp_path / "rerendered")


def test_a_drafted_bundle_cannot_stand_in_for_a_verified_lane(tmp_path: Path):
    """`run-eval` stamps `item_grounding` only on a drafted bundle, so the absence is load-bearing."""
    recorded = _recorded(tmp_path)
    run_dir = Path(_payload(recorded)["lanes"][VECTOR]["run_dirs"][0])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["config"]["item_grounding"] = "drafted"
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BundleMismatch, match="item_grounding"):
        rerender_from_bundles(recorded, out_dir=tmp_path / "rerendered")


def test_a_missing_bundle_is_refused_rather_than_shrinking_the_item_set(tmp_path: Path):
    recorded = _recorded(tmp_path)
    payload = _payload(recorded)
    payload["lanes"][FUSED]["run_dirs"] = [str(tmp_path / "run-eval" / "gone")]
    _rewrite(recorded, payload)

    with pytest.raises(BundleMismatch, match="manifest.json"):
        rerender_from_bundles(recorded, out_dir=tmp_path / "rerendered")


def test_a_bundle_that_lost_its_retrieval_sidecar_is_refused(tmp_path: Path):
    """Without the sidecar the coverage columns silently vanish, which is a different comparison."""
    recorded = _recorded(tmp_path)
    for run_dir in _payload(recorded)["lanes"][FUSED]["run_dirs"]:
        (Path(run_dir) / "retrieval.jsonl").unlink()

    with pytest.raises(BundleMismatch, match="no longer measure"):
        rerender_from_bundles(recorded, out_dir=tmp_path / "rerendered")


def test_a_gold_set_that_no_longer_slices_the_items_is_refused(tmp_path: Path):
    recorded = _recorded(tmp_path)
    (tmp_path / "needle_items.jsonl").unlink()

    with pytest.raises(BundleMismatch, match="question-type sidecar"):
        rerender_from_bundles(recorded, out_dir=tmp_path / "rerendered")


def test_a_column_the_recorded_run_never_had_reaches_it_on_a_re_render(tmp_path: Path):
    """The point of the path: an improved report gains a column without re-running generation."""
    recorded = _recorded(tmp_path)
    payload = _payload(recorded)
    payload["metrics"] = [metric for metric in payload["metrics"] if metric != "span_coverage"]
    _rewrite(recorded, payload)

    run = rerender_from_bundles(recorded, out_dir=tmp_path / "rerendered")

    assert "span_coverage" in run.report["metrics"]


def test_a_payload_that_is_not_an_answer_quality_comparison_is_refused(tmp_path: Path):
    other = tmp_path / "comparison.json"
    other.write_text(json.dumps({"verdict": {"best_row": "vector"}}), encoding="utf-8")

    with pytest.raises(BundleMismatch, match="not a compare-answer-quality comparison"):
        read_recorded(other)
