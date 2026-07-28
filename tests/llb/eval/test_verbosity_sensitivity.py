import json
from pathlib import Path

import pytest

from llb.eval.verbosity_sensitivity import analyze, render, write
from llb.scoring.verbosity import POLICY_NAME, ranking_score


def _bundle(root: Path, model: str, rows: list[dict[str, float | int | str]]) -> Path:
    run_dir = root / model
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"config": {"model": model}, "metrics": {}}), encoding="utf-8"
    )
    (run_dir / "scores.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return run_dir


def _row(item_id: str, precision: float, recall: float, found: float, length: int) -> dict:
    f1 = 0.0 if not precision else 2 * precision * recall / (precision + recall)
    return {
        "item_id": item_id,
        "objective_score": f1,
        "token_f1": f1,
        "token_precision": precision,
        "token_recall": recall,
        "ranking_score": ranking_score(precision, recall),
        "contains": found,
        "completion_tokens": length,
    }


def test_study_names_rank_flip_between_f1_and_declared_policy(tmp_path: Path):
    verbose = _bundle(
        tmp_path,
        "verbose",
        [_row("a", 0.4, 1.0, 1.0, 20), _row("b", 0.3, 1.0, 1.0, 30)],
    )
    terse = _bundle(
        tmp_path,
        "terse",
        [_row("a", 1.0, 0.5, 0.0, 2), _row("b", 1.0, 0.5, 0.0, 3)],
    )

    report = analyze([verbose, terse])

    assert report["orders"]["token_f1"] == ["terse", "verbose"]
    assert report["orders"][POLICY_NAME] == ["verbose", "terse"]
    assert {change["model"] for change in report["rank_changes"]} == {"terse", "verbose"}
    assert "F1 rank 1 -> chosen-policy rank 2" in render(report)


def test_study_refuses_legacy_bundle_without_decomposition(tmp_path: Path):
    old = _bundle(
        tmp_path,
        "old",
        [{"item_id": "a", "objective_score": 1.0, "token_f1": 1.0}],
    )
    other = _bundle(tmp_path, "other", [_row("a", 1.0, 1.0, 1.0, 1)])

    with pytest.raises(ValueError, match="predates verbosity decomposition"):
        analyze([old, other])


def test_study_refuses_manifest_objective_drift(tmp_path: Path):
    drifted = _bundle(tmp_path, "drifted", [_row("a", 1.0, 1.0, 1.0, 1)])
    (drifted / "manifest.json").write_text(
        json.dumps(
            {
                "config": {"model": "drifted"},
                "metrics": {"objective_score": 0.0},
            }
        ),
        encoding="utf-8",
    )
    other = _bundle(tmp_path, "other", [_row("a", 1.0, 1.0, 1.0, 1)])

    with pytest.raises(ValueError, match="manifest objective"):
        analyze([drifted, other])


def test_study_writes_json_and_ascii_markdown(tmp_path: Path):
    one = _bundle(tmp_path, "one", [_row("a", 1.0, 1.0, 1.0, 1)])
    two = _bundle(tmp_path, "two", [_row("a", 0.5, 1.0, 1.0, 2)])
    report = analyze([one, two])

    paths = write(report, tmp_path / "out")

    assert Path(paths["json"]).exists()
    assert Path(paths["report"]).read_text(encoding="utf-8").isascii()
