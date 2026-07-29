"""Vector-backed artifact re-reading under the calibrated paired rule."""

import json
from pathlib import Path

from llb.rag.embedding_bakeoff_uncertainty import (
    METRIC_MRR,
    METRIC_RECALL,
    paired_rows,
)
from llb.rag.paired_reading_audit import audit_paired_readings
from llb.rag.paired_reading_audit_report import format_audit


def test_bakeoff_audit_names_a_recorded_adopt_that_calibration_withdraws(tmp_path: Path):
    baseline = "e5"
    candidate = "bge"
    n = 30
    reference = [0.0] * n
    values = [1.0] * 7 + [-1.0] + [0.0] * (n - 8)
    vectors = {
        baseline: {METRIC_RECALL: reference, METRIC_MRR: reference},
        candidate: {METRIC_RECALL: values, METRIC_MRR: values},
    }
    paired = paired_rows(vectors, baseline, resamples=2000, seed=13)
    # Emulate the pre-calibration artifact: its interval cleared zero and both bars were recorded
    # as separated, while the exact sign-flip p for 7 wins / 1 loss is 0.0352.
    for comparison in paired[candidate]["metrics"].values():
        comparison.pop("randomization_p")
        comparison.pop("randomization_method")
        comparison.pop("randomization_samples")
        comparison["stability"].update(
            {
                "reading": "separated",
                "looser_reading": "separated",
                "tighter_reading": "separated",
                "borderline": False,
                "side": None,
            }
        )
        comparison["stability"].pop("randomization_p")
        comparison["stability"].pop("randomization_method")
        comparison["stability"].pop("randomization_samples")
    items = [
        {
            "item_id": f"q{index}",
            "models": {
                baseline: {METRIC_RECALL: 0.0, METRIC_MRR: 0.0},
                candidate: {METRIC_RECALL: value, METRIC_MRR: value},
            },
        }
        for index, value in enumerate(values)
    ]
    payload = {
        "uncertainty": {
            "baseline": baseline,
            "bars": [METRIC_RECALL],
            "resamples": 2000,
            "confidence": 0.95,
            "seed": 13,
        },
        "paired_items": items,
        "candidates": [
            {"model": baseline, "paired_vs_baseline": paired[baseline]},
            {"model": candidate, "paired_vs_baseline": paired[candidate]},
        ],
        "verdict": {"decision": "adopt"},
    }
    target = tmp_path / "compare-embeddings" / "recorded"
    target.mkdir(parents=True)
    (target / "report.json").write_text(json.dumps(payload), encoding="utf-8")

    report = audit_paired_readings(tmp_path)

    assert report["artifacts"] == 1
    assert report["comparisons"] == 4
    assert len(report["reading_changes"]) == 2
    assert report["verdicts"][0]["previous"] == "adopt"
    assert report["verdicts"][0]["calibrated"] == "retain"
    assert report["verdicts"][0]["selection_survives"] is False
    assert len(report["selection_readings"]) == 1
    text = format_audit(report)
    assert "adopt | retain | no | YES" in text
    assert text.isascii()
