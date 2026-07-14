"""Tests for miss classifier."""

from llb.board.miss_analysis.model import (
    MISS_ARTIFACT,
    MISS_CLASSES,
    MISS_GENERATION,
    MISS_JUDGE,
    MISS_REFUSAL,
    MISS_RETRIEVAL,
)
from miss_analysis_helpers import _analyze


def test_classifier_separates_every_class_with_zero_leakage(tmp_path):
    analysis = _analyze(tmp_path)
    by_id = {m.item_id: m.miss_class for m in analysis.misses}
    assert by_id == {
        "m-retr": MISS_RETRIEVAL,
        "m-gen": MISS_GENERATION,
        "m-refuse": MISS_REFUSAL,
        "m-empty": MISS_ARTIFACT,
        "m-judge": MISS_JUDGE,
    }
    assert "hit" not in by_id
    assert analysis.class_counts == {cls: 1 for cls in MISS_CLASSES}
