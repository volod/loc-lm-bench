"""Focused tests split from ``test_fusion_evidence.py``."""

from _fusion_evidence_helpers import (
    _floor_rows,
    _identities,
    _multi_hop_item,
)

from llb.rag.fusion_evidence import (
    evaluate_fusion_evidence,
    format_report,
)
from llb.rag.fusion_evidence.models import METRIC_RECALL
from llb.rag.fusion_evidence.rows import VECTOR_ROW


def test_fusion_floor_is_opt_in():
    items = [_multi_hop_item("mh-1", "q")]
    report = evaluate_fusion_evidence(_floor_rows(), items, 3, baseline=VECTOR_ROW, resamples=20)
    assert "noise_floor" not in report  # the default sweep artifact is unchanged
    assert "Measurement floor" not in format_report(report)


def test_fusion_floor_covers_every_swept_row_and_renders():
    items = [_multi_hop_item("mh-1", "q")]
    rows = _floor_rows(pool_depth=9)
    report = evaluate_fusion_evidence(
        rows,
        items,
        3,
        baseline=VECTOR_ROW,
        resamples=20,
        noise_floor=True,
        noise_floor_replicates=16,
    )
    floor = report["noise_floor"]
    assert set(floor["lanes"]) == set(rows)
    for label, lane in floor["lanes"].items():
        # Each row's floor base is the recall the sweep table published for that row.
        assert (
            lane["recall_at_k"]["base"]
            == report["rows"][label]["overall"]["metrics"][METRIC_RECALL]["mean"]
        )
    rendered = format_report(report)
    assert "### Measurement floor" in rendered and rendered.isascii()
    # The verdict is read on the focus slice, so that slice carries its own floor.
    assert set(report["noise_floor_focus"]["lanes"]) == set(rows)
    assert "### Measurement floor: multi-hop" in rendered


def test_a_fused_row_pool_keeps_the_ranking_the_sweep_published():
    # The pool seam must extend the row's own top-k, not re-fuse a deeper per-lane pool -- that
    # deeper fusion is a DIFFERENT row (the candidate-depth knob).
    rows = _floor_rows(pool_depth=9)
    fused = rows["fused/local_khop@0.30/d3"]
    pool = fused.retrieve_candidate_pool("q", 3, 9)
    assert _identities(pool[:3]) == _identities(fused.retrieve("q", 3))
    assert len(pool) > 3  # and it reaches below the cut, so there is something to perturb
    # Asking the row itself for 9 fuses a 9-deep per-lane pool: a different ranking, not this one.
    assert _identities(fused.retrieve("q", 9)) != _identities(pool)


def test_pool_depth_never_changes_a_swept_row():
    shallow = _floor_rows()
    deep = _floor_rows(pool_depth=9)
    for label, row in shallow.items():
        assert row.retrieve("q", 3) == deep[label].retrieve("q", 3)
