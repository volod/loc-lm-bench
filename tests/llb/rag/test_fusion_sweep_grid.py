"""Focused tests split from ``test_fusion_evidence.py``."""

import pytest
from _fusion_evidence_helpers import (
    _ByQuestion,
    _chunk,
)

from llb.rag.fusion_evidence import (
    build_sweep_rows,
    parse_candidates,
    parse_weights,
)
from llb.rag.fusion_evidence.rows import VECTOR_ROW


def test_parse_weights_dedupes_and_rejects_out_of_range():
    assert parse_weights("0, 0.3 ,0.3, 1") == (0.0, 0.3, 1.0)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        parse_weights("1.5")
    with pytest.raises(ValueError, match="no graph weight"):
        parse_weights(" , ")


def test_parse_candidates_reads_k_as_the_scored_cutoff_and_rejects_a_zero_depth():
    assert parse_candidates("k, 50 ,50, 20") == (None, 50, 20)
    with pytest.raises(ValueError, match="at least 1"):
        parse_candidates("0")
    with pytest.raises(ValueError, match="an integer or 'k'"):
        parse_candidates("deep")
    with pytest.raises(ValueError, match="no candidate depth"):
        parse_candidates(" , ")


def test_a_deeper_pool_surfaces_the_span_both_lanes_agree_on():
    # `d2` is the span BOTH lanes rank -- vector rank 4 (below a k=3 cutoff) and graph rank 2.
    # At depth k its vector evidence is invisible, so the fused row spends its third seat on the
    # graph lane's own top hit; at depth 4 the two lanes' agreement outranks that graph-only hit.
    vector = _ByQuestion(
        {
            "q": [
                _chunk("d1", 0, 10),
                _chunk("d1", 20, 30),
                _chunk("d1", 40, 50),
                _chunk("d2", 0, 10),
            ]
        }
    )
    graph = _ByQuestion(
        {"q": [_chunk("d9", 0, 10), _chunk("d2", 0, 10), _chunk("d8", 0, 10)]},
    )
    rows = build_sweep_rows(
        vector, {"local_khop": graph}, ["q"], k=3, weights=(0.3,), candidates=(None, 4)
    )
    shallow = [chunk["doc_id"] for chunk in rows["fused/local_khop@0.30/d3"].retrieve("q", 3)]
    deep = [chunk["doc_id"] for chunk in rows["fused/local_khop@0.30/d4"].retrieve("q", 3)]
    assert shallow == ["d1", "d1", "d9"]
    assert deep == ["d1", "d1", "d2"]


def test_depths_resolve_against_k_and_deduplicate_into_one_row():
    vector = _ByQuestion({"q": [_chunk("d1", 0, 10)]})
    graph = _ByQuestion({"q": [_chunk("d2", 0, 10)]})
    rows = build_sweep_rows(
        vector,
        {"local_khop": graph},
        ["q"],
        k=10,
        weights=(0.0, 0.3, 1.0),
        candidates=(None, 4, 10, 50),
    )
    assert set(rows) == {
        VECTOR_ROW,
        "graph/local_khop",
        # endpoint weights are lane passthroughs, so they carry no depth variants
        "fused/local_khop@0.00/d10",
        "fused/local_khop@1.00/d10",
        "fused/local_khop@0.30/d10",
        "fused/local_khop@0.30/d50",
    }
    assert vector.calls == 1 and graph.calls == 1  # one pass per lane, at the deepest pool


def test_depth_equal_to_k_reproduces_the_default_fused_row_exactly():
    hits = {"q": [_chunk("d1", 0, 10), _chunk("d1", 20, 30)]}
    graph_hits = {"q": [_chunk("d2", 0, 10), _chunk("d1", 20, 30)]}
    default = build_sweep_rows(
        _ByQuestion(hits), {"local_khop": _ByQuestion(graph_hits)}, ["q"], k=2, weights=(0.3,)
    )
    explicit = build_sweep_rows(
        _ByQuestion(hits),
        {"local_khop": _ByQuestion(graph_hits)},
        ["q"],
        k=2,
        weights=(0.3,),
        candidates=(2,),
    )
    row = "fused/local_khop@0.30/d2"
    assert explicit[row].retrieve("q", 2) == default[row].retrieve("q", 2)
