"""Focused tests split from ``test_embedding_bakeoff.py``."""

from _embedding_bakeoff_helpers import (
    _chunk,
    _FakeStore,
    _fixed_builder,
    _items,
)

from llb.rag.embedding_bakeoff import run_bakeoff
from llb.rag.embedding_bakeoff_report import (
    format_report,
    render_markdown,
)


def test_format_report_is_ascii_and_lists_models():
    report = run_bakeoff(
        _items(),
        k=10,
        corpus_root="corpus",
        local_models=["e5"],
        build_local=_fixed_builder(_FakeStore([_chunk("d1", 0, 10)])),
    )
    text = format_report(report)
    assert text.isascii()  # AGENTS.md: ASCII-only output
    assert "recall@k" in text and "chunks/s" in text and "best (recall@k): e5" in text


def test_render_markdown_has_table_and_recommendation():
    report = run_bakeoff(
        _items(),
        k=10,
        corpus_root="corpus",
        local_models=["e5-base", "e5-large"],
        build_local=_fixed_builder(_FakeStore([_chunk("d1", 0, 10)])),
    )
    md = render_markdown(report)
    assert "| model | kind | recall@k |" in md
    # The point-estimate leader is reported as such; the RECOMMENDATION is the paired verdict
    # (see tests/llb/rag/test_embedding_bakeoff_uncertainty.py).
    assert "Point-estimate leader" in md and "Verdict:" in md
    assert "build-index --embedding-model" in md


def test_format_report_handles_no_candidates():
    empty = {"k": 10, "n": 0, "corpus_root": "c", "candidates": [], "best_recall": None}
    assert "no candidates" in format_report(empty)  # type: ignore[arg-type]
