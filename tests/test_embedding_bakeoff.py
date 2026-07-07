"""Embedding bake-off core (`llb.rag.embedding_bakeoff`).

Pure: fake stores expose the `.retrieve` + `.meta` seam and a fake store-builder stands in for the
heavy FAISS build, so scoring, ranking, the consent/open-data gate, and report shaping run in the
lightweight CI install (no GPU, no FAISS, no numpy, no network).
"""

import pytest

from llb.core.contracts import ChunkRecord, SourceSpanRecord
from llb.rag.embedding_bakeoff import (
    KIND_API,
    KIND_LOCAL,
    BuiltStore,
    api_lane_enabled,
    best_recall,
    format_report,
    render_markdown,
    run_bakeoff,
    score_candidate,
    slugify_model,
)


class _FakeStore:
    """Returns fixed hits (truncated to k) and carries store meta (dim / n_indexed / model)."""

    def __init__(
        self, hits: list[ChunkRecord], *, dim: int = 8, n_indexed: int = 3, model: str = "m"
    ):
        self._hits = hits
        self.meta = {"dim": dim, "n_indexed": n_indexed, "embedding_model": model}

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        return self._hits[:k]


def _chunk(doc: str, start: int, end: int) -> ChunkRecord:
    return {"doc_id": doc, "char_start": start, "char_end": end, "text": "x"}


def _span(doc: str, start: int, end: int) -> SourceSpanRecord:
    return {"doc_id": doc, "char_start": start, "char_end": end, "text": "g"}


def _items() -> list[tuple[str, list[SourceSpanRecord]]]:
    return [("питання", [_span("d1", 0, 10)])]


def test_slugify_model_is_filesystem_safe():
    assert slugify_model("intfloat/multilingual-e5-base") == "intfloat_multilingual-e5-base"
    assert slugify_model("BAAI/bge-m3") == "BAAI_bge-m3"
    assert slugify_model("///") == "model"


def test_score_candidate_carries_meta_and_measurements():
    hit = _FakeStore([_chunk("d1", 0, 10)], dim=768, n_indexed=42, model="e5")
    built = BuiltStore(store=hit, embed_seconds=2.5, index_bytes=1000, device="cuda")
    row = score_candidate("intfloat/multilingual-e5-base", built, _items(), k=10)
    assert row["recall_at_k"] == 1.0 and row["mrr"] == 1.0
    assert row["dim"] == 768 and row["n_indexed"] == 42
    assert row["embed_seconds"] == 2.5 and row["index_bytes"] == 1000
    assert row["kind"] == KIND_LOCAL and row["device"] == "cuda"
    assert "cost_usd" not in row  # local row has no cost


def test_score_candidate_records_api_cost():
    built = BuiltStore(
        store=_FakeStore([_chunk("d1", 0, 10)]),
        embed_seconds=1.0,
        index_bytes=10,
        kind=KIND_API,
        cost_usd=0.0123,
    )
    row = score_candidate("cohere/embed-multilingual-v3.0", built, _items(), k=10)
    assert row["kind"] == KIND_API and row["cost_usd"] == pytest.approx(0.0123)


def test_best_recall_ranks_recall_then_mrr_then_throughput():
    rows = [
        {"model": "a", "recall_at_k": 0.5, "mrr": 0.5, "embed_seconds": 1.0},
        {"model": "b", "recall_at_k": 0.9, "mrr": 0.4, "embed_seconds": 9.0},  # best recall
        {
            "model": "c",
            "recall_at_k": 0.9,
            "mrr": 0.4,
            "embed_seconds": 2.0,
        },  # tie recall+mrr, faster
    ]
    assert best_recall(rows) == "c"  # recall+MRR tie broken by faster embed
    assert best_recall([]) is None


def _fixed_builder(store: _FakeStore):
    return lambda model: BuiltStore(store=store, embed_seconds=1.0, index_bytes=100)


def test_run_bakeoff_scores_each_candidate_on_its_own_store():
    # a hits, b misses -> store isolation: each model scored against the store the builder returns.
    stores = {
        "hit-model": _FakeStore([_chunk("d1", 0, 10)]),
        "miss-model": _FakeStore([_chunk("d1", 50, 60)]),
    }

    def build_local(model: str) -> BuiltStore:
        return BuiltStore(store=stores[model], embed_seconds=1.0, index_bytes=100)

    report = run_bakeoff(
        _items(),
        k=10,
        corpus_root="corpus",
        local_models=["hit-model", "miss-model"],
        build_local=build_local,
    )
    by_model = {r["model"]: r for r in report["candidates"]}
    assert by_model["hit-model"]["recall_at_k"] == 1.0
    assert by_model["miss-model"]["recall_at_k"] == 0.0
    assert report["best_recall"] == "hit-model"
    assert report["n"] == 1 and report["k"] == 10


def test_api_lane_disabled_without_api_model():
    assert api_lane_enabled(None, "open", lambda: True) is False


def test_api_lane_refuses_non_open_corpus():
    with pytest.raises(SystemExit, match="open"):
        api_lane_enabled("cohere/embed-multilingual-v3.0", "internal", lambda: True)
    with pytest.raises(SystemExit, match="open"):
        api_lane_enabled("cohere/embed-multilingual-v3.0", None, lambda: True)


def test_api_lane_skips_on_declined_consent():
    assert api_lane_enabled("cohere/embed-multilingual-v3.0", "open", lambda: False) is False


def test_api_lane_runs_only_after_open_and_consent():
    assert api_lane_enabled("cohere/embed-multilingual-v3.0", "open", lambda: True) is True


def test_run_bakeoff_api_row_never_built_without_consent():
    built_api: list[str] = []

    def build_api(model: str) -> BuiltStore:
        built_api.append(model)  # would be the network call
        return BuiltStore(store=_FakeStore([]), embed_seconds=1.0, index_bytes=1, kind=KIND_API)

    report = run_bakeoff(
        _items(),
        k=10,
        corpus_root="corpus",
        local_models=["m"],
        build_local=_fixed_builder(_FakeStore([_chunk("d1", 0, 10)])),
        api_model="cohere/embed-multilingual-v3.0",
        build_api=build_api,
        data_classification="open",
        consent=lambda: False,  # declined
    )
    assert built_api == []  # fake litellm client never called
    assert all(r["kind"] != KIND_API for r in report["candidates"])  # no cohere row


def test_run_bakeoff_api_row_appears_after_consent():
    def build_api(model: str) -> BuiltStore:
        return BuiltStore(
            store=_FakeStore([_chunk("d1", 0, 10)], model=model),
            embed_seconds=1.0,
            index_bytes=1,
            kind=KIND_API,
            cost_usd=0.01,
        )

    report = run_bakeoff(
        _items(),
        k=10,
        corpus_root="corpus",
        local_models=["m"],
        build_local=_fixed_builder(_FakeStore([_chunk("d1", 50, 60)])),
        api_model="cohere/embed-multilingual-v3.0",
        build_api=build_api,
        data_classification="open",
        consent=lambda: True,
    )
    api_rows = [r for r in report["candidates"] if r["kind"] == KIND_API]
    assert len(api_rows) == 1
    assert api_rows[0]["model"] == "cohere/embed-multilingual-v3.0"
    assert report["best_recall"] == "cohere/embed-multilingual-v3.0"  # only the API row hits


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
    assert "Recommended embedder" in md
    assert "build-index --embedding-model" in md


def test_format_report_handles_no_candidates():
    empty = {"k": 10, "n": 0, "corpus_root": "c", "candidates": [], "best_recall": None}
    assert "no candidates" in format_report(empty)  # type: ignore[arg-type]
