"""Tests for graph retrieval."""

import pytest
from llb.goldset.schema import GoldItem
from llb.graph.community import assign_communities
from llb.graph.constants import (
    KIND_EDGE_FACT,
    STRATEGY_GLOBAL_COMMUNITY,
    STRATEGY_LOCAL_KHOP,
)
from llb.graph.retrieval import serialize_subgraph
from llb.rag import retrieval as span_metric
from test_graph import TEXT, _answer_span, _build_store, _doc, _extraction, _graph, _span


@pytest.mark.heavy_env
def test_local_khop_recovers_the_answer_span():
    store = _build_store()
    store.strategy = STRATEGY_LOCAL_KHOP
    hits = store.retrieve("Хто написав Кобзар?", 5)
    # the linked seed (Кобзар) reaches Шевченко via the edge; some hit overlaps the gold answer span
    gold = [{"doc_id": "d1", **_answer_span("Шевченко написав Кобзар")}]
    assert span_metric.recall_at_k(hits, gold, 5) == 1.0
    assert any(h["metadata"]["kind"] == KIND_EDGE_FACT for h in hits)


@pytest.mark.heavy_env
def test_global_community_serializes_member_spans():
    store = _build_store()
    store.strategy = STRATEGY_GLOBAL_COMMUNITY
    hits = store.retrieve("Розкажи про Франко", 5)
    assert hits
    # every returned chunk is offset-bearing and from Franko's narrative cluster
    assert all(h["text"] == TEXT[h["char_start"] : h["char_end"]] for h in hits)
    assert any("Франко" in h["text"] for h in hits)


@pytest.mark.heavy_env
def test_khop_depth_bounds_expansion():
    store = _build_store()
    store.strategy = STRATEGY_LOCAL_KHOP
    store.n_seeds = 1
    store.khop_depth = 1
    near = {(h["char_start"], h["char_end"]) for h in store.retrieve("Хто Кобзар?", 20)}
    store.khop_depth = 2
    far = {(h["char_start"], h["char_end"]) for h in store.retrieve("Хто Кобзар?", 20)}
    assert near <= far  # a wider radius never returns fewer spans


@pytest.mark.heavy_env
def test_unlinked_question_returns_empty():
    store = _build_store()
    for strategy in (STRATEGY_LOCAL_KHOP, STRATEGY_GLOBAL_COMMUNITY):
        store.strategy = strategy
        assert store.retrieve("xyzzy недоречне питання 99", 5) == []


@pytest.mark.heavy_env
def test_save_load_roundtrip(tmp_path):
    store = _build_store()
    store.save(tmp_path)
    assert (tmp_path / "nodes.jsonl").exists() and (tmp_path / "edges.jsonl").exists()
    from llb.graph.store import GraphStore

    loaded = GraphStore.load(tmp_path, strategy=STRATEGY_LOCAL_KHOP)
    assert loaded.meta["n_nodes"] == store.meta["n_nodes"]
    assert loaded.retrieve("Хто написав Кобзар?", 3)  # queryable after reload


@pytest.mark.heavy_env
def test_rejects_unknown_strategy():
    pytest.importorskip("duckdb")
    from llb.graph.store import GraphStore

    with pytest.raises(ValueError, match="unknown retrieval strategy"):
        GraphStore.build([_extraction()], [_doc()], strategy="typo")


def test_community_summaries_are_diagnostic_only():
    from llb.graph.summary import summarize_communities

    graph = _graph()
    assign_communities(graph)
    summaries = summarize_communities(graph, lambda _prompt: "тематичний опис", min_size=3)
    assert summaries  # sizable communities summarized
    # the summary text never appears as an offset-bearing retrieved chunk
    relevance = {n.node_id: 1.0 for n in graph.nodes}
    rendered = {r["text"] for r in serialize_subgraph(graph, relevance, k=50)}
    assert "тематичний опис" not in rendered


def test_summarize_skips_endpoint_errors():
    from llb.graph.summary import summarize_communities

    def boom(_prompt: str) -> str:
        raise RuntimeError("endpoint down")

    graph = _graph()
    assign_communities(graph)
    assert summarize_communities(graph, boom, min_size=3) == {}  # errors skipped, not fatal


def test_build_graph_summarize_requires_a_model():
    # --summarize with no endpoint model is a clean CLI error, not a crash
    import typer

    from llb.cli.rag.graph_index import _summarize_graph

    with pytest.raises(typer.Exit) as excinfo:
        _summarize_graph(object(), None)
    assert excinfo.value.exit_code == 2


def test_load_extractions_roundtrip(tmp_path):
    from llb.graph.ingest import load_extractions

    path = tmp_path / "extraction.jsonl"
    path.write_text(_extraction().model_dump_json() + "\n", encoding="utf-8")
    loaded = load_extractions(path)
    assert len(loaded) == 1 and loaded[0].doc_id == "d1"


@pytest.mark.heavy_env
def test_run_eval_with_graph_backend_records_strategy(tmp_path):
    pytest.importorskip("duckdb")
    from llb.backends.base import BackendLauncher, ChatResult
    from llb.core.config import RunConfig
    from llb.executor.runner import run_eval
    from llb.graph.store import GraphStore

    GraphStore.build([_extraction()], [_doc()]).save(tmp_path / "llb" / "graph")
    item = GoldItem(
        id="g1",
        lang="uk",
        question="Хто написав Кобзар?",
        reference_answer="Шевченко",
        source_doc_id="d1",
        source_spans=[_span("Шевченко написав Кобзар").model_dump()],
        provenance="public-reused",
        verified=True,
        split="final",
    )
    goldset = tmp_path / "goldset.jsonl"
    goldset.write_text(item.model_dump_json() + "\n", encoding="utf-8")

    class FakeLauncher(BackendLauncher):
        def __init__(self) -> None:
            super().__init__(model="fake", meta={"backend": "fake"})

        def chat(self, messages, max_tokens, temperature, timeout):
            return ChatResult(
                text="Шевченко", error=None, prompt_tokens=4, completion_tokens=1, latency_s=0.01
            )

    cfg = RunConfig(
        data_dir=str(tmp_path),
        goldset_path=str(goldset),
        retrieval_backend="graph",
        retrieval_strategy="local_khop",
        model="fake",
    )
    res = run_eval(cfg, launcher=FakeLauncher(), emit=False, mirror=lambda *_: None)
    manifest = res["manifest"].model_dump()
    assert manifest["config"]["retrieval_backend"] == "graph"
    assert manifest["config"]["retrieval_strategy"] == "local_khop"
    assert res["retrieval"]["recall_at_k"] == 1.0  # graph context scores on the span metric
