"""Milestone 6 GraphRAG backend: construction, community detection, the two span-preserving
retrieval strategies, persistence, the tagged-diagnostic summaries, and the full vertical.

The pure pieces (build / community / linking / serialize / ingest) run everywhere. The GraphStore
tests use the DuckDB k-hop + community engine, so they `importorskip("duckdb")` (the `[graph]`
extra) -- skipped in the base CI install, run locally via `make test`.
"""

import pytest

from llb.goldset.schema import GoldItem, SourceSpan
from llb.graph.build import build_graph
from llb.graph.community import assign_communities, detect_communities
from llb.graph.constants import (
    KIND_EDGE_FACT,
    KIND_NODE_MENTION,
    STRATEGY_GLOBAL_COMMUNITY,
    STRATEGY_LOCAL_KHOP,
)
from llb.graph.model import KnowledgeGraph
from llb.graph.retrieval import (
    link_communities,
    link_seed_nodes,
    morph_key,
    node_link_scores,
    serialize_subgraph,
    tokenize,
)
from llb.prep.ontology.models import DocExtraction, DocRecord, Entity, Section, SROFact
from llb.rag import retrieval as span_metric

# Two thematic clusters (Shevchenko / Franko) in one document.
TEXT = (
    "Тарас Шевченко народився в селі Моринці. "
    "Шевченко написав Кобзар. "
    "Іван Франко народився у Нагуєвичах. "
    "Франко написав Мойсей."
)


def _span(sub: str, doc: str = "d1") -> SourceSpan:
    start = TEXT.index(sub)
    return SourceSpan(doc_id=doc, char_start=start, char_end=start + len(sub), text=sub)


def _doc() -> DocRecord:
    return DocRecord(
        doc_id="d1",
        text=TEXT,
        sha256="x",
        n_chars=len(TEXT),
        sections=[Section(title="Поети", char_start=0, char_end=len(TEXT))],
    )


def _extraction() -> DocExtraction:
    return DocExtraction(
        doc_id="d1",
        entities=[
            Entity(
                name="Тарас Шевченко",
                type="PERSON",
                aliases=["Шевченко"],
                mentions=[_span("Тарас Шевченко"), _span("Шевченко")],
            ),
            Entity(name="Кобзар", type="WORK", mentions=[_span("Кобзар")]),
            Entity(name="Моринці", type="LOC", mentions=[_span("Моринці")]),
            Entity(
                name="Іван Франко",
                type="PERSON",
                aliases=["Франко"],
                mentions=[_span("Іван Франко"), _span("Франко")],
            ),
            Entity(name="Мойсей", type="WORK", mentions=[_span("Мойсей")]),
            Entity(name="Нагуєвичах", type="LOC", mentions=[_span("Нагуєвичах")]),
        ],
        facts=[
            SROFact(
                subject="Тарас Шевченко",
                relation="написав",
                object="Кобзар",
                evidence=_span("Шевченко написав Кобзар"),
            ),
            SROFact(
                subject="Тарас Шевченко",
                relation="народився",
                object="Моринці",
                evidence=_span("Тарас Шевченко народився в селі Моринці"),
            ),
            SROFact(
                subject="Іван Франко",
                relation="написав",
                object="Мойсей",
                evidence=_span("Франко написав Мойсей"),
            ),
            SROFact(
                subject="Іван Франко",
                relation="народився",
                object="Нагуєвичах",
                evidence=_span("Іван Франко народився у Нагуєвичах"),
            ),
        ],
    )


def _graph() -> KnowledgeGraph:
    return build_graph([_extraction()], [_doc()])


# --- construction --------------------------------------------------------------------------


def test_build_keeps_offsets_section_and_confidence():
    graph = _graph()
    by_name = {n.name: n for n in graph.nodes}
    shevchenko = by_name["Тарас Шевченко"]
    # mentions keep exact doc_id + offsets + the containing section title
    assert shevchenko.mentions[0]["doc_id"] == "d1"
    assert TEXT[shevchenko.mentions[0]["char_start"] : shevchenko.mentions[0]["char_end"]] == (
        "Тарас Шевченко"
    )
    assert shevchenko.mentions[0]["section_title"] == "Поети"
    # confidence carried from the induced ontology (PERSON appears across the corpus -> > 0)
    assert shevchenko.confidence > 0.0


def test_build_links_facts_to_entity_nodes():
    graph = _graph()
    by_name = {n.name: n.node_id for n in graph.nodes}
    edge = next(
        e for e in graph.edges if e.relation == "написав" and e.src == by_name["Тарас Шевченко"]
    )
    assert edge.dst == by_name["Кобзар"]
    # edge evidence is offset-bearing
    assert (
        TEXT[edge.evidence["char_start"] : edge.evidence["char_end"]] == "Шевченко написав Кобзар"
    )


def test_build_creates_fact_only_node_for_unknown_endpoint():
    doc = DocRecord(
        doc_id="d2",
        text="Альфа діє на Бету.",
        sha256="x",
        n_chars=18,
        sections=[Section(title="t", char_start=0, char_end=18)],
    )
    ext = DocExtraction(
        doc_id="d2",
        entities=[Entity(name="Альфа", type="ORG", mentions=[_span_in("d2", "Альфа", doc.text)])],
        facts=[
            SROFact(
                subject="Альфа",
                relation="діє на",
                object="Бета",
                evidence=_span_in("d2", "Альфа діє на Бету", doc.text),
            )
        ],
    )
    graph = build_graph([ext], [doc])
    names = {n.name for n in graph.nodes}
    assert "Бета" in names  # the unknown object became a fact-only node
    assert len(graph.edges) == 1


def _span_in(doc: str, sub: str, text: str) -> SourceSpan:
    s = text.index(sub)
    return SourceSpan(doc_id=doc, char_start=s, char_end=s + len(sub), text=sub)


# --- community detection -------------------------------------------------------------------


def test_communities_split_two_clusters_deterministically():
    graph = _graph()
    n = assign_communities(graph)
    assert n == 2
    by_name = {node.name: node.community_id for node in graph.nodes}
    assert by_name["Тарас Шевченко"] == by_name["Кобзар"] == by_name["Моринці"]
    assert by_name["Іван Франко"] == by_name["Мойсей"] == by_name["Нагуєвичах"]
    assert by_name["Тарас Шевченко"] != by_name["Іван Франко"]


def test_detect_communities_isolated_node_is_its_own_community():
    # 0-1 connected, 2 isolated
    out = detect_communities({0: {1}, 1: {0}, 2: set()})
    assert out[0] == out[1]
    assert out[2] != out[0]


def test_detect_communities_is_stable_across_calls():
    adj = _graph().adjacency()
    assert detect_communities(adj) == detect_communities(adj)


# --- linking + serialization (pure) --------------------------------------------------------


def test_tokenize_drops_stopwords_and_short_tokens():
    assert tokenize("Хто написав Кобзар?") == {"написав", "кобзар"}


def test_link_seed_nodes_keys_on_name_not_relation_verb():
    graph = _graph()
    assign_communities(graph)
    seeds = link_seed_nodes(graph, "Що написав Шевченко?", 5)
    names = {graph.node_by_id()[s].name for s in seeds}
    assert "Тарас Шевченко" in names
    assert "Іван Франко" not in names  # shared verb "написав" must NOT link Franko


def test_link_communities_picks_the_matching_cluster():
    graph = _graph()
    assign_communities(graph)
    franko_community = next(n.community_id for n in graph.nodes if n.name == "Іван Франко")
    assert link_communities(graph, "Розкажи про Франко", 1) == [franko_community]


def test_morph_key_collapses_inflected_forms_but_keeps_distinct_names_apart():
    # the genitive "Франка" and nominative "Франко" share one stem key
    assert morph_key("франка") == morph_key("франко")
    # unrelated names stay distinct
    assert morph_key("шевченко") != morph_key("франко")
    # short tokens key on themselves (only ever match exactly)
    assert morph_key("ук") == "ук"


def test_morphology_links_inflected_question_form():
    graph = _graph()
    assign_communities(graph)
    # the node name is "Франко"; the question uses the genitive "Франка" -- still links via the stem
    seeds = link_seed_nodes(graph, "Що написав Франка?", 5)
    names = {graph.node_by_id()[s].name for s in seeds}
    assert "Іван Франко" in names
    assert "Тарас Шевченко" not in names  # an unrelated stem must NOT link


def test_exact_token_match_outranks_morphological_match():
    graph = _graph()
    by_name = {n.name: n.node_id for n in graph.nodes}
    # "Шевченка" (genitive, stem match) vs "Кобзар" (exact match) in one question
    scores = node_link_scores(graph, "Що Шевченка написано у Кобзар?")
    assert scores[by_name["Кобзар"]] > scores[by_name["Тарас Шевченко"]]


def test_serialize_subgraph_is_offset_bearing_and_dedups():
    graph = _graph()
    # one member node with two mentions -> two distinct spans, ranked, no duplicates
    relevance = {n.node_id: 1.0 for n in graph.nodes if n.name == "Тарас Шевченко"}
    records = serialize_subgraph(graph, relevance, k=10)
    assert records and all(r["metadata"]["kind"] == KIND_NODE_MENTION for r in records)
    markers = [(r["doc_id"], r["char_start"], r["char_end"]) for r in records]
    assert len(markers) == len(set(markers))  # deduplicated
    assert [r["rank"] for r in records] == list(range(1, len(records) + 1))


def test_serialize_subgraph_empty_members_returns_nothing():
    assert serialize_subgraph(_graph(), {}, k=5) == []


# --- GraphStore (DuckDB engine) ------------------------------------------------------------


def _build_store():
    pytest.importorskip("duckdb")
    from llb.graph.store import GraphStore

    return GraphStore.build([_extraction()], [_doc()])


def test_local_khop_recovers_the_answer_span():
    store = _build_store()
    store.strategy = STRATEGY_LOCAL_KHOP
    hits = store.retrieve("Хто написав Кобзар?", 5)
    # the linked seed (Кобзар) reaches Шевченко via the edge; some hit overlaps the gold answer span
    gold = [{"doc_id": "d1", **_answer_span("Шевченко написав Кобзар")}]
    assert span_metric.recall_at_k(hits, gold, 5) == 1.0
    assert any(h["metadata"]["kind"] == KIND_EDGE_FACT for h in hits)


def test_global_community_serializes_member_spans():
    store = _build_store()
    store.strategy = STRATEGY_GLOBAL_COMMUNITY
    hits = store.retrieve("Розкажи про Франко", 5)
    assert hits
    # every returned chunk is offset-bearing and from Franko's narrative cluster
    assert all(h["text"] == TEXT[h["char_start"] : h["char_end"]] for h in hits)
    assert any("Франко" in h["text"] for h in hits)


def test_khop_depth_bounds_expansion():
    store = _build_store()
    store.strategy = STRATEGY_LOCAL_KHOP
    store.n_seeds = 1
    store.khop_depth = 1
    near = {(h["char_start"], h["char_end"]) for h in store.retrieve("Хто Кобзар?", 20)}
    store.khop_depth = 2
    far = {(h["char_start"], h["char_end"]) for h in store.retrieve("Хто Кобзар?", 20)}
    assert near <= far  # a wider radius never returns fewer spans


def test_unlinked_question_returns_empty():
    store = _build_store()
    for strategy in (STRATEGY_LOCAL_KHOP, STRATEGY_GLOBAL_COMMUNITY):
        store.strategy = strategy
        assert store.retrieve("xyzzy недоречне питання 99", 5) == []


def test_save_load_roundtrip(tmp_path):
    store = _build_store()
    store.save(tmp_path)
    assert (tmp_path / "nodes.jsonl").exists() and (tmp_path / "edges.jsonl").exists()
    from llb.graph.store import GraphStore

    loaded = GraphStore.load(tmp_path, strategy=STRATEGY_LOCAL_KHOP)
    assert loaded.meta["n_nodes"] == store.meta["n_nodes"]
    assert loaded.retrieve("Хто написав Кобзар?", 3)  # queryable after reload


def test_rejects_unknown_strategy():
    pytest.importorskip("duckdb")
    from llb.graph.store import GraphStore

    with pytest.raises(ValueError, match="unknown retrieval strategy"):
        GraphStore.build([_extraction()], [_doc()], strategy="typo")


def _answer_span(sub: str) -> dict[str, int]:
    s = TEXT.index(sub)
    return {"char_start": s, "char_end": s + len(sub)}


# --- diagnostic community summaries (never span-scored) ------------------------------------


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

    from llb.cli.rag import _summarize_graph

    with pytest.raises(typer.Exit) as excinfo:
        _summarize_graph(object(), None)
    assert excinfo.value.exit_code == 2


# --- ingest --------------------------------------------------------------------------------


def test_load_extractions_roundtrip(tmp_path):
    from llb.graph.ingest import load_extractions

    path = tmp_path / "extraction.jsonl"
    path.write_text(_extraction().model_dump_json() + "\n", encoding="utf-8")
    loaded = load_extractions(path)
    assert len(loaded) == 1 and loaded[0].doc_id == "d1"


# --- full vertical through run_eval --------------------------------------------------------


def test_run_eval_with_graph_backend_records_strategy(tmp_path):
    pytest.importorskip("duckdb")
    from llb.backends.base import BackendLauncher, ChatResult
    from llb.config import RunConfig
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
