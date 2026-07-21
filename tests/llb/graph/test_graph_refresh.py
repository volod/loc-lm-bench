"""GraphRAG incremental refresh: per-doc extraction reuse, deletion propagation, equivalence.

Pure path only (build / diff / persistence) -- no DuckDB queries, so it runs in the base CI
install like the other graph construction tests.
"""

from dataclasses import asdict

import pytest

from llb.goldset.schema import SourceSpan
from llb.graph.refresh import refresh_graph_store, save_graph_inputs
from llb.graph.store import GraphStore
from llb.prep.ontology.inventory import inventory_corpus
from llb.prep.ontology.models import DocExtraction, Entity, SROFact

TS = "20990101T000000Z"

D1_TEXT = "Тарас Шевченко народився в селі Моринці. Шевченко написав Кобзар."
D2_TEXT = "Іван Франко народився у Нагуєвичах. Франко написав Мойсей."
D2_TEXT_V2 = "Іван Франко написав Захара Беркута. Франко жив у Львові."


def _span(doc_id: str, text: str, sub: str) -> SourceSpan:
    start = text.index(sub)
    return SourceSpan(doc_id=doc_id, char_start=start, char_end=start + len(sub), text=sub)


def _extraction_d1() -> DocExtraction:
    return DocExtraction(
        doc_id="d1.md",
        entities=[
            Entity(
                name="Тарас Шевченко",
                type="PERSON",
                mentions=[_span("d1.md", D1_TEXT, "Тарас Шевченко")],
            ),
            Entity(name="Кобзар", type="WORK", mentions=[_span("d1.md", D1_TEXT, "Кобзар")]),
        ],
        facts=[
            SROFact(
                subject="Тарас Шевченко",
                relation="написав",
                object="Кобзар",
                evidence=_span("d1.md", D1_TEXT, "Шевченко написав Кобзар"),
            )
        ],
    )


def _extraction_d2(text: str, work: str) -> DocExtraction:
    return DocExtraction(
        doc_id="d2.md",
        entities=[
            Entity(
                name="Іван Франко", type="PERSON", mentions=[_span("d2.md", text, "Іван Франко")]
            ),
            Entity(name=work, type="WORK", mentions=[_span("d2.md", text, work)]),
        ],
        facts=[
            SROFact(
                subject="Іван Франко",
                relation="написав",
                object=work,
                evidence=_span("d2.md", text, f"Франко написав {work}" if work in text else work),
            )
        ],
    )


def _write_corpus(root, docs):
    root.mkdir(parents=True, exist_ok=True)
    for stale in root.glob("*.md"):
        stale.unlink()
    for name, text in docs.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def _built_graph_dir(tmp_path):
    """v1 graph store persisted with its inputs (the build-graph on-disk contract)."""
    corpus = _write_corpus(tmp_path / "corpus", {"d1.md": D1_TEXT, "d2.md": D2_TEXT})
    extractions = [_extraction_d1(), _extraction_d2(D2_TEXT, "Мойсей")]
    store = GraphStore.build(extractions, inventory_corpus(corpus))
    graph_dir = tmp_path / "graph"
    store.save(graph_dir)
    save_graph_inputs(graph_dir, extractions, None)
    return corpus, graph_dir


def _assert_graph_equivalent(refreshed: GraphStore, rebuilt: GraphStore) -> None:
    assert [asdict(n) for n in refreshed.graph.nodes] == [asdict(n) for n in rebuilt.graph.nodes]
    assert [asdict(e) for e in refreshed.graph.edges] == [asdict(e) for e in rebuilt.graph.edges]
    for key in ("n_nodes", "n_edges", "n_communities", "n_documents", "doc_fingerprints"):
        assert refreshed.meta.get(key) == rebuilt.meta.get(key), key


def test_build_records_doc_fingerprints(tmp_path):
    corpus, graph_dir = _built_graph_dir(tmp_path)
    store = GraphStore.load(graph_dir)
    docs = inventory_corpus(corpus)
    assert store.meta["doc_fingerprints"] == {doc.doc_id: doc.sha256 for doc in docs}


def test_noop_when_corpus_unchanged(tmp_path):
    corpus, graph_dir = _built_graph_dir(tmp_path)
    result = refresh_graph_store(graph_dir, corpus)
    assert result.refreshed is False
    assert not (graph_dir / "generations").exists()


def test_modified_doc_refresh_matches_rebuild(tmp_path):
    corpus, graph_dir = _built_graph_dir(tmp_path)
    _write_corpus(corpus, {"d1.md": D1_TEXT, "d2.md": D2_TEXT_V2})
    update = _extraction_d2(D2_TEXT_V2, "Захара Беркута")
    result = refresh_graph_store(graph_dir, corpus, extraction_update=[update], timestamp=TS)
    assert result.refreshed
    assert result.diff.modified == ["d2.md"] and result.diff.unchanged == ["d1.md"]
    rebuilt = GraphStore.build([_extraction_d1(), update], inventory_corpus(corpus))
    _assert_graph_equivalent(result.store, rebuilt)
    # the refreshed generation reloads and chains: its inputs are persisted beside it
    reloaded = GraphStore.load(graph_dir)
    _assert_graph_equivalent(reloaded, rebuilt)
    assert (result.generation_dir / "extraction.jsonl").is_file()


def test_deletion_propagates_without_extraction_rows(tmp_path):
    corpus, graph_dir = _built_graph_dir(tmp_path)
    _write_corpus(corpus, {"d1.md": D1_TEXT})
    result = refresh_graph_store(graph_dir, corpus, timestamp=TS)
    assert result.refreshed and result.diff.deleted == ["d2.md"]
    refreshed = result.store
    names = {n.name for n in refreshed.graph.nodes}
    assert "Іван Франко" not in names  # retired doc's mentions/nodes are gone
    mentions = [m for n in refreshed.graph.nodes for m in n.mentions]
    assert all(m["doc_id"] == "d1.md" for m in mentions)
    _assert_graph_equivalent(
        refreshed, GraphStore.build([_extraction_d1()], inventory_corpus(corpus))
    )


def test_changed_doc_without_extraction_rows_refuses_with_doc_list(tmp_path):
    corpus, graph_dir = _built_graph_dir(tmp_path)
    _write_corpus(corpus, {"d1.md": D1_TEXT, "d2.md": D2_TEXT_V2})
    with pytest.raises(SystemExit, match="d2.md"):
        refresh_graph_store(graph_dir, corpus, timestamp=TS)


def test_store_without_persisted_inputs_refuses(tmp_path):
    corpus = _write_corpus(tmp_path / "corpus", {"d1.md": D1_TEXT, "d2.md": D2_TEXT})
    extractions = [_extraction_d1(), _extraction_d2(D2_TEXT, "Мойсей")]
    store = GraphStore.build(extractions, inventory_corpus(corpus))
    graph_dir = tmp_path / "graph"
    store.save(graph_dir)  # legacy store: no extraction.jsonl beside it
    _write_corpus(corpus, {"d1.md": D1_TEXT})
    with pytest.raises(SystemExit, match="extraction.jsonl"):
        refresh_graph_store(graph_dir, corpus, timestamp=TS)
