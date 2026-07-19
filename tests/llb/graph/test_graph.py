"""GraphRAG backend: construction, community detection, the two span-preserving
retrieval strategies, persistence, the tagged-diagnostic summaries, and the full vertical.

The pure pieces (build / community / linking / serialize / ingest) run everywhere. The GraphStore
tests (in `test_graph_retrieval.py`) use the DuckDB k-hop + community engine (the `[graph]`
extra), so they are marked `heavy_env` -- deselected by `make ci-github` in the base
`[dev]`-only GitHub env, run locally via `make ci` / `make test`; `importorskip("duckdb")`
additionally guards partial local installs.
"""

import pytest

from llb.goldset.schema import SourceSpan
from llb.graph.build import build_graph
from llb.graph.model import KnowledgeGraph
from llb.prep.ontology.models import DocExtraction, DocRecord, Entity, Section, SROFact

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


# --- linking + serialization (pure) --------------------------------------------------------


# --- GraphStore (DuckDB engine) ------------------------------------------------------------


def _build_store():
    pytest.importorskip("duckdb")
    from llb.graph.store import GraphStore

    return GraphStore.build([_extraction()], [_doc()])


def _answer_span(sub: str) -> dict[str, int]:
    s = TEXT.index(sub)
    return {"char_start": s, "char_end": s + len(sub)}


# --- diagnostic community summaries (never span-scored) ------------------------------------


# --- ingest --------------------------------------------------------------------------------


# --- full vertical through run_eval --------------------------------------------------------
