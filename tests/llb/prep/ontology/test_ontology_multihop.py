"""Tests for ontology multihop."""

import json
from llb.goldset.chains import validate_chains
from llb.goldset.validate import validate_items
from llb.graph.model import GraphEdge, GraphMention, GraphNode, KnowledgeGraph
from llb.prep.ontology.constants import (
    QUESTION_TYPE_MULTI_HOP,
)
from llb.prep.ontology.chains import build_chain_items
from llb.prep.ontology.graph_paths import walk_chain_paths, walk_two_hop_paths
from llb.prep.ontology.models import (
    DocRecord,
)
from llb.prep.ontology.multi_hop import build_multi_hop_items, draft_multi_hop
from ontology_yield_helpers import CHAIN_DOC, _chain_graph


def test_walk_two_hop_paths_builds_a_distinct_span_chain():
    seeds = walk_two_hop_paths(_chain_graph(), max_paths=10)
    assert len(seeds) == 1
    seed = seeds[0]
    assert seed.start == "Alpha" and seed.bridge == "Beta" and seed.end == "Gamma"
    assert [step.relation for step in seed.steps] == ["керує", "належить"]
    # the two hops cite DISTINCT spans -> a genuine multi-span question
    keys = {(s.evidence.char_start, s.evidence.char_end) for s in seed.steps}
    assert len(keys) == 2


def test_walk_two_hop_paths_skips_when_no_bridge_node():
    # two edges that do not share a middle node -> no 2-hop path
    m: GraphMention = {
        "doc_id": "d.md",
        "char_start": 0,
        "char_end": 3,
        "text": "abc",
        "section_title": "S",
    }
    graph = KnowledgeGraph(
        nodes=[GraphNode(node_id=i, name=f"n{i}", type="X", confidence=0.0) for i in range(4)],
        edges=[
            GraphEdge(edge_id=0, src=0, dst=1, relation="r", evidence=m),
            GraphEdge(edge_id=1, src=2, dst=3, relation="r", evidence=m),
        ],
    )
    assert walk_two_hop_paths(graph, max_paths=10) == []


def test_walk_chain_paths_fills_from_shared_topic_facts():
    graph = _chain_graph()
    graph.edges[1] = GraphEdge(
        edge_id=1,
        src=0,
        dst=2,
        relation="підтримує",
        evidence=graph.edges[1].evidence,
    )

    assert walk_two_hop_paths(graph, max_paths=10) == []
    seeds = walk_chain_paths(graph, max_paths=10)
    assert len(seeds) == 1
    assert seeds[0].bridge == "Alpha"
    assert [step.relation for step in seeds[0].steps] == ["керує", "підтримує"]


def test_build_multi_hop_items_carries_two_spans_and_validates(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "chain.md").write_text(CHAIN_DOC, encoding="utf-8")
    docs = [DocRecord(doc_id="chain.md", text=CHAIN_DOC, sha256="x", n_chars=len(CHAIN_DOC))]
    seeds = walk_two_hop_paths(_chain_graph(), max_paths=10)

    drafts = draft_multi_hop(
        lambda _p: json.dumps(
            {
                "question": "Кому належить компанія, якою керує Alpha?",
                "reference_answer": "Кінцевою організацією є Gamma.",
            }
        ),
        docs,
        seeds,
    )
    items, labels = build_multi_hop_items(docs, seeds, drafts)

    assert len(items) == 1
    item = items[0]
    assert len(item.source_spans) >= 2  # >= 2 grounded spans
    assert labels[item.id].question_type == QUESTION_TYPE_MULTI_HOP
    assert labels[item.id].difficulty == "hard"
    # span-exact validation against the copied corpus passes
    report = validate_items(items, corpus)
    assert report["errors"] == []


def test_build_multi_hop_items_drops_draft_without_question():
    docs = [DocRecord(doc_id="chain.md", text=CHAIN_DOC, sha256="x", n_chars=len(CHAIN_DOC))]
    seeds = walk_two_hop_paths(_chain_graph(), max_paths=10)
    items, labels = build_multi_hop_items(docs, seeds, [{"reference_answer": "Gamma"}])
    assert items == [] and labels == {}


def test_build_multi_hop_items_drops_reference_without_bridge_or_end_entity():
    docs = [DocRecord(doc_id="chain.md", text=CHAIN_DOC, sha256="x", n_chars=len(CHAIN_DOC))]
    seeds = walk_two_hop_paths(_chain_graph(), max_paths=10)
    drafts = [
        {
            "question": "Кому належить компанія, якою керує Alpha?",
            "reference_answer": "Відповідь не називає сутність із ланцюжка.",
        }
    ]

    items, labels = build_multi_hop_items(docs, seeds, drafts)

    assert items == [] and labels == {}


def test_build_multi_hop_items_accepts_reference_containing_bridge_entity():
    docs = [DocRecord(doc_id="chain.md", text=CHAIN_DOC, sha256="x", n_chars=len(CHAIN_DOC))]
    seeds = walk_two_hop_paths(_chain_graph(), max_paths=10)
    drafts = [
        {
            "question": "Яка проміжна організація поєднує Alpha та Gamma?",
            "reference_answer": "Проміжною організацією є Beta.",
        }
    ]

    items, labels = build_multi_hop_items(docs, seeds, drafts)

    assert len(items) == 1
    assert items[0].reference_answer == "Проміжною організацією є Beta."
    assert labels[items[0].id].question_type == QUESTION_TYPE_MULTI_HOP


def test_build_chain_items_carries_ordered_grounded_steps(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "chain.md").write_text(CHAIN_DOC, encoding="utf-8")
    docs = [DocRecord(doc_id="chain.md", text=CHAIN_DOC, sha256="x", n_chars=len(CHAIN_DOC))]
    seeds = walk_two_hop_paths(_chain_graph(), max_paths=10)

    chains = build_chain_items(docs, seeds)

    assert len(chains) == 1
    chain = chains[0]
    assert [step.order for step in chain.steps] == [1, 2]
    assert chain.steps[0].dependency_note == ""
    assert chain.steps[1].dependency_note
    assert chain.steps[0].question.startswith("Який факт")
    assert chain.steps[1].question.startswith("З урахуванням")
    assert validate_chains(chains, corpus)["errors"] == []


def test_multi_hop_stage_bridge_fill_recovers_a_sparse_graph():
    """Extracted graphs rarely have directed object-to-subject links, so the strict walk can
    yield nothing at all; `bridge_fill` falls back to shared-bridge fact pairs, which still
    cite two distinct spans -- the >= 2-span retrieval problem a multi-hop slice measures."""
    from llb.prep.ontology.pipeline.stages import _multi_hop_stage

    graph = _chain_graph()
    graph.edges[1] = GraphEdge(
        edge_id=1, src=0, dst=2, relation="підтримує", evidence=graph.edges[1].evidence
    )
    docs = [DocRecord(doc_id="chain.md", text=CHAIN_DOC, sha256="x", n_chars=len(CHAIN_DOC))]

    def complete(_prompt: str) -> str:
        return json.dumps(
            {
                "question": "Яка організація поєднує обидві згадані компанії?",
                "reference_answer": "Їх поєднує Alpha.",
            }
        )

    def _stage(bridge_fill: bool):
        return _multi_hop_stage(
            complete,
            docs,
            [],
            None,
            graph_dir=None,
            max_paths=10,
            seed=13,
            bridge_fill=bridge_fill,
            graph=graph,
        )

    assert _stage(False) == ([], {})  # strict directed walk finds no path in this graph
    items, labels = _stage(True)
    assert len(items) == 1
    assert len(items[0].source_spans) >= 2
    assert labels[items[0].id].question_type == QUESTION_TYPE_MULTI_HOP
