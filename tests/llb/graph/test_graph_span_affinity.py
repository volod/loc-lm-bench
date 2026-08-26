"""The per-span affinity tie-break: what it orders, and what it is forbidden to reorder.

Node relevance is a coarse level every mention of a node inherits, so the graph lane's rank-k cut
routinely lands inside a block of exactly-tied spans and is settled by a document id. These tests
pin the two halves of the fix: the affinity term DOES order spans inside one level, and it NEVER
moves a span across one.
"""

from llb.graph.constants import SECTION_TITLE_MATCH_WEIGHT, STEM_MATCH_WEIGHT
from llb.graph.model import GraphMention, GraphNode, KnowledgeGraph
from llb.graph.retrieval import serialize_subgraph
from llb.graph.span_affinity import QuestionKeys, span_affinity


def _mention(text: str, start: int, section: str = "Розділ", doc: str = "d1") -> GraphMention:
    return {
        "doc_id": doc,
        "char_start": start,
        "char_end": start + len(text),
        "text": text,
        "section_title": section,
    }


def _node(node_id: int, name: str, mentions: list[GraphMention]) -> GraphNode:
    return GraphNode(node_id=node_id, name=name, type="MISC", confidence=0.0, mentions=mentions)


def test_affinity_is_the_share_of_the_question_the_span_covers():
    question = QuestionKeys("Шевченко написав Кобзар")  # 3 content tokens
    covered = span_affinity(question, _mention("Шевченко написав Кобзар", 0))
    assert covered == 1.0
    partial = span_affinity(question, _mention("Шевченко", 0, section="Інше"))
    assert partial == 1.0 / 3.0


def test_a_section_title_hit_counts_less_than_the_span_s_own_text():
    question = QuestionKeys("Кобзар")
    in_text = span_affinity(question, _mention("Кобзар", 0, section="Інше"))
    in_title = span_affinity(question, _mention("щось", 0, section="Кобзар"))
    assert in_title == SECTION_TITLE_MATCH_WEIGHT * in_text
    assert 0.0 < in_title < in_text


def test_an_inflected_form_scores_the_stem_weight_not_a_full_hit():
    question = QuestionKeys("Франко")
    inflected = span_affinity(question, _mention("Франкові", 0, section="Інше"))
    assert inflected == STEM_MATCH_WEIGHT


def test_a_question_with_no_content_tokens_scores_every_span_zero():
    question = QuestionKeys("що це")  # stopwords only
    assert span_affinity(question, _mention("Кобзар", 0)) == 0.0


def _one_node_many_mentions() -> KnowledgeGraph:
    """One node whose three mentions differ only in text and section -- an exact-tie block."""
    return KnowledgeGraph(
        nodes=[
            _node(
                0,
                "Франко",
                [
                    _mention("Франко", 0, section="Вступ"),
                    _mention("Франко", 40, section="Мойсей"),
                    _mention("Франко", 80, section="Примітки"),
                ],
            )
        ]
    )


def test_affinity_orders_spans_the_lane_scored_identically():
    graph = _one_node_many_mentions()
    hits = serialize_subgraph(graph, {0: 1.0}, k=3, question="Що таке Мойсей?")
    assert [hit["metadata"]["section_title"] for hit in hits][0] == "Мойсей"
    # the two spans the question says nothing about stay in the lane's stable span order
    assert [hit["char_start"] for hit in hits[1:]] == [0, 80]


def test_an_empty_question_leaves_the_level_exactly_tied():
    graph = _one_node_many_mentions()
    hits = serialize_subgraph(graph, {0: 1.0}, k=3)
    assert {hit["retrieval_score"] for hit in hits} == {1.0}
    assert [hit["char_start"] for hit in hits] == [0, 40, 80]  # doc-order fallback, as before


def _two_levels() -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            _node(0, "Франко", [_mention("Франко", 0, section="Вступ")]),
            _node(1, "Мойсей", [_mention("Мойсей", 40, section="Мойсей")]),
        ]
    )


def test_affinity_never_lifts_a_span_across_a_relevance_level():
    graph = _two_levels()
    # node 1 matches the question on both its text and its section; node 0 matches nothing --
    # yet node 0 sits a whole hop closer, so it must still rank first.
    hits = serialize_subgraph(graph, {0: 1.0, 1: 0.5}, k=2, question="Що таке Мойсей?")
    assert [hit["metadata"]["node"] for hit in hits] == ["Франко", "Мойсей"]
    assert hits[0]["retrieval_score"] > hits[1]["retrieval_score"]


def test_a_refined_score_stays_inside_its_own_level_s_band():
    graph = _two_levels()
    hits = serialize_subgraph(graph, {0: 1.0, 1: 0.5}, k=2, question="Мойсей")
    top, below = hits[0]["retrieval_score"], hits[1]["retrieval_score"]
    assert 0.75 <= top <= 1.0  # level 1.0, banded halfway down to the 0.5 level
    assert 0.25 <= below <= 0.5  # level 0.5, banded halfway down to zero
