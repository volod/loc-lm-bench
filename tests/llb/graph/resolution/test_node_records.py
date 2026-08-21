"""The graph-node record table and its comparison specification (base install, no Splink)."""

import pytest

from llb.graph.model import GraphNode, KnowledgeGraph
from llb.graph.resolution.constants import (
    ENTITY_TYPE_COLUMN,
    HEAD_KEY_COLUMN,
    MAX_NODE_TEXT_CHARS,
    MENTION_VECTOR_COLUMN,
    NODE_ID_COLUMN,
    TAIL_KEY_COLUMN,
)
from llb.graph.resolution.records import (
    blocking_keys,
    build_node_spec,
    build_records,
    doc_ids,
    node_id_of,
    node_text,
    record_id,
    surface_forms,
)
from llb.linkage.constants import RESERVED_COLUMNS


def _node(node_id: int, name: str, **kwargs) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        name=name,
        type=kwargs.get("etype", "PERSON"),
        confidence=0.5,
        aliases=list(kwargs.get("aliases", ())),
        mentions=[
            {
                "doc_id": doc,
                "char_start": start,
                "char_end": start + 4,
                "text": text,
                "section_title": "s",
            }
            for doc, start, text in kwargs.get("mentions", ())
        ],
    )


def test_record_id_round_trips():
    assert node_id_of(record_id(42)) == 42


def test_surface_forms_carry_the_name_normalized_and_deduplicated():
    node = _node(0, "  Іван   Франко ", aliases=["іван франко", "Франко"])
    assert surface_forms(node) == sorted({"іван франко", "франко"})


def test_doc_ids_are_the_deduplicated_documents_the_mentions_cite():
    node = _node(0, "X", mentions=(("doc-b", 0, "a"), ("doc-a", 9, "b"), ("doc-b", 4, "c")))
    assert doc_ids(node) == ["doc-a", "doc-b"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Іван Франко", ("іван", "фран")),
        ("Франко", ("фран", "фран")),
        ("Франка", ("фран", "фран")),
        ("ЗСУ", ("зсу", "зсу")),
        ("   ", ("", "")),
    ],
)
def test_blocking_keys_are_the_head_and_tail_morphological_stems(name, expected):
    assert blocking_keys(_node(0, name)) == expected


def test_node_text_is_capped_at_the_embedding_budget():
    node = _node(0, "X" * 100, mentions=(("d", 0, "Y" * 900),))
    assert len(node_text(node)) == MAX_NODE_TEXT_CHARS


def test_records_align_with_the_node_list_and_carry_the_vector():
    graph = KnowledgeGraph(nodes=[_node(0, "A"), _node(1, "B")])
    records = build_records(graph, [[0.1, 0.2], [0.3, 0.4]])
    assert [record[NODE_ID_COLUMN] for record in records] == ["0", "1"]
    assert records[1][MENTION_VECTOR_COLUMN] == [0.3, 0.4]


def test_records_omit_the_vector_column_when_no_embedder_ran():
    graph = KnowledgeGraph(nodes=[_node(0, "A")])
    assert MENTION_VECTOR_COLUMN not in build_records(graph, None)[0]


def test_misaligned_vectors_are_refused_before_any_table_is_built():
    graph = KnowledgeGraph(nodes=[_node(0, "A"), _node(1, "B")])
    with pytest.raises(ValueError, match="align one-to-one"):
        build_records(graph, [[0.1]])


def test_the_specification_drops_the_cosine_comparison_without_an_embedder():
    without = build_node_spec(0, 0.9)
    with_vectors = build_node_spec(8, 0.9)
    assert MENTION_VECTOR_COLUMN not in without.compared_columns
    assert MENTION_VECTOR_COLUMN in with_vectors.compared_columns
    assert len(with_vectors.comparisons) == len(without.comparisons) + 1


def test_the_specification_blocks_on_both_stems_and_the_entity_type():
    spec = build_node_spec(8, 0.9)
    assert [rule.label for rule in spec.blocking_rules] == [
        TAIL_KEY_COLUMN,
        HEAD_KEY_COLUMN,
        ENTITY_TYPE_COLUMN,
    ]


def test_the_entity_type_pass_trains_the_name_and_the_stem_pass_trains_the_type():
    spec = build_node_spec(8, 0.9)
    assert [rule.label for rule in spec.em_rules] == [ENTITY_TYPE_COLUMN, TAIL_KEY_COLUMN]


def test_no_record_column_collides_with_a_name_the_clustering_step_reserves():
    graph = KnowledgeGraph(nodes=[_node(0, "A")])
    columns = set(build_records(graph, [[0.1]])[0])
    assert not columns & set(RESERVED_COLUMNS)
