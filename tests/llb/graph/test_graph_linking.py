"""Tests for graph linking."""

from llb.graph.community import assign_communities, detect_communities
from llb.graph.constants import (
    KIND_NODE_MENTION,
)
from llb.graph.linking import (
    link_communities,
    link_seed_nodes,
    morph_key,
    node_link_scores,
    tokenize,
)
from llb.graph.retrieval import serialize_subgraph
from test_graph import _graph


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
