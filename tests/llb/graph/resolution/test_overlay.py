"""The node-cluster overlay: what it proposes, what applying it builds, and what it never does."""

import pytest

from llb.graph.linking import link_seed_nodes
from llb.graph.model import GraphEdge, GraphNode, KnowledgeGraph
from llb.graph.resolution.overlay import (
    NodeCluster,
    NodeOverlay,
    apply_overlay,
    overlay_from_clusters,
    read_overlay,
    write_overlay,
)
from llb.graph.resolution.records import record_id
from llb.linkage.model import LinkageCluster


def _mention(doc_id: str, start: int, text: str):
    return {
        "doc_id": doc_id,
        "char_start": start,
        "char_end": start + len(text),
        "text": text,
        "section_title": "s",
    }


def _graph() -> KnowledgeGraph:
    """Three nodes of one entity, one unrelated node, and a fact between two of them."""
    nodes = [
        GraphNode(0, "Іван Франко", "PERSON", 0.9, ["Франко"], [_mention("d", 0, "Іван Франко")]),
        GraphNode(
            1,
            "Франка",
            "PERSON",
            0.9,
            ["Іван Франко"],
            [_mention("d", 20, "Франка"), _mention("d", 40, "Франка")],
        ),
        GraphNode(2, "Каменяр", "PERSON", 0.5, [], [_mention("e", 5, "Каменяр")]),
        GraphNode(3, "Львів", "LOC", 0.9, [], [_mention("d", 60, "Львів")]),
    ]
    edges = [
        GraphEdge(0, 0, 3, "жив_у", _mention("d", 80, "Франко жив у Львові")),
        GraphEdge(1, 1, 3, "жив_у", _mention("d", 80, "Франко жив у Львові")),
        GraphEdge(2, 0, 1, "те_саме", _mention("d", 100, "Франко і Франка")),
    ]
    return KnowledgeGraph(nodes=nodes, edges=edges)


def _overlay(threshold: float = 0.9) -> NodeOverlay:
    return NodeOverlay(
        threshold=threshold, clusters=(NodeCluster(canonical_id=1, member_ids=(0, 1, 2)),)
    )


def test_the_canonical_is_the_most_grounded_member():
    graph = _graph()
    clusters = (
        LinkageCluster("c", tuple(record_id(node_id) for node_id in (0, 1, 2))),
        LinkageCluster("d", (record_id(3),)),
    )
    overlay = overlay_from_clusters(clusters, graph, 0.9)
    assert [cluster.canonical_id for cluster in overlay.clusters] == [1]
    assert overlay.clusters[0].member_ids == (0, 1, 2)


def test_singleton_clusters_are_not_carried():
    graph = _graph()
    overlay = overlay_from_clusters(
        tuple(LinkageCluster(str(n), (record_id(n),)) for n in range(4)), graph, 0.9
    )
    assert overlay.clusters == ()
    assert overlay.summary() == {
        "threshold": 0.9,
        "n_clusters": 0,
        "n_nodes_merged": 0,
        "largest_cluster": 0,
    }


def test_applying_an_overlay_unions_surface_forms_and_mentions():
    merged = apply_overlay(_graph(), _overlay())
    by_id = merged.node_by_id()
    assert sorted(by_id) == [1, 3]
    canonical = by_id[1]
    assert canonical.name == "Франка"
    assert set(canonical.aliases) == {"Іван Франко", "Франко", "Каменяр"}
    assert len(canonical.mentions) == 4


def test_applying_an_overlay_remaps_edges_and_drops_the_duplicate_fact():
    merged = apply_overlay(_graph(), _overlay())
    facts = {(edge.src, edge.dst, edge.relation) for edge in merged.edges}
    # The two `жив_у` edges cited the same span from two fragments of one entity: after the merge
    # they are one fact, while the self-loop keeps its own distinct evidence span.
    assert facts == {(1, 3, "жив_у"), (1, 1, "те_саме")}
    assert [edge.edge_id for edge in merged.edges] == list(range(len(merged.edges)))


def test_the_source_graph_is_never_mutated():
    graph = _graph()
    apply_overlay(graph, _overlay())
    assert [node.node_id for node in graph.nodes] == [0, 1, 2, 3]
    assert graph.nodes[1].aliases == ["Іван Франко"]
    assert len(graph.edges) == 3


def test_communities_are_recomputed_over_the_merged_adjacency():
    merged = apply_overlay(_graph(), _overlay())
    assert {node.community_id for node in merged.nodes} == {0}


def test_the_merged_node_links_on_a_form_only_a_fragment_carried():
    """The recall mechanism the whole pass is measured on, stated as an invariant."""
    graph = _graph()
    # Before the merge the epithet is its own node, so a question naming only that form reaches
    # the one mention that node holds and none of the entity's others.
    before = link_seed_nodes(graph, "Каменяр", 1)
    assert before == [2]
    assert len(graph.node_by_id()[before[0]].mentions) == 1
    merged = apply_overlay(graph, _overlay())
    after = link_seed_nodes(merged, "Каменяр", 1)
    assert after == [1]
    assert len(merged.node_by_id()[after[0]].mentions) == 4


def test_an_overlay_round_trips_through_its_written_artifact(tmp_path):
    graph = _graph()
    path = tmp_path / "overlay.jsonl"
    write_overlay(_overlay(0.75), path, graph)
    recovered = read_overlay(path)
    assert recovered == _overlay(0.75)
    assert "Франка" in path.read_text(encoding="utf-8")


def test_a_written_overlay_without_a_header_is_refused(tmp_path):
    path = tmp_path / "overlay.jsonl"
    path.write_text('{"kind": "cluster", "canonical_id": 1, "member_ids": [1, 2]}\n', "utf-8")
    with pytest.raises(ValueError, match="no overlay header"):
        read_overlay(path)
