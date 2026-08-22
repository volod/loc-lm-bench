"""The whole pass over a planted graph whose correct clustering is known (needs the extra)."""

import json

import pytest

from llb.graph.model import KnowledgeGraph
from llb.graph.resolution.compare import LaneItems, base_lane, overlay_lane
from llb.graph.resolution.constants import (
    COMPARISON_FILE,
    MAX_RESOLUTION_NODES,
    MIN_RESOLUTION_NODES,
    OVERLAYS_DIR,
    PRE_MERGE_DIR,
    RECORDS_FILE,
    REPORT_FILE,
    SUMMARY_FILE,
)
from llb.graph.resolution.overlay import read_overlay
from llb.graph.resolution.run import declined_reason, resolve_graph_entities
from llb.graph.resolution.verdict import DECISION_NEGATIVE

pytestmark = pytest.mark.heavy_env

# The cut the planted fixture is recovered EXACTLY at. The three-form entities separate far above
# it; an initialism against its spelled-out form ("ЗСУ" against "Збройні Сили України") agrees on
# no name similarity at all, only on a surface form, the type, the document, and what its mentions
# are about -- and that combination is what the fitted model prices down here.
RECOVERY_THRESHOLD = 0.3
GRID = (0.99, 0.9, RECOVERY_THRESHOLD)


def _extra():
    pytest.importorskip("splink")
    pytest.importorskip("duckdb")


def _lane_items(planted) -> LaneItems:
    """One question per fragmented entity, asked by ONE form and labelled on the OTHER pieces.

    That is the retrieval question fragmentation actually breaks: the asked form is on a node of
    its own, and the evidence the answer needs sits on the sibling nodes nothing links it to. The
    morphological seed linker already bridges the pieces that share a stem, so the pre-overlay
    lane scores those and misses the abbreviations -- which is exactly the headroom a merge takes.
    """
    questions = []
    item_ids = []
    for index, (_entity, node_ids) in enumerate(sorted(planted.fragmented_groups.items())):
        nodes = [planted.graph.node_by_id()[node_id] for node_id in sorted(node_ids)]
        asked = nodes[-1]
        spans = [
            {
                "doc_id": mention["doc_id"],
                "char_start": mention["char_start"],
                "char_end": mention["char_end"],
                "text": mention["text"],
            }
            for node in nodes[:-1]
            for mention in node.mentions
        ]
        questions.append((f"Що відомо про {asked.name}?", spans))
        item_ids.append(f"item-{index}")
    return LaneItems(items=questions, item_ids=item_ids)


def _run(planted, node_embedder, out_dir, **kwargs):
    return resolve_graph_entities(
        planted.graph,
        _lane_items(planted),
        out_dir,
        k=kwargs.pop("k", 10),
        strategies=kwargs.pop("strategies", ("local_khop",)),
        khop_depth=2,
        graph_meta={"backend": "graph"},
        thresholds=kwargs.pop("thresholds", GRID),
        embedder=node_embedder,
        resamples=kwargs.pop("resamples", 200),
        **kwargs,
    )


def test_the_fit_recovers_the_planted_clustering_exactly_at_one_cut(
    planted, node_embedder, tmp_path
):
    _extra()
    published = _run(planted, node_embedder, tmp_path / "run")
    at_recovery = next(
        overlay
        for overlay in published.overlays
        if overlay.threshold == pytest.approx(RECOVERY_THRESHOLD)
    )
    found = {frozenset(cluster.member_ids) for cluster in at_recovery.clusters}
    truth = {frozenset(ids) for ids in planted.fragmented_groups.values()}
    assert found == truth


def test_no_cross_entity_merge_is_proposed_at_any_priced_cut(planted, node_embedder, tmp_path):
    _extra()
    published = _run(planted, node_embedder, tmp_path / "run")
    for overlay in published.overlays:
        for cluster in overlay.clusters:
            assert len({planted.truth[member] for member in cluster.member_ids}) == 1


def test_a_tighter_cut_never_merges_more_than_a_looser_one(planted, node_embedder, tmp_path):
    _extra()
    published = _run(planted, node_embedder, tmp_path / "run")
    merged = [overlay.n_nodes_merged for overlay in published.overlays]
    assert merged == sorted(merged)  # overlays are ordered tightest cut first


def test_the_paired_rerun_scores_the_same_items_under_every_lane(planted, node_embedder, tmp_path):
    _extra()
    published = _run(planted, node_embedder, tmp_path / "run")
    report = published.reports["local_khop"]
    assert set(report["backends"]) == {base_lane("local_khop")} | {
        overlay_lane("local_khop", threshold) for threshold in GRID
    }
    assert {row["n"] for row in report["backends"].values()} == {report["n"]}
    assert report["uncertainty"]["baseline"] == base_lane("local_khop")


def test_the_recovered_overlay_costs_the_graph_lane_nothing(planted, node_embedder, tmp_path):
    """A correct merge must not cost recall or rank -- that is the floor a reading is read above.

    It is deliberately not asserted to GAIN: the graph's own seed linker already keys on a node's
    aliases and on a Ukrainian stem, so it bridges most of what this plant fragments, and a fixture
    demanding a lift would be demanding the linker be worse than it is. What a merge buys where the
    linker cannot reach is asserted directly on the mechanism in `test_overlay.py`.
    """
    _extra()
    published = _run(planted, node_embedder, tmp_path / "run")
    backends = published.reports["local_khop"]["backends"]
    before = backends[base_lane("local_khop")]
    after = backends[overlay_lane("local_khop", RECOVERY_THRESHOLD)]
    assert after["recall_at_k"] >= before["recall_at_k"]
    assert after["mrr"] >= before["mrr"]


def test_a_merged_node_carries_every_fragment_mention_the_overlay_named(
    planted, node_embedder, tmp_path
):
    _extra()
    from llb.graph.resolution.overlay import apply_overlay

    published = _run(planted, node_embedder, tmp_path / "run")
    overlay = next(
        o for o in published.overlays if o.threshold == pytest.approx(RECOVERY_THRESHOLD)
    )
    merged = apply_overlay(planted.graph, overlay).node_by_id()
    source = planted.graph.node_by_id()
    for cluster in overlay.clusters:
        expected = sum(len(source[member].mentions) for member in cluster.member_ids)
        assert len(merged[cluster.canonical_id].mentions) == expected


def test_the_bundle_holds_every_documented_artifact(planted, node_embedder, tmp_path):
    _extra()
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    (graph_dir / "graph_meta.json").write_text('{"backend": "graph"}', encoding="utf-8")
    out_dir = tmp_path / "run"
    published = _run(planted, node_embedder, out_dir, graph_dir=graph_dir)
    assert (out_dir / "linkage" / "pairs.jsonl").exists()
    assert len((out_dir / RECORDS_FILE).read_text(encoding="utf-8").splitlines()) == len(
        planted.graph.nodes
    )
    for threshold in GRID:
        assert read_overlay(
            out_dir / OVERLAYS_DIR / f"overlay_{threshold:g}.jsonl"
        ).threshold == pytest.approx(threshold)
    assert set(json.loads((out_dir / COMPARISON_FILE).read_text("utf-8"))["strategies"]) == {
        "local_khop"
    }
    assert json.loads((out_dir / SUMMARY_FILE).read_text("utf-8")) == published.summary
    assert (out_dir / REPORT_FILE).read_text("utf-8").startswith("# Graph entity node resolution")
    assert (out_dir / PRE_MERGE_DIR / "graph_meta.json").exists()


def test_a_flat_reading_is_recorded_as_the_negative_result_it_is(planted, node_embedder, tmp_path):
    """The outcome this fixture actually produces, asserted rather than left to the reader.

    The planted clustering is recovered exactly and the lane does not move, because the seed
    linker already bridged most of it. That is a negative result about the OVERLAY, not a failure
    of the run, and it has to arrive labelled as one.
    """
    _extra()
    published = _run(planted, node_embedder, tmp_path / "run")
    verdict = published.summary["verdict"]
    assert verdict["decision"] == DECISION_NEGATIVE
    assert verdict["lane"] is None
    assert "not adopted" in verdict["note"]


def test_both_strategies_are_rerun_when_both_are_asked_for(planted, node_embedder, tmp_path):
    _extra()
    published = _run(
        planted,
        node_embedder,
        tmp_path / "run",
        strategies=("local_khop", "global_community"),
        thresholds=(RECOVERY_THRESHOLD,),
    )
    assert set(published.reports) == {"local_khop", "global_community"}
    assert set(published.summary["thresholds"][0]["strategies"]) == {
        "local_khop",
        "global_community",
    }


def test_a_graph_below_the_fit_floor_declines_with_its_reason(planted, node_embedder, tmp_path):
    small = KnowledgeGraph(nodes=planted.graph.nodes[:3])
    reason = declined_reason(small)
    assert reason is not None and str(MIN_RESOLUTION_NODES) in reason
    published = resolve_graph_entities(
        small,
        _lane_items(planted),
        tmp_path / "declined",
        k=10,
        strategies=("local_khop",),
        khop_depth=2,
        graph_meta={},
        embedder=node_embedder,
    )
    assert published.declined
    assert published.summary["reason"] == reason
    assert (tmp_path / "declined" / SUMMARY_FILE).exists()


def test_a_graph_above_the_blocking_cap_declines_with_its_reason(planted):
    oversized = KnowledgeGraph(nodes=planted.graph.nodes * (MAX_RESOLUTION_NODES + 1))
    reason = declined_reason(oversized)
    assert reason is not None and str(MAX_RESOLUTION_NODES) in reason
