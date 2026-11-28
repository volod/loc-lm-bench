"""Pin the structural graph and its paired reading against the recovered synthetic book."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llb.graph.book_inputs import load_book_sources
from llb.graph.book_store import BOOK_GRAPH_DIR, BOOK_KHOP_DEPTH, build_book_store
from llb.graph.store import GraphStore
from llb.main import app
from llb.rag.vector_store import factory, persistence
from tests.llb.rag.book_graph_fixture import (
    CITING,
    CUTOFF,
    GRID,
    MDE,
    N_PER_SLICE,
    RESAMPLES,
    BookGraphEmbedder,
    write_graph_inputs,
)


@pytest.fixture(scope="module")
def book(tmp_path_factory):
    return write_graph_inputs(tmp_path_factory.mktemp("book-reference-graph"))


def test_graph_bytes_and_printed_evidence_are_exact(book, tmp_path):
    projection, _, _ = book
    first, store = build_book_store(projection, tmp_path, khop_depth=BOOK_KHOP_DEPTH)
    second, _ = build_book_store(projection, tmp_path, khop_depth=BOOK_KHOP_DEPTH)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }
    sources = load_book_sources(projection)
    graph = store.graph
    assert {node.type for node in graph.nodes} == {
        "section",
        "page",
        "table",
        "equation",
        "figure",
        "index-term",
    }
    assert {edge.relation for edge in graph.edges} == {"contains", "cites", "indexes"}
    _assert_grounding(graph, sources)
    _assert_printed_relations(graph, sources)


def _assert_grounding(graph, sources):
    for evidence in [edge.evidence for edge in graph.edges] + [
        mention for node in graph.nodes for mention in node.mentions
    ]:
        doc = sources.documents[evidence["doc_id"]]
        assert doc.source.text[evidence["char_start"] : evidence["char_end"]] == evidence["text"]
        assert evidence["text"] and evidence["text"] != "# Book"


def _assert_printed_relations(graph, sources):
    resolved, unresolved = _cues(sources)
    by_id = graph.node_by_id()
    cited = [edge for edge in graph.edges if edge.relation == "cites"]
    assert cited and all(_cue_key(edge.evidence) in resolved for edge in cited)
    assert unresolved
    assert all(_cue_key(edge.evidence) not in unresolved for edge in graph.edges)
    table = next(edge for edge in cited if by_id[edge.dst].type == "table")
    assert table.evidence["text"] == "2-1"
    assert by_id[table.src].type == "section"
    assert any(CITING in mention["text"] for mention in by_id[table.src].mentions)
    assert any(GRID in mention["text"] for mention in by_id[table.dst].mentions)
    assert any(
        node.type == "equation" and any("E = mc^2" in m["text"] for m in node.mentions)
        for node in graph.nodes
    )
    assert any(
        edge.relation == "indexes"
        and by_id[edge.src].name == "transfer"
        and edge.evidence["text"] == "11"
        for edge in graph.edges
    )


def _cues(sources):
    resolved, unresolved = set(), set()
    for document in sources.documents.values():
        for row in document.references.citations:
            cues = resolved if row.target else unresolved
            cues.add((document.source.doc_id, row.cue_start, row.cue_end))
    return resolved, unresolved


def _cue_key(evidence):
    return evidence["doc_id"], evidence["char_start"], evidence["char_end"]


@pytest.mark.heavy_env
def test_book_graph_cli_compares_fixed_lanes_and_replays(book, monkeypatch, tmp_path):
    projection, goldset, config = book
    monkeypatch.setattr(factory, "Embedder", BookGraphEmbedder)
    monkeypatch.setattr(persistence, "Embedder", BookGraphEmbedder)
    runner = CliRunner()
    cfg = ["--config", str(config)]
    built = runner.invoke(app, ["build-graph", *cfg, "--book-corpus", str(projection)])
    assert built.exit_code == 0, built.output
    graph_path = Path(built.output.strip().rsplit(" -> ", 1)[-1])
    assert graph_path.parent.parent.name == BOOK_GRAPH_DIR
    _assert_one_hop(graph_path)
    indexed = runner.invoke(app, ["build-index", *cfg])
    assert indexed.exit_code == 0, indexed.output
    args = [
        "compare-graph-fusion",
        *cfg,
        "--book-corpus",
        str(projection),
        "--goldset",
        str(goldset),
        "--minimum-detectable-delta",
        str(MDE),
        "--k",
        str(CUTOFF),
        "--resamples",
        str(RESAMPLES),
        "--noise-floor",
    ]
    path = _report(runner, args)
    report = json.loads(path.read_bytes())
    _assert_reading(report)
    assert path.read_bytes() == _report(runner, args).read_bytes()
    _refusals(runner, args, projection, goldset, tmp_path)


def _assert_one_hop(graph_path: Path) -> None:
    graph = GraphStore.load(graph_path)
    zero = GraphStore.load(graph_path, khop_depth=0)
    question = "Where is ignition documented?"
    cutoff = 12
    assert graph.khop_depth == 1
    assert not any(GRID in hit["text"] for hit in zero.retrieve(question, cutoff))
    assert GRID in graph.retrieve(question, cutoff)[-1]["text"]
    assert any(hit["text"] == "transfer" for hit in graph.retrieve("transfer", CUTOFF))


def _report(runner, args):
    result = runner.invoke(app, args)
    assert result.exit_code == 0, (result.output, result.exception)
    return Path(result.output.strip().rsplit(" -> ", 1)[-1])


def _refusals(runner, args, projection, goldset, tmp_path):
    sweep = runner.invoke(app, [*args, "--graph-weights", "0.1,0.2"])
    assert sweep.exit_code == 2 and "fixed book reading" in sweep.output
    rejected = tmp_path / "unverified.jsonl"
    rows = goldset.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    rows[0] = json.dumps({**row, "verified": False})
    rejected.write_text("\n".join(rows) + "\n", encoding="utf-8")
    refused = runner.invoke(app, [*args, "--goldset", str(rejected)])
    assert refused.exit_code == 2 and "unverified" in refused.output
    first = next(projection.glob("*.references.json"))
    original = first.read_bytes()
    try:
        first.write_bytes(original + b"\n")
        refused = runner.invoke(app, args)
        assert refused.exit_code == 2 and "stale book graph" in refused.output
    finally:
        first.write_bytes(original)


def _assert_reading(report):
    settings = report["settings"]
    assert settings["khop_depth"] == 1
    assert settings["k"] == CUTOFF
    assert settings["ppr"] == "unavailable"
    assert settings["minimum_detectable_delta"] == MDE
    assert settings["minimum_discordant_pairs"] == N_PER_SLICE
    all_items, technical, text = (
        report["slices"][key] for key in ("all", "technical_evidence", "text_only")
    )
    assert (all_items["n"], technical["n"], text["n"]) == (12, 6, 6)
    assert {tuple(item["technical_kinds"]) for item in report["items"]} == {("table",), ()}
    assert all("noise_floor" in group for group in (all_items, technical, text))
    rows = technical["rows"]
    assert set(rows) == {"vector", "graph/local_khop", f"fused/local_khop@0.75/d{CUTOFF}"}
    for label in rows.keys() - {"vector"}:
        metric = rows[label]["all_spans_at_k"]
        assert metric["paired"]["delta"] == {"mean": 1.0, "lo": 1.0, "hi": 1.0}
        assert metric["paired"]["wins"] == N_PER_SLICE
        assert metric["verdict"] == "improved"
        assert metric["sensitivity"]["target_reached"]
        null = text["rows"][label]["all_spans_at_k"]
        assert null["paired"]["delta"] == {"mean": 0.0, "lo": 0.0, "hi": 0.0}
        assert null["paired"]["ties"] == N_PER_SLICE
        assert null["verdict"] == "inconclusive"
        assert not null["sensitivity"]["target_reached"]


def test_graph_refuses_a_target_that_is_not_a_printed_block(tmp_path):
    projection, _, _ = write_graph_inputs(tmp_path)
    for path in projection.glob("*.references.json"):
        refs = json.loads(path.read_bytes())
        citation = next(
            (row for row in refs["citations"] if row["kind"] == "table" and "target" in row),
            None,
        )
        if citation is not None:
            citation["target"]["char_end"] -= 1
            path.write_text(json.dumps(refs), encoding="utf-8")
            break
    else:
        pytest.fail("the synthetic projection has no table citation")
    with pytest.raises(ValueError, match="reference target is not a printed block"):
        build_book_store(projection, tmp_path / "out", khop_depth=BOOK_KHOP_DEPTH)
