"""Graph store IO: the DuckDB engine and the contract-bound node/edge row files.

Node and edge rows are written with their registered identity and read back through the registry,
so a graph written by a newer build refuses at the door rather than losing the fields this reader
cannot see. The identity is a property of the FILE: it is stripped again on the way in, because
`GraphNode` and `GraphEdge` are the dataclasses every strategy is written against.
"""

import json

from dataclasses import asdict
from pathlib import Path

from typing import Any

from llb.artifacts.records import decode, encode
from llb.core.contracts.retrieval.graph import (
    GRAPH_COMMUNITY_SUMMARIES_SCHEMA_ID,
    GRAPH_EDGE_SCHEMA_ID,
    GRAPH_NODE_SCHEMA_ID,
    GRAPH_STORE_META_SCHEMA_ID,
)
from llb.graph.constants import EDGES_FILE, META_FILE, NODES_FILE, SUMMARIES_FILE
from llb.graph.model import KnowledgeGraph

NODE_CONTRACT_VERSION = "1.0.0"
EDGE_CONTRACT_VERSION = "1.0.0"
GRAPH_META_CONTRACT_VERSION = "1.0.0"
SUMMARIES_CONTRACT_VERSION = "1.0.0"


def _connect(graph: KnowledgeGraph) -> Any:
    """Build an in-memory DuckDB engine over the graph's edges + node community ids."""
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SystemExit(
            'ERROR: the graph backend needs the [graph] extra. Run: uv pip install -e ".[graph]"'
        ) from exc
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE node(node_id INTEGER, community_id INTEGER)")
    con.execute("CREATE TABLE edge(src INTEGER, dst INTEGER)")
    if graph.nodes:
        con.executemany(
            "INSERT INTO node VALUES (?, ?)",
            [(n.node_id, n.community_id) for n in graph.nodes],
        )
    if graph.edges:
        con.executemany("INSERT INTO edge VALUES (?, ?)", [(e.src, e.dst) for e in graph.edges])
    return con


def write_rows(rows: Any, path: Path, schema_id: str, version: str) -> None:
    """Write graph rows with their contract identity, one JSON object per line."""
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(encode(schema_id, version, row), ensure_ascii=False) + "\n")


def write_graph(
    graph_dir: Path, graph: KnowledgeGraph, meta: dict[str, Any], summaries: dict[str, str]
) -> None:
    """Write one graph's project-owned members, each with its contract identity.

    The community summaries are written only when the graph carries them: they are tagged
    DIAGNOSTIC everywhere, so a graph without them is complete rather than incomplete.
    """
    graph_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        (asdict(node) for node in graph.nodes),
        graph_dir / NODES_FILE,
        GRAPH_NODE_SCHEMA_ID,
        NODE_CONTRACT_VERSION,
    )
    write_rows(
        (asdict(edge) for edge in graph.edges),
        graph_dir / EDGES_FILE,
        GRAPH_EDGE_SCHEMA_ID,
        EDGE_CONTRACT_VERSION,
    )
    _write_record(
        graph_dir / META_FILE, GRAPH_STORE_META_SCHEMA_ID, GRAPH_META_CONTRACT_VERSION, meta
    )
    if summaries:
        _write_record(
            graph_dir / SUMMARIES_FILE,
            GRAPH_COMMUNITY_SUMMARIES_SCHEMA_ID,
            SUMMARIES_CONTRACT_VERSION,
            {"role": "diagnostic", "summaries": summaries},
        )


def _write_record(path: Path, schema_id: str, version: str, payload: dict[str, Any]) -> None:
    record = encode(schema_id, version, payload)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def read_rows(path: Path, schema_id: str) -> list[dict[str, Any]]:
    """Read graph rows at the current contract, identity removed, migrating an older file."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [
        decode(schema_id, json.loads(line), source=f"{path}#record-{index}")
        for index, line in enumerate(lines, start=1)
    ]


def read_node_rows(path: Path) -> list[dict[str, Any]]:
    return read_rows(path, GRAPH_NODE_SCHEMA_ID)


def read_edge_rows(path: Path) -> list[dict[str, Any]]:
    return read_rows(path, GRAPH_EDGE_SCHEMA_ID)


def read_graph_meta(path: Path) -> dict[str, Any]:
    """The graph metadata at the current contract, migrating a pre-contract file forward."""
    return decode(
        GRAPH_STORE_META_SCHEMA_ID,
        json.loads(path.read_text(encoding="utf-8")),
        source=str(path),
    )


def read_community_summaries(path: Path) -> dict[str, str]:
    """The diagnostic community summaries, or an empty map when the graph carries none.

    A pre-contract file was the bare `{community_id: summary}` map, which has no place to put an
    identity; the family's declared normalizer says that map became the `summaries` field, so this
    reader and the dataset reader wrap it the same way.
    """
    if not path.exists():
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    summaries = decode(GRAPH_COMMUNITY_SUMMARIES_SCHEMA_ID, record, source=str(path)).get(
        "summaries"
    )
    return dict(summaries) if isinstance(summaries, dict) else {}
