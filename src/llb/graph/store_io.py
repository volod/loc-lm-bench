"""Focused store io implementation."""

import json

from pathlib import Path

from typing import Any

from llb.graph.model import KnowledgeGraph


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


def _write_jsonl(rows: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
