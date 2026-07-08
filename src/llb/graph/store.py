"""GraphStore -- the GraphRAG retrieval backend behind the RAG-store seam.

It quacks like `rag.store.RagStore`: a `.retrieve(question, k) -> list[ChunkRecord]` of
offset-bearing context, so the eval graph, scoring (incl. the gated judge), isolation, and the
board are UNCHANGED (selected with `--retrieval-backend graph`). FAISS stays the default and is
untouched.

Store choice = DuckDB (already a dependency; no abandoned graph DB). The graph is persisted as
node/edge JSONL (inspectable, diffable, the same shape as the FAISS store's chunks) and loaded
into an in-memory DuckDB engine that carries the two graph queries the strategies need:

  - local_khop      -- entity-link the question to seed nodes, then expand k hops with a
                       recursive CTE over the (undirected) edge table.
  - global_community -- map the question to its communities, then group members with
                       `WHERE community_id IN (...)`; community ids are precomputed OFFLINE and
                       stored as a column, so query time needs no graph-analytics dependency.

Both serialize node mentions + edge evidence WITH their source spans, so the existing span metric
applies. `duckdb` is lazy-imported, so the package still imports in the base install.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llb.graph.community import assign_communities
from llb.graph.constants import (
    BACKEND_GRAPH,
    DEFAULT_KHOP_DEPTH,
    DEFAULT_N_COMMUNITIES,
    DEFAULT_N_SEED_NODES,
    EDGES_FILE,
    META_FILE,
    NODES_FILE,
    STRATEGIES,
    STRATEGY_GLOBAL_COMMUNITY,
    STRATEGY_LOCAL_KHOP,
    SUMMARIES_FILE,
)
from llb.graph.model import GraphEdge, GraphNode, KnowledgeGraph
from llb.graph.retrieval import (
    link_communities,
    link_seed_nodes,
    node_link_scores,
    serialize_subgraph,
)
from llb.core.contracts import ChunkRecord
from llb.prep.ontology.models import DocExtraction, DocRecord, OntologyCandidate

_LOG = logging.getLogger(__name__)

# global_community ranks matched nodes by link score; this tiny floor still serializes the rest of
# the community (corpus-level context) below the matched members.
_UNMATCHED_MEMBER_FLOOR = 0.001


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


# Undirected k-hop neighborhood of the seed nodes: a recursive CTE bounded by `depth`, returning
# each reached node at its shortest hop distance.
_KHOP_SQL = """
WITH RECURSIVE undirected(a, b) AS (
    SELECT src, dst FROM edge
    UNION ALL
    SELECT dst, src FROM edge
),
reach(node_id, depth) AS (
    SELECT x, 0 FROM (SELECT UNNEST(?) AS x)
    UNION
    SELECT u.b, r.depth + 1
    FROM reach r JOIN undirected u ON u.a = r.node_id
    WHERE r.depth < ?
)
SELECT node_id, min(depth) AS depth FROM reach GROUP BY node_id
"""

_COMMUNITY_SQL = "SELECT node_id FROM node WHERE community_id IN (SELECT UNNEST(?))"


class GraphStore:
    """Persisted knowledge graph + DuckDB query engine + the two span-preserving strategies."""

    def __init__(
        self,
        graph: KnowledgeGraph,
        meta: dict[str, Any],
        *,
        strategy: str = STRATEGY_LOCAL_KHOP,
        khop_depth: int = DEFAULT_KHOP_DEPTH,
        n_seeds: int = DEFAULT_N_SEED_NODES,
        n_communities: int = DEFAULT_N_COMMUNITIES,
        community_summaries: dict[str, str] | None = None,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(
                f"unknown retrieval strategy: {strategy} (expected one of {STRATEGIES})"
            )
        self.graph = graph
        self.meta = meta
        self.strategy = strategy
        self.khop_depth = khop_depth
        self.n_seeds = n_seeds
        self.n_communities = n_communities
        # tagged DIAGNOSTIC only (community_id -> summary); NEVER returned by retrieve / span-scored
        self.community_summaries = community_summaries or {}
        self._con: Any = None

    @property
    def connection(self) -> Any:
        if self._con is None:
            self._con = _connect(self.graph)
        return self._con

    @classmethod
    def build(
        cls,
        extractions: list[DocExtraction],
        docs: list[DocRecord],
        ontology: OntologyCandidate | None = None,
        *,
        strategy: str = STRATEGY_LOCAL_KHOP,
        khop_depth: int = DEFAULT_KHOP_DEPTH,
        n_seeds: int = DEFAULT_N_SEED_NODES,
        n_communities: int = DEFAULT_N_COMMUNITIES,
    ) -> "GraphStore":
        """Build the graph from ontology-assisted drafting extraction, detect communities, return the store."""
        from llb.graph.build import build_graph

        graph = build_graph(extractions, docs, ontology)
        n_communities_found = assign_communities(graph)
        meta = {
            "backend": BACKEND_GRAPH,
            "n_nodes": len(graph.nodes),
            "n_edges": len(graph.edges),
            "n_communities": n_communities_found,
            "n_documents": len(docs),
            "khop_depth": khop_depth,
        }
        return cls(
            graph,
            meta,
            strategy=strategy,
            khop_depth=khop_depth,
            n_seeds=n_seeds,
            n_communities=n_communities,
        )

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        """Top-k offset-bearing context for `question` under the configured strategy."""
        if self.strategy == STRATEGY_GLOBAL_COMMUNITY:
            return self._retrieve_global_community(question, k)
        return self._retrieve_local_khop(question, k)

    def _retrieve_local_khop(self, question: str, k: int) -> list[ChunkRecord]:
        seeds = link_seed_nodes(self.graph, question, self.n_seeds)
        if not seeds:
            return []
        rows = self.connection.execute(_KHOP_SQL, [seeds, self.khop_depth]).fetchall()
        # relevance decays with hop distance from the seeds (seed mentions rank first)
        node_relevance = {int(node_id): 1.0 / (1 + int(depth)) for node_id, depth in rows}
        return serialize_subgraph(self.graph, node_relevance, k)

    def _retrieve_global_community(self, question: str, k: int) -> list[ChunkRecord]:
        community_ids = link_communities(self.graph, question, self.n_communities)
        if not community_ids:
            return []
        member_rows = self.connection.execute(_COMMUNITY_SQL, [community_ids]).fetchall()
        link = node_link_scores(self.graph, question)
        node_relevance = {
            int(row[0]): link.get(int(row[0]), 0.0) + _UNMATCHED_MEMBER_FLOOR for row in member_rows
        }
        return serialize_subgraph(self.graph, node_relevance, k)

    def save(self, graph_dir: Path | str) -> None:
        graph_dir = Path(graph_dir)
        graph_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl((asdict(n) for n in self.graph.nodes), graph_dir / NODES_FILE)
        _write_jsonl((asdict(e) for e in self.graph.edges), graph_dir / EDGES_FILE)
        (graph_dir / META_FILE).write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if self.community_summaries:
            (graph_dir / SUMMARIES_FILE).write_text(
                json.dumps(self.community_summaries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    @classmethod
    def load(
        cls,
        graph_dir: Path | str,
        *,
        strategy: str = STRATEGY_LOCAL_KHOP,
        khop_depth: int | None = None,
        n_seeds: int = DEFAULT_N_SEED_NODES,
        n_communities: int = DEFAULT_N_COMMUNITIES,
    ) -> "GraphStore":
        graph_dir = Path(graph_dir)
        meta_path = graph_dir / META_FILE
        if not meta_path.exists():
            raise SystemExit(f"no graph store at {graph_dir} (run `llb build-graph` first)")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        nodes = [GraphNode(**row) for row in _read_jsonl(graph_dir / NODES_FILE)]
        edges = [GraphEdge(**row) for row in _read_jsonl(graph_dir / EDGES_FILE)]
        summaries_path = graph_dir / SUMMARIES_FILE
        summaries = (
            json.loads(summaries_path.read_text(encoding="utf-8"))
            if summaries_path.exists()
            else {}
        )
        return cls(
            KnowledgeGraph(nodes=nodes, edges=edges),
            meta,
            strategy=strategy,
            khop_depth=khop_depth
            if khop_depth is not None
            else int(meta.get("khop_depth", DEFAULT_KHOP_DEPTH)),
            n_seeds=n_seeds,
            n_communities=n_communities,
            community_summaries=summaries,
        )


def _write_jsonl(rows: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
