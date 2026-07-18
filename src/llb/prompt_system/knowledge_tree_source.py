"""Knowledge-tree source models and artifact loading."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llb.graph.constants import EDGES_FILE, NODES_FILE, SUMMARIES_FILE
from llb.graph.model import GraphEdge, GraphNode, KnowledgeGraph

MAX_VOCABULARY_ITEMS = 12


@dataclass(slots=True)
class KnowledgeTreeSource:
    """Ontology vocabulary, graph communities, and optional diagnostic summaries."""

    entity_types: list[tuple[str, int]]
    relation_types: list[tuple[str, int]]
    graph: KnowledgeGraph
    community_summaries: dict[str, str]
    source_kind: str
    source_digest: str


def load_knowledge_tree_source(
    *,
    ontology_bundle: Path | str | None = None,
    graph_dir: Path | str | None = None,
) -> KnowledgeTreeSource:
    """Load a tree source from an ontology bundle, a graph store, or both."""
    if ontology_bundle is None and graph_dir is None:
        raise ValueError("knowledge-tree generation needs an ontology bundle or graph store")
    ontology, graph, summaries, kinds = _resolve_sources(ontology_bundle, graph_dir)
    entity_types, relation_types = _vocabulary(ontology, graph)
    digest = _source_digest(entity_types, relation_types, graph, summaries)
    return KnowledgeTreeSource(
        entity_types=_ranked(entity_types),
        relation_types=_ranked(relation_types),
        graph=graph,
        community_summaries=summaries,
        source_kind="+".join(kinds),
        source_digest=digest,
    )


def _resolve_sources(
    ontology_bundle: Path | str | None, graph_dir: Path | str | None
) -> tuple[Any, KnowledgeGraph, dict[str, str], list[str]]:
    ontology = None
    graph = None
    summaries: dict[str, str] = {}
    kinds: list[str] = []
    if ontology_bundle is not None:
        ontology = _load_ontology(Path(ontology_bundle))
        kinds.append("ontology-bundle")
    if graph_dir is not None:
        graph, summaries = _load_graph(Path(graph_dir))
        kinds.append("graph-store")
    else:
        assert ontology_bundle is not None and ontology is not None
        graph = _build_ontology_graph(Path(ontology_bundle), ontology)
    assert graph is not None
    return ontology, graph, summaries, kinds


def _vocabulary(
    ontology: Any, graph: KnowledgeGraph
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    entity_types = (
        [(item.name, item.count) for item in ontology.entity_types]
        if ontology is not None
        else _counts(node.type for node in graph.nodes)
    )
    relation_types = (
        [(item.name, item.count) for item in ontology.relation_types]
        if ontology is not None
        else _counts(edge.relation for edge in graph.edges)
    )
    return entity_types, relation_types


def _source_digest(
    entity_types: list[tuple[str, int]],
    relation_types: list[tuple[str, int]],
    graph: KnowledgeGraph,
    summaries: dict[str, str],
) -> str:
    payload = {
        "entity_types": entity_types,
        "relation_types": relation_types,
        "nodes": [asdict(node) for node in graph.nodes],
        "edges": [asdict(edge) for edge in graph.edges],
        "community_summaries": summaries,
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()[:12]


def _load_ontology(bundle: Path) -> Any:
    from llb.graph.ingest import load_ontology
    from llb.prep.ontology.constants import ONTOLOGY_FILENAME

    ontology = load_ontology(bundle / ONTOLOGY_FILENAME)
    if ontology is None:
        raise ValueError(f"ontology bundle has no {ONTOLOGY_FILENAME}: {bundle}")
    return ontology


def _build_ontology_graph(bundle: Path, ontology: Any) -> KnowledgeGraph:
    from llb.graph.build import build_graph
    from llb.graph.community import assign_communities
    from llb.graph.ingest import load_bundle

    extractions, docs, _ = load_bundle(bundle)
    graph = build_graph(extractions, docs, ontology)
    assign_communities(graph)
    return graph


def _load_graph(path: Path) -> tuple[KnowledgeGraph, dict[str, str]]:
    nodes = [GraphNode(**row) for row in _read_jsonl(path / NODES_FILE)]
    edges = [GraphEdge(**row) for row in _read_jsonl(path / EDGES_FILE)]
    summaries_path = path / SUMMARIES_FILE
    summaries = (
        json.loads(summaries_path.read_text(encoding="utf-8")) if summaries_path.exists() else {}
    )
    return KnowledgeGraph(nodes=nodes, edges=edges), {str(k): str(v) for k, v in summaries.items()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"knowledge-tree graph store is missing {path.name}: {path.parent}")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _counts(values: Any) -> list[tuple[str, int]]:
    return list(Counter(str(value) for value in values if str(value)).items())


def _ranked(items: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return sorted(items, key=lambda item: (-item[1], item[0].casefold()))


def names(items: list[tuple[str, int]]) -> str:
    """Render the bounded vocabulary summary used by depth-one trees."""
    return ", ".join(name for name, _ in items[:MAX_VOCABULARY_ITEMS]) or "none"


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
