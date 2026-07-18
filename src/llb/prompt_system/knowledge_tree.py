"""Compact knowledge-tree inputs and deterministic token-budgeted rendering."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llb.graph.constants import EDGES_FILE, NODES_FILE, SUMMARIES_FILE
from llb.graph.model import GraphEdge, GraphNode, KnowledgeGraph
from llb.prompt_system.budget import Tokenizer
from llb.prompts.registry import render_text

MAX_TREE_DEPTH = 3
DEFAULT_TREE_DEPTHS = (1, 2, 3)
DEFAULT_TREE_BUDGETS = (128, 256)
MAX_VOCABULARY_ITEMS = 12
MAX_COMMUNITY_MEMBERS = 8
MAX_NODE_RELATIONS = 3


@dataclass(slots=True)
class KnowledgeTreeSource:
    """Ontology vocabulary, graph communities, and optional diagnostic summaries."""

    entity_types: list[tuple[str, int]]
    relation_types: list[tuple[str, int]]
    graph: KnowledgeGraph
    community_summaries: dict[str, str]
    source_kind: str
    source_digest: str


@dataclass(slots=True)
class KnowledgeTreeRender:
    """One rendered tree block and its strict knob/provenance report."""

    text: str
    depth: int
    budget_tokens: int
    used_tokens: int
    kept_lines: int
    dropped_lines: int
    source_kind: str
    source_digest: str

    def report(self) -> dict[str, object]:
        report = asdict(self)
        del report["text"]
        return report


def load_knowledge_tree_source(
    *,
    ontology_bundle: Path | str | None = None,
    graph_dir: Path | str | None = None,
) -> KnowledgeTreeSource:
    """Load a tree source from an ontology bundle, a graph store, or both."""
    if ontology_bundle is None and graph_dir is None:
        raise ValueError("knowledge-tree generation needs an ontology bundle or graph store")

    ontology = None
    graph = None
    summaries: dict[str, str] = {}
    kinds: list[str] = []
    if ontology_bundle is not None:
        bundle = Path(ontology_bundle)
        from llb.graph.ingest import load_bundle, load_ontology
        from llb.prep.ontology.constants import ONTOLOGY_FILENAME

        ontology = load_ontology(bundle / ONTOLOGY_FILENAME)
        if ontology is None:
            raise ValueError(f"ontology bundle has no {ONTOLOGY_FILENAME}: {bundle}")
        kinds.append("ontology-bundle")
        if graph_dir is None:
            from llb.graph.build import build_graph
            from llb.graph.community import assign_communities

            extractions, docs, _ = load_bundle(bundle)
            graph = build_graph(extractions, docs, ontology)
            assign_communities(graph)
    if graph_dir is not None:
        graph, summaries = _load_graph(Path(graph_dir))
        kinds.append("graph-store")
    assert graph is not None

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
    payload = {
        "entity_types": entity_types,
        "relation_types": relation_types,
        "nodes": [asdict(node) for node in graph.nodes],
        "edges": [asdict(edge) for edge in graph.edges],
        "community_summaries": summaries,
    }
    digest = hashlib.sha256(_canonical(payload)).hexdigest()[:12]
    return KnowledgeTreeSource(
        entity_types=_ranked(entity_types),
        relation_types=_ranked(relation_types),
        graph=graph,
        community_summaries=summaries,
        source_kind="+".join(kinds),
        source_digest=digest,
    )


def render_knowledge_tree(
    source: KnowledgeTreeSource,
    *,
    depth: int,
    budget_tokens: int,
    tokenizer: Tokenizer,
) -> KnowledgeTreeRender:
    """Render the requested maximum depth without exceeding its token budget."""
    if depth < 1 or depth > MAX_TREE_DEPTH:
        raise ValueError(f"knowledge_tree_depth must be between 1 and {MAX_TREE_DEPTH}")
    if budget_tokens < 0:
        raise ValueError("knowledge_tree_budget must be >= 0")
    lines = _tree_lines(source, depth)
    kept: list[str] = []
    for line in lines:
        candidate = render_text(
            "prompt_system.knowledge_tree_block", {"tree": "\n".join(kept + [line])}
        )
        if tokenizer.count(candidate) <= budget_tokens:
            kept.append(line)
    text = (
        render_text("prompt_system.knowledge_tree_block", {"tree": "\n".join(kept)}) if kept else ""
    )
    return KnowledgeTreeRender(
        text=text,
        depth=depth,
        budget_tokens=budget_tokens,
        used_tokens=tokenizer.count(text),
        kept_lines=len(kept),
        dropped_lines=len(lines) - len(kept),
        source_kind=source.source_kind,
        source_digest=source.source_digest,
    )


def _tree_lines(source: KnowledgeTreeSource, depth: int) -> list[str]:
    if depth == 1:
        lines = [
            f"- Entity types: {_names(source.entity_types)}",
            f"- Relation types: {_names(source.relation_types)}",
        ]
        lines.extend(
            _community_line(source, cid, members, "- ") for cid, members in _communities(source)
        )
        return lines

    lines = ["- Vocabulary"]
    lines.extend(
        f"  - Entity type {name} ({count})"
        for name, count in source.entity_types[:MAX_VOCABULARY_ITEMS]
    )
    lines.extend(
        f"  - Relation type {name} ({count})"
        for name, count in source.relation_types[:MAX_VOCABULARY_ITEMS]
    )
    lines.append("- Communities")
    by_id = source.graph.node_by_id()
    outgoing: dict[int, list[GraphEdge]] = {}
    for edge in source.graph.edges:
        outgoing.setdefault(edge.src, []).append(edge)
    for cid, members in _communities(source):
        lines.append(_community_line(source, cid, members, "  - "))
        if depth < 3:
            continue
        for node in _rank_nodes(source.graph, members)[:MAX_COMMUNITY_MEMBERS]:
            relations = [
                f"{edge.relation} -> {by_id[edge.dst].name}"
                for edge in outgoing.get(node.node_id, [])
                if edge.dst in by_id
            ][:MAX_NODE_RELATIONS]
            suffix = f": {'; '.join(relations)}" if relations else ""
            lines.append(f"    - {node.name} [{node.type}]{suffix}")
    return lines


def _community_line(
    source: KnowledgeTreeSource, community_id: int, members: list[int], prefix: str
) -> str:
    names = ", ".join(
        node.name for node in _rank_nodes(source.graph, members)[:MAX_COMMUNITY_MEMBERS]
    )
    summary = " ".join(source.community_summaries.get(str(community_id), "").split())
    detail = summary or names
    return f"{prefix}Community {community_id}: {detail}"


def _communities(source: KnowledgeTreeSource) -> list[tuple[int, list[int]]]:
    return sorted(
        source.graph.community_members().items(), key=lambda item: (-len(item[1]), item[0])
    )


def _rank_nodes(graph: KnowledgeGraph, members: list[int]) -> list[GraphNode]:
    degree = {node_id: 0 for node_id in members}
    for edge in graph.edges:
        if edge.src in degree:
            degree[edge.src] += 1
        if edge.dst in degree:
            degree[edge.dst] += 1
    by_id = graph.node_by_id()
    return sorted(
        (by_id[nid] for nid in members),
        key=lambda node: (-degree[node.node_id], node.name.casefold()),
    )


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


def _names(items: list[tuple[str, int]]) -> str:
    return ", ".join(name for name, _ in items[:MAX_VOCABULARY_ITEMS]) or "none"


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
