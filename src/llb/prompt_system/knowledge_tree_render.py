"""Deterministic, token-budgeted knowledge-tree rendering."""

from dataclasses import asdict, dataclass

from llb.graph.model import GraphEdge, GraphNode, KnowledgeGraph
from llb.prompt_system.budget import Tokenizer
from llb.prompt_system.knowledge_tree_source import (
    MAX_VOCABULARY_ITEMS,
    KnowledgeTreeSource,
    names,
)
from llb.prompts.registry import render_text

MAX_TREE_DEPTH = 3
DEFAULT_TREE_DEPTHS = (1, 2, 3)
DEFAULT_TREE_BUDGETS = (128, 256)
MAX_COMMUNITY_MEMBERS = 8
MAX_NODE_RELATIONS = 3


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
            f"- Entity types: {names(source.entity_types)}",
            f"- Relation types: {names(source.relation_types)}",
        ]
        lines.extend(
            _community_line(source, community_id, members, "- ")
            for community_id, members in _communities(source)
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
    for community_id, members in _communities(source):
        lines.append(_community_line(source, community_id, members, "  - "))
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
    node_names = ", ".join(
        node.name for node in _rank_nodes(source.graph, members)[:MAX_COMMUNITY_MEMBERS]
    )
    summary = " ".join(source.community_summaries.get(str(community_id), "").split())
    return f"{prefix}Community {community_id}: {summary or node_names}"


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
        (by_id[node_id] for node_id in members),
        key=lambda node: (-degree[node.node_id], node.name.casefold()),
    )
