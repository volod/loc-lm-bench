"""The node-cluster overlay: a proposed canonical identity per node, applied to a COPY.

An overlay is the only thing this lane produces. It names, for one candidate threshold, which
node ids the linkage model put in one identity and which of them is canonical. Applying it builds
a NEW `KnowledgeGraph` -- the stored graph is never rewritten, so the pre-merge reading can always
be redone (`docs/design/spec.md#entity-resolution-and-record-linkage`).
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.core.fsutil import atomic_write_text
from llb.graph.community import assign_communities
from llb.graph.model import GraphEdge, GraphMention, GraphNode, KnowledgeGraph
from llb.graph.resolution.records import node_id_of
from llb.linkage.model import LinkageCluster


@dataclass(frozen=True)
class NodeCluster:
    """One proposed identity: the member node ids and the one that speaks for them."""

    canonical_id: int
    member_ids: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.member_ids)

    def payload(self) -> JsonObject:
        return {
            "canonical_id": self.canonical_id,
            "size": self.size,
            "member_ids": list(self.member_ids),
        }

    @classmethod
    def from_payload(cls, payload: JsonObject) -> "NodeCluster":
        return cls(
            canonical_id=int(payload["canonical_id"]),
            member_ids=tuple(int(member) for member in payload["member_ids"]),
        )


@dataclass(frozen=True)
class NodeOverlay:
    """Every multi-node identity proposed at one threshold. Singletons are not carried."""

    threshold: float
    clusters: tuple[NodeCluster, ...]

    @property
    def canonical_of(self) -> dict[int, int]:
        """member node id -> canonical node id, for the merged members only."""
        return {
            member: cluster.canonical_id
            for cluster in self.clusters
            for member in cluster.member_ids
        }

    @property
    def n_nodes_merged(self) -> int:
        """How many nodes disappear into a canonical one -- the size of the change proposed."""
        return sum(cluster.size - 1 for cluster in self.clusters)

    @property
    def largest_cluster(self) -> int:
        return max((cluster.size for cluster in self.clusters), default=0)

    def summary(self) -> JsonObject:
        return {
            "threshold": self.threshold,
            "n_clusters": len(self.clusters),
            "n_nodes_merged": self.n_nodes_merged,
            "largest_cluster": self.largest_cluster,
        }


def _canonical(nodes: Sequence[GraphNode]) -> GraphNode:
    """The member that speaks for a cluster: most grounded, then longest name, then lowest id.

    Mention count first because the canonical node is the one whose evidence the graph already
    trusts most; the name length tie-break prefers the spelled-out form over an initialism when
    both were seen equally often, which is what a reader of a merged node wants to see.
    """
    return max(nodes, key=lambda node: (len(node.mentions), len(node.name), -node.node_id))


def overlay_from_clusters(
    clusters: Sequence[LinkageCluster], graph: KnowledgeGraph, threshold: float
) -> NodeOverlay:
    """Turn the linkage seam's record clusters into a node overlay at one threshold."""
    by_id = graph.node_by_id()
    proposed: list[NodeCluster] = []
    for cluster in clusters:
        member_ids = sorted(node_id_of(record) for record in cluster.record_ids)
        members = [by_id[node_id] for node_id in member_ids if node_id in by_id]
        if len(members) < 2:
            continue
        proposed.append(
            NodeCluster(
                canonical_id=_canonical(members).node_id,
                member_ids=tuple(member.node_id for member in members),
            )
        )
    proposed.sort(key=lambda cluster: (-cluster.size, cluster.canonical_id))
    return NodeOverlay(threshold=threshold, clusters=tuple(proposed))


def _merged_node(canonical: GraphNode, members: Sequence[GraphNode]) -> GraphNode:
    """One node carrying every member's surface forms and every member's mention spans."""
    aliases: list[str] = []
    seen_forms = {canonical.name.casefold()}
    mentions: list[GraphMention] = []
    seen_spans: set[tuple[str, int, int]] = set()
    for member in members:
        for form in (member.name, *member.aliases):
            key = str(form).casefold()
            if key and key not in seen_forms:
                seen_forms.add(key)
                aliases.append(str(form))
        for mention in member.mentions:
            marker = (mention["doc_id"], mention["char_start"], mention["char_end"])
            if marker not in seen_spans:
                seen_spans.add(marker)
                mentions.append(mention)
    return GraphNode(
        node_id=canonical.node_id,
        name=canonical.name,
        type=canonical.type,
        confidence=canonical.confidence,
        aliases=aliases,
        mentions=mentions,
        community_id=canonical.community_id,
    )


def apply_overlay(graph: KnowledgeGraph, overlay: NodeOverlay) -> KnowledgeGraph:
    """Build the merged COPY of `graph` the overlay proposes, with communities re-detected.

    Communities are re-detected rather than carried: merging nodes changes the adjacency the
    label propagation reads, so a carried `community_id` would describe a graph that no longer
    exists. The detection is deterministic, so the merged partition is reproducible too.
    """
    canonical_of = overlay.canonical_of
    by_id = graph.node_by_id()
    members_of: dict[int, list[GraphNode]] = {}
    for cluster in overlay.clusters:
        members_of[cluster.canonical_id] = [
            by_id[member] for member in cluster.member_ids if member in by_id
        ]
    nodes = [
        _merged_node(node, members_of[node.node_id])
        if node.node_id in members_of
        else _copied_node(node)
        for node in graph.nodes
        if canonical_of.get(node.node_id, node.node_id) == node.node_id
    ]
    merged = KnowledgeGraph(nodes=nodes, edges=_merged_edges(graph, canonical_of))
    assign_communities(merged)
    return merged


def _copied_node(node: GraphNode) -> GraphNode:
    """An untouched node, copied so the overlay graph shares no mutable state with the source."""
    return GraphNode(
        node_id=node.node_id,
        name=node.name,
        type=node.type,
        confidence=node.confidence,
        aliases=list(node.aliases),
        mentions=list(node.mentions),
        community_id=node.community_id,
    )


def _merged_edges(graph: KnowledgeGraph, canonical_of: dict[int, int]) -> list[GraphEdge]:
    """Every edge remapped onto canonical endpoints, deduplicated by fact and evidence span.

    A self-loop is KEPT when it survives the remap: its evidence is a grounded fact whose span the
    serializer still emits, and dropping it would lose evidence the pre-merge graph carried.
    """
    edges: list[GraphEdge] = []
    seen: set[tuple[int, int, str, str, int, int]] = set()
    for edge in graph.edges:
        src = canonical_of.get(edge.src, edge.src)
        dst = canonical_of.get(edge.dst, edge.dst)
        evidence = edge.evidence
        marker = (
            src,
            dst,
            edge.relation,
            evidence["doc_id"],
            evidence["char_start"],
            evidence["char_end"],
        )
        if marker in seen:
            continue
        seen.add(marker)
        edges.append(
            GraphEdge(
                edge_id=len(edges), src=src, dst=dst, relation=edge.relation, evidence=evidence
            )
        )
    return edges


def write_overlay(overlay: NodeOverlay, path: Path, graph: KnowledgeGraph) -> None:
    """Write one overlay as JSONL: a header row, then one row per proposed identity.

    Each row names the canonical form and every member's name, because a reader deciding whether
    a merge is right reads names, not node ids.
    """
    by_id = graph.node_by_id()
    rows = [{"kind": "overlay", **overlay.summary()}]
    rows.extend(
        {
            "kind": "cluster",
            **cluster.payload(),
            "canonical_name": by_id[cluster.canonical_id].name,
            "member_names": [by_id[member].name for member in cluster.member_ids],
        }
        for cluster in overlay.clusters
    )
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _overlay_rows(path: Path) -> list[JsonObject]:
    """Every JSONL row of a written overlay, header included, in file order."""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_overlay_surfaces(path: Path) -> dict[str, str]:
    """member entity NAME -> the canonical name its cluster proposes, for a reader with no graph.

    `write_overlay` already records both names on every row, because a reviewer deciding whether a
    merge is right reads names rather than node ids. Reading them back is what lets a consumer
    that holds no `KnowledgeGraph` -- the answer gate, which checks a declared answer against an
    extraction ledger -- fold a surface through the identity THIS lane proposed instead of
    inventing a second notion of it.
    """
    return {
        str(member): str(row["canonical_name"])
        for row in _overlay_rows(path)
        if row.get("kind") == "cluster"
        for member in row.get("member_names", [])
    }


def read_overlay(path: Path) -> NodeOverlay:
    """Read back a written overlay (the names in each row are for the reader, not the model)."""
    rows = _overlay_rows(path)
    header = next((row for row in rows if row.get("kind") == "overlay"), None)
    if header is None:
        raise ValueError(f"no overlay header row in {path}")
    return NodeOverlay(
        threshold=float(header["threshold"]),
        clusters=tuple(
            NodeCluster.from_payload(row) for row in rows if row.get("kind") == "cluster"
        ),
    )
