"""Walk 2-hop knowledge-graph paths into multi-hop draft seeds (yield-max).

A multi-hop question needs evidence from more than one fact. This module walks directed 2-hop
chains `A -r1-> B -r2-> C` over the GraphRAG knowledge graph (built by REUSING the ontology
extraction -- no second extraction framework) and turns each into a `MultiHopSeed` whose two steps
carry their exact evidence spans. The drafter (`multi_hop.py`) then asks for a question that needs
BOTH facts; the two evidence spans become the item's grounded, multi-span source spans.

Pure + deterministic: middle nodes and edges are visited in stable id order and each distinct span
pair is emitted once, so a resume reproduces the same seeds. Imports only the graph MODEL (no
DuckDB / store), so the base install still imports this module.
"""

import logging
from collections import defaultdict

from llb.goldset.schema import SourceSpan
from llb.graph.model import GraphEdge, GraphMention, KnowledgeGraph
from llb.prep.ontology.constants import DEFAULT_MULTI_HOP_MAX_PATHS
from llb.prep.ontology.models import MultiHopSeed, MultiHopStep

_LOG = logging.getLogger(__name__)


def _span(mention: GraphMention) -> SourceSpan:
    return SourceSpan(
        doc_id=mention["doc_id"],
        char_start=mention["char_start"],
        char_end=mention["char_end"],
        text=mention["text"],
    )


def _span_key(mention: GraphMention) -> tuple[str, int, int]:
    return (mention["doc_id"], mention["char_start"], mention["char_end"])


def _step(edge: GraphEdge, subject: str, obj: str) -> MultiHopStep:
    return MultiHopStep(
        subject=subject,
        relation=edge.relation,
        object=obj,
        section_title=edge.evidence["section_title"],
        evidence=_span(edge.evidence),
    )


def walk_two_hop_paths(
    graph: KnowledgeGraph, *, max_paths: int = DEFAULT_MULTI_HOP_MAX_PATHS, seed: int = 13
) -> list[MultiHopSeed]:
    """Emit up to `max_paths` distinct 2-hop `A -r1-> B -r2-> C` seeds, deterministically.

    A path is kept only when the endpoints differ (`A != C`, and neither equals the bridge B) and
    the two hops cite DISTINCT evidence spans, so the drafted item genuinely needs both facts and
    carries >= 2 grounded spans. `seed` is accepted for signature symmetry with the flat sampler;
    the walk is already fully deterministic by id order.
    """
    del seed  # walk order is deterministic by node/edge id; kept for API symmetry
    by_id = graph.node_by_id()
    incoming: dict[int, list[GraphEdge]] = defaultdict(list)
    outgoing: dict[int, list[GraphEdge]] = defaultdict(list)
    for edge in graph.edges:  # edges are in ascending edge_id order -> stable
        incoming[edge.dst].append(edge)
        outgoing[edge.src].append(edge)

    seeds: list[MultiHopSeed] = []
    seen_pairs: set[tuple[tuple[str, int, int], tuple[str, int, int]]] = set()
    for mid in sorted(set(incoming) & set(outgoing)):
        if mid not in by_id:
            continue
        for e1 in incoming[mid]:
            for e2 in outgoing[mid]:
                a, c = e1.src, e2.dst
                if a == c or a == mid or c == mid:
                    continue
                k1, k2 = _span_key(e1.evidence), _span_key(e2.evidence)
                if k1 == k2:
                    continue
                pair = (k1, k2) if k1 <= k2 else (k2, k1)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                seeds.append(
                    MultiHopSeed(
                        steps=[
                            _step(e1, by_id[a].name, by_id[mid].name),
                            _step(e2, by_id[mid].name, by_id[c].name),
                        ],
                        bridge=by_id[mid].name,
                        start=by_id[a].name,
                        end=by_id[c].name,
                    )
                )
                if len(seeds) >= max_paths:
                    _LOG.info("[ontology] multi-hop: %d 2-hop seeds (capped)", len(seeds))
                    return seeds
    _LOG.info("[ontology] multi-hop: %d 2-hop seeds walked", len(seeds))
    return seeds
