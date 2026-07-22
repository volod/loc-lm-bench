"""Build the compared row set of a graph-weight sweep, retrieving each lane exactly once.

A sweep over `w` graph weights and `s` graph strategies would otherwise hit FAISS `w*s` times and
DuckDB `w*s` times per question, even though neither lane's ranking depends on the weight. These
wrappers cache each lane's top-k per question and re-fuse the SAME candidates at every weight
through the production `fuse_lane_hits`, so the sweep costs `1 + s` retrieval passes and still
scores exactly what `FusedRetriever` would return.
"""

from llb.core.contracts.rag import ChunkRecord
from llb.rag.fusion import fuse_lane_hits
from llb.rag.fusion_evidence.models import (
    FUSED_ROW_TEMPLATE,
    GRAPH_ROW_PREFIX,
    VECTOR_ROW,
    Retriever,
)

DEFAULT_GRAPH_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)


class LaneCache:
    """One lane's top-k per question, retrieved once at the sweep's `k`.

    Requests for a larger k than the cache was built with return the cached (shorter) list rather
    than silently re-querying, so a sweep can never mix depths across rows.
    """

    def __init__(self, store: Retriever, questions: list[str], k: int) -> None:
        self.k = k
        self._hits: dict[str, list[ChunkRecord]] = {
            question: store.retrieve(question, k) for question in dict.fromkeys(questions)
        }

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        return self._hits.get(question, [])[:k]


class FusedReplay:
    """A fused row at one graph weight, fusing two cached lanes with the production rule."""

    def __init__(self, vector: LaneCache, graph: LaneCache, graph_weight: float) -> None:
        if not 0.0 <= graph_weight <= 1.0:
            raise ValueError(f"graph weight must be within [0, 1], got {graph_weight}")
        self.vector = vector
        self.graph = graph
        self.graph_weight = graph_weight

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        return fuse_lane_hits(
            self.vector.retrieve(question, k),
            self.graph.retrieve(question, k),
            self.graph_weight,
            k,
        )


def build_sweep_rows(
    vector: Retriever,
    graphs: dict[str, Retriever],
    questions: list[str],
    k: int,
    weights: tuple[float, ...] = DEFAULT_GRAPH_WEIGHTS,
) -> dict[str, Retriever]:
    """`vector` + one row per graph strategy + one fused row per (strategy, graph weight)."""
    vector_cache = LaneCache(vector, questions, k)
    rows: dict[str, Retriever] = {VECTOR_ROW: vector_cache}
    for strategy, store in graphs.items():
        graph_cache = LaneCache(store, questions, k)
        rows[f"{GRAPH_ROW_PREFIX}{strategy}"] = graph_cache
        for weight in weights:
            label = FUSED_ROW_TEMPLATE.format(strategy=strategy, weight=weight)
            rows[label] = FusedReplay(vector_cache, graph_cache, weight)
    return rows


def parse_weights(spec: str) -> tuple[float, ...]:
    """Parse a `0,0.3,1` graph-weight grid; raises `ValueError` on an out-of-range entry."""
    weights = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        weight = float(token)
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"graph weight must be within [0, 1], got {weight}")
        weights.append(weight)
    if not weights:
        raise ValueError("no graph weight parsed from the grid spec")
    return tuple(dict.fromkeys(weights))
