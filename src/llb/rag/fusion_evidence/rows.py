"""Build the compared row set of a graph-weight sweep, retrieving each lane exactly once.

A sweep over `w` graph weights and `s` graph strategies would otherwise hit FAISS `w*s` times and
DuckDB `w*s` times per question, even though neither lane's ranking depends on the weight. These
wrappers cache each lane's candidates per question and re-fuse the SAME candidates at every weight
through the production `fuse_lane_hits`, so the sweep costs `1 + s` retrieval passes and still
scores exactly what `FusedRetriever` would return.

A candidate-depth sweep rides the same cache: each lane is retrieved once at the DEEPEST compared
depth and every shallower fused row slices that one ranking, because a lane's top-d truncated to
d' < d is exactly its top-d' (every lane ranks by a total order that does not depend on the
requested depth).
"""

from llb.core.contracts.rag import ChunkRecord
from llb.rag.fusion import fuse_lane_hits, lane_depth
from llb.rag.fusion_evidence.models import (
    FUSED_ROW_TEMPLATE,
    GRAPH_ROW_PREFIX,
    VECTOR_ROW,
    Retriever,
)

DEFAULT_GRAPH_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
# `None` == the historical single depth: ask each lane for exactly the sweep's `k`.
DEFAULT_GRAPH_CANDIDATES: tuple[int | None, ...] = (None,)


class LaneCache:
    """One lane's top-`depth` per question, retrieved once at the sweep's deepest pool.

    Requests for a larger depth than the cache was built with return the cached (shorter) list
    rather than silently re-querying, so a sweep can never mix depths across rows.
    """

    def __init__(self, store: Retriever, questions: list[str], depth: int) -> None:
        self.depth = depth
        self._hits: dict[str, list[ChunkRecord]] = {
            question: store.retrieve(question, depth) for question in dict.fromkeys(questions)
        }

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        return self._hits.get(question, [])[:k]


class FusedReplay:
    """A fused row at one (graph weight, candidate depth), fusing two cached lanes.

    `depth` is the per-lane candidate pool the weight is applied over; the fused ranking is then
    cut to the scored `k`, exactly as `FusedRetriever` does at query time.
    """

    def __init__(
        self, vector: LaneCache, graph: LaneCache, graph_weight: float, depth: int | None = None
    ) -> None:
        if not 0.0 <= graph_weight <= 1.0:
            raise ValueError(f"graph weight must be within [0, 1], got {graph_weight}")
        if depth is not None and depth < 1:
            raise ValueError(f"fusion candidate depth must be at least 1, got {depth}")
        self.vector = vector
        self.graph = graph
        self.graph_weight = graph_weight
        self.depth = depth

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        depth = lane_depth(self.depth, k)
        return fuse_lane_hits(
            self.vector.retrieve(question, depth),
            self.graph.retrieve(question, depth),
            self.graph_weight,
            k,
        )


def build_sweep_rows(
    vector: Retriever,
    graphs: dict[str, Retriever],
    questions: list[str],
    k: int,
    weights: tuple[float, ...] = DEFAULT_GRAPH_WEIGHTS,
    candidates: tuple[int | None, ...] = DEFAULT_GRAPH_CANDIDATES,
) -> dict[str, Retriever]:
    """`vector` + a row per graph strategy + a fused row per (strategy, weight, candidate depth).

    Depths are resolved against `k` first (a request below `k` is lifted to `k`) and then
    de-duplicated, so two requested depths that resolve to the same pool produce one row rather
    than two identical ones under different labels.
    """
    depths = tuple(dict.fromkeys(lane_depth(depth, k) for depth in candidates)) or (k,)
    cache_depth = max(depths)
    vector_cache = LaneCache(vector, questions, cache_depth)
    rows: dict[str, Retriever] = {VECTOR_ROW: vector_cache}
    for strategy, store in graphs.items():
        graph_cache = LaneCache(store, questions, cache_depth)
        rows[f"{GRAPH_ROW_PREFIX}{strategy}"] = graph_cache
        for weight in weights:
            # An endpoint weight is a single-lane passthrough, so a deeper pool cannot change it;
            # emitting one row per depth there would report the same ranking several times.
            for depth in depths if 0.0 < weight < 1.0 else (k,):
                label = FUSED_ROW_TEMPLATE.format(strategy=strategy, weight=weight, depth=depth)
                rows[label] = FusedReplay(vector_cache, graph_cache, weight, depth)
    return rows


def parse_candidates(spec: str) -> tuple[int | None, ...]:
    """Parse a `10,50` candidate-depth grid; raises `ValueError` on a non-positive entry.

    `k` names the sweep's own cutoff (the historical depth), so `k,50` compares the shallow pool
    against a deeper one without the caller having to repeat the `--k` value.
    """
    depths: list[int | None] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "k":
            depths.append(None)
            continue
        try:
            depth = int(token)
        except ValueError:
            raise ValueError(f"candidate depth must be an integer or 'k', got {token!r}") from None
        if depth < 1:
            raise ValueError(f"fusion candidate depth must be at least 1, got {depth}")
        depths.append(depth)
    if not depths:
        raise ValueError("no candidate depth parsed from the grid spec")
    return tuple(dict.fromkeys(depths))


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
