"""Build fixed and question-routed rows, retrieving each physical lane exactly once.

A sweep over `w` graph weights and `s` graph strategies would otherwise hit FAISS `w*s` times and
DuckDB `w*s` times per question, even though neither lane's ranking depends on the weight. These
wrappers cache each lane's candidates per question and re-fuse the SAME candidates at every weight
through the production `fuse_lane_hits`, so the sweep costs `1 + s` retrieval passes and still
scores exactly what `FusedRetriever` would return.

A candidate-depth sweep rides the same cache: each lane is retrieved once at the DEEPEST compared
depth and every shallower fused row slices that one ranking, because a lane's top-d truncated to
d' < d is exactly its top-d' (every lane ranks by a total order that does not depend on the
requested depth). A span-identity sweep -- and the merge-threshold sweep inside it -- rides it too:
the policy decides how the two cached rankings are MAPPED onto candidates, never what either lane
returns.
"""

from llb.core.contracts.rag import ChunkRecord
from llb.rag.fusion import fuse_lane_hits, lane_agreement, lane_depth
from llb.rag.fusion_evidence.models import (
    GRAPH_ROW_PREFIX,
    VECTOR_ROW,
    Retriever,
    fused_row_label,
    routed_row_label,
)
from llb.rag.fusion_routing import (
    DEFAULT_HEURISTIC_POLICY,
    HeuristicPolicy,
    QuestionTypeRouter,
    RoutingDecision,
)
from llb.rag.fusion_spans import (
    DEFAULT_SPAN_IDENTITY,
    SPAN_MERGE_MIN_RATIO,
    merges_spans,
    resolve_merge_ratio,
    resolve_span_identity,
)

DEFAULT_GRAPH_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
# `None` == the historical single depth: ask each lane for exactly the sweep's `k`.
DEFAULT_GRAPH_CANDIDATES: tuple[int | None, ...] = (None,)
# The historical identity rule stays the only swept policy until an operator asks for the other.
DEFAULT_SPAN_IDENTITIES: tuple[str, ...] = (DEFAULT_SPAN_IDENTITY,)
# Likewise the shipped merge threshold: a ratio grid only produces distinct rows under a folding
# identity policy, so the default grid is the one pinned value.
DEFAULT_SPAN_MERGE_RATIOS: tuple[float, ...] = (SPAN_MERGE_MIN_RATIO,)


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
        self,
        vector: LaneCache,
        graph: LaneCache,
        graph_weight: float,
        depth: int | None = None,
        span_identity: str = DEFAULT_SPAN_IDENTITY,
        router: QuestionTypeRouter | None = None,
        merge_ratio: float = SPAN_MERGE_MIN_RATIO,
    ) -> None:
        if not 0.0 <= graph_weight <= 1.0:
            raise ValueError(f"graph weight must be within [0, 1], got {graph_weight}")
        if depth is not None and depth < 1:
            raise ValueError(f"fusion candidate depth must be at least 1, got {depth}")
        self.vector = vector
        self.graph = graph
        self.graph_weight = graph_weight
        self.depth = depth
        self.span_identity = resolve_span_identity(span_identity)
        self.merge_ratio = resolve_merge_ratio(merge_ratio)
        self.router = router

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        vector_hits, graph_hits = self._lane_hits(question, k)
        graph_weight = (
            self.router.graph_weight(question) if self.router is not None else self.graph_weight
        )
        return fuse_lane_hits(
            vector_hits,
            graph_hits,
            graph_weight,
            k,
            span_identity=self.span_identity,
            merge_ratio=self.merge_ratio,
        )

    def routing_decision(self, question: str) -> RoutingDecision | None:
        """Expose the route for the evidence report without coupling it to this concrete class."""
        return self.router.decide(question) if self.router is not None else None

    def lane_agreement(self, question: str, k: int) -> int:
        """Candidates BOTH lanes returned in this row's pool -- the sweep's agreement diagnostic."""
        return lane_agreement(*self._lane_hits(question, k), self.span_identity, self.merge_ratio)

    def _lane_hits(self, question: str, k: int) -> tuple[list[ChunkRecord], list[ChunkRecord]]:
        depth = lane_depth(self.depth, k)
        return self.vector.retrieve(question, depth), self.graph.retrieve(question, depth)


def build_sweep_rows(
    vector: Retriever,
    graphs: dict[str, Retriever],
    questions: list[str],
    k: int,
    weights: tuple[float, ...] = DEFAULT_GRAPH_WEIGHTS,
    candidates: tuple[int | None, ...] = DEFAULT_GRAPH_CANDIDATES,
    identities: tuple[str, ...] = DEFAULT_SPAN_IDENTITIES,
    routed_graph_weight: float | None = None,
    question_types: dict[str, str] | None = None,
    heuristic_policy: HeuristicPolicy = DEFAULT_HEURISTIC_POLICY,
    merge_ratios: tuple[float, ...] = DEFAULT_SPAN_MERGE_RATIOS,
) -> dict[str, Retriever]:
    """`vector` + a graph row per strategy + a fused row per (strategy, weight, depth, identity).

    Depths are resolved against `k` first (a request below `k` is lifted to `k`) and then
    de-duplicated, so two requested depths that resolve to the same pool produce one row rather
    than two identical ones under different labels.
    """
    depths = tuple(dict.fromkeys(lane_depth(depth, k) for depth in candidates)) or (k,)
    if routed_graph_weight is not None and not 0.0 <= routed_graph_weight <= 1.0:
        raise ValueError(f"graph weight must be within [0, 1], got {routed_graph_weight}")
    cache_depth = max(depths)
    vector_cache = LaneCache(vector, questions, cache_depth)
    rows: dict[str, Retriever] = {VECTOR_ROW: vector_cache}
    for strategy, store in graphs.items():
        graph_cache = LaneCache(store, questions, cache_depth)
        rows[f"{GRAPH_ROW_PREFIX}{strategy}"] = graph_cache
        for weight in weights:
            for depth, identity, ratio in _fusion_points(
                weight, depths, identities, merge_ratios, k
            ):
                label = fused_row_label(strategy, weight, depth, identity, ratio)
                rows[label] = FusedReplay(
                    vector_cache, graph_cache, weight, depth, identity, merge_ratio=ratio
                )
        if routed_graph_weight is not None:
            router = QuestionTypeRouter(routed_graph_weight, question_types, heuristic_policy)
            for depth, identity, ratio in _fusion_points(
                routed_graph_weight, depths, identities, merge_ratios, k, endpoints=False
            ):
                label = routed_row_label(strategy, routed_graph_weight, depth, identity, ratio)
                rows[label] = FusedReplay(
                    vector_cache,
                    graph_cache,
                    routed_graph_weight,
                    depth,
                    identity,
                    router,
                    ratio,
                )
    return rows


def _fusion_points(
    weight: float,
    depths: tuple[int, ...],
    identities: tuple[str, ...],
    merge_ratios: tuple[float, ...],
    k: int,
    endpoints: bool = True,
) -> list[tuple[int, str, float]]:
    """The (depth, identity, merge ratio) points worth a fused row at this graph weight.

    Two collapses keep the table free of rows that are the same ranking under another label:

    - An endpoint weight is a single-lane passthrough -- nothing is fused, so no fusion knob can
      change its ranking. (A routed row applies its weight per question, so it is never a pure
      passthrough even at an endpoint weight; `endpoints=False` keeps its full grid.)
    - The merge ratio is a parameter of a FOLDING identity policy. Under `exact` there is no
      partial overlap to threshold, so that policy takes the default ratio only.
    """
    if endpoints and not 0.0 < weight < 1.0:
        return [(k, DEFAULT_SPAN_IDENTITY, SPAN_MERGE_MIN_RATIO)]
    return [
        (depth, identity, ratio)
        for depth in depths
        for identity in identities
        for ratio in (merge_ratios if merges_spans(identity) else DEFAULT_SPAN_MERGE_RATIOS)
    ]
