"""Span identity policies: WHEN do a graph evidence span and a vector chunk name one candidate?

Fusion can only reward cross-lane agreement when both lanes point at the SAME candidate. Under the
`exact` policy that means identical `(doc_id, char_start, char_end)` boundaries -- but a graph
mention is a few dozen characters cut around an entity, while a vector chunk is an ~800-character
recursive window, so the two lanes essentially never agree by construction and RRF degenerates into
two disjoint rankings competing for the same seats.

The `overlap` policy folds a graph span into the vector chunk that CONTAINS it (and, for spans no
chunk covers, into whichever graph span it mutually overlaps), so the pair becomes one candidate
both lanes vouch for.

Two invariants keep the rule safe for span-level scoring:

- **Vector chunks are never merged with each other.** Consecutive recursive chunks share their
  `chunk_overlap` tail, so a transitive union would chain a whole document into one candidate. Only
  the vector lane creates anchors; the graph lane joins them.
- **The surviving record is always one of the input records, verbatim.** A merge never synthesizes
  a union span, so the fused chunk's text stays an exact corpus slice at its own offsets and
  `recall@k` / `MRR` score the same rule they always did.

How much overlap a fold requires is `SPAN_MERGE_MIN_RATIO`, exposed as
`graph_fusion_span_merge_ratio`. It was swept and pinned at two chunk scales: on an ~800-character
chunking a graph mention is either wholly inside a chunk or misses it, so 0.25 / 0.5 / 0.75 score
identically; at `size=200`, where a chunk and a mention are the same order of magnitude, the
threshold re-decides real merges yet still moves one headline metric in one row -- and that row
prefers the pinned value to a stricter one. See the GraphRAG span merge-threshold evidence.
"""

from typing import NamedTuple

from typing_extensions import TypedDict

from llb.core.contracts.rag import ChunkRecord

SPAN_IDENTITY_EXACT = "exact"
SPAN_IDENTITY_OVERLAP = "overlap"
SPAN_IDENTITIES = (SPAN_IDENTITY_EXACT, SPAN_IDENTITY_OVERLAP)
# `exact` stays the default until a measured sweep says otherwise.
DEFAULT_SPAN_IDENTITY = SPAN_IDENTITY_EXACT

# Share of the SHORTER span that the intersection must cover before two spans are one candidate.
# Containment scores 1.0, a graph mention clipped by a chunk boundary scores the covered share, and
# an incidental one-character touch between neighbouring chunks scores ~0 and stays separate.
# The default is the swept-and-pinned value; 1.0 is the strictest setting (containment only) and a
# lower value admits more clipped mentions. Zero is refused: it would merge on a bare touch.
SPAN_MERGE_MIN_RATIO = 0.5

LANE_VECTOR = "vector"
LANE_GRAPH = "graph"

SpanKey = tuple[str, int, int]


class MergedSpan(TypedDict):
    """One span folded into a surviving candidate, recorded so a merge stays auditable."""

    lane: str
    doc_id: str
    char_start: int
    char_end: int


class LaneCandidates(NamedTuple):
    """The candidate set both lanes are ranked over, plus what each candidate is made of."""

    records: dict[SpanKey, ChunkRecord]
    """Surviving record per candidate -- always an input record, offsets untouched."""
    rankings: list[list[SpanKey]]
    """`[vector, graph]` rankings as candidate keys, de-duplicated in lane order."""
    lanes: dict[SpanKey, list[str]]
    """Which lanes vouch for each candidate, in `[vector, graph]` order."""
    merged: dict[SpanKey, list[MergedSpan]]
    """Spans folded into each candidate (empty under the `exact` policy)."""

    def shared(self) -> list[SpanKey]:
        """Candidates BOTH lanes returned -- the agreement RRF is designed to exploit."""
        return [key for key, lanes in self.lanes.items() if len(lanes) > 1]


def span_key(chunk: ChunkRecord) -> SpanKey:
    """Stable identity shared by vector chunks and graph evidence records."""
    return (chunk["doc_id"], chunk["char_start"], chunk["char_end"])


def resolve_span_identity(identity: str | None) -> str:
    """Validate a policy name, mapping `None` to the default."""
    resolved = identity or DEFAULT_SPAN_IDENTITY
    if resolved not in SPAN_IDENTITIES:
        raise ValueError(
            f"span identity must be one of {', '.join(SPAN_IDENTITIES)}, got {identity!r}"
        )
    return resolved


def merges_spans(identity: str) -> bool:
    """Whether a policy can fold one span into another -- the only case the ratio applies to.

    Under `exact` two spans are one candidate or they are not; there is no partial overlap to
    threshold, so a merge-ratio grid would report one ranking under several labels.
    """
    return resolve_span_identity(identity) != SPAN_IDENTITY_EXACT


def resolve_merge_ratio(ratio: float | None) -> float:
    """Validate a merge threshold, mapping `None` to the default."""
    resolved = SPAN_MERGE_MIN_RATIO if ratio is None else float(ratio)
    if not 0.0 < resolved <= 1.0:
        raise ValueError(f"span merge ratio must be within (0, 1], got {ratio!r}")
    return resolved


def overlap_ratio(left: SpanKey, right: SpanKey) -> float:
    """Intersection as a share of the shorter span; 0.0 across documents or on a bare touch."""
    if left[0] != right[0]:
        return 0.0
    intersection = min(left[2], right[2]) - max(left[1], right[1])
    if intersection <= 0:
        return 0.0
    shortest = min(left[2] - left[1], right[2] - right[1])
    return intersection / shortest if shortest > 0 else 0.0


def lane_candidates(
    vector_hits: list[ChunkRecord],
    graph_hits: list[ChunkRecord],
    identity: str = DEFAULT_SPAN_IDENTITY,
    merge_ratio: float = SPAN_MERGE_MIN_RATIO,
) -> LaneCandidates:
    """Map both lane rankings onto one candidate set under the selected span-identity policy."""
    if not merges_spans(identity):
        return _exact_candidates(vector_hits, graph_hits)
    return _overlap_candidates(vector_hits, graph_hits, resolve_merge_ratio(merge_ratio))


def _exact_candidates(
    vector_hits: list[ChunkRecord], graph_hits: list[ChunkRecord]
) -> LaneCandidates:
    """Identity is the exact `(doc_id, char_start, char_end)` triple; the vector record survives."""
    builder = _CandidateSet()
    for hit in vector_hits:
        builder.add(span_key(hit), hit, LANE_VECTOR)
    for hit in graph_hits:
        builder.add(span_key(hit), hit, LANE_GRAPH)
    return builder.result()


def _overlap_candidates(
    vector_hits: list[ChunkRecord], graph_hits: list[ChunkRecord], merge_ratio: float
) -> LaneCandidates:
    """Fold each graph span into the vector chunk that covers it, else into a graph sibling."""
    builder = _CandidateSet()
    for hit in vector_hits:
        builder.add(span_key(hit), hit, LANE_VECTOR)
    anchors = list(builder.keys)
    for hit in graph_hits:
        key = span_key(hit)
        host = key if key in builder.keys else _best_host(key, anchors, merge_ratio)
        if host is None:
            builder.add(key, hit, LANE_GRAPH)
            anchors.append(key)
            continue
        builder.add(host, hit, LANE_GRAPH, folded=key)
    return builder.result()


def _best_host(key: SpanKey, anchors: list[SpanKey], merge_ratio: float) -> SpanKey | None:
    """The anchor this span belongs to: strongest overlap, then the earlier-ranked anchor."""
    best: SpanKey | None = None
    best_ratio = 0.0
    for anchor in anchors:
        ratio = overlap_ratio(key, anchor)
        # Strict `>` keeps the earliest (best-ranked) anchor on a tie, so a span sitting in the
        # shared tail of two consecutive chunks joins the one the vector lane ranked higher.
        if ratio >= merge_ratio and ratio > best_ratio:
            best, best_ratio = anchor, ratio
    return best


class _CandidateSet:
    """Accumulates candidates in first-seen order and records lane membership per candidate."""

    def __init__(self) -> None:
        self.keys: list[SpanKey] = []
        self._records: dict[SpanKey, ChunkRecord] = {}
        self._lanes: dict[SpanKey, list[str]] = {}
        self._merged: dict[SpanKey, list[MergedSpan]] = {}
        self._rankings: dict[str, list[SpanKey]] = {LANE_VECTOR: [], LANE_GRAPH: []}

    def add(
        self, key: SpanKey, record: ChunkRecord, lane: str, folded: SpanKey | None = None
    ) -> None:
        """Attribute one lane hit to candidate `key`, creating the candidate on first sight."""
        if key not in self._records:
            self.keys.append(key)
            self._records[key] = record
            self._lanes[key] = []
            self._merged[key] = []
        if lane not in self._lanes[key]:
            self._lanes[key].append(lane)
        if key not in self._rankings[lane]:
            self._rankings[lane].append(key)
        if folded is not None:
            self._merged[key].append(
                {
                    "lane": lane,
                    "doc_id": folded[0],
                    "char_start": folded[1],
                    "char_end": folded[2],
                }
            )

    def result(self) -> LaneCandidates:
        return LaneCandidates(
            records=self._records,
            rankings=[self._rankings[LANE_VECTOR], self._rankings[LANE_GRAPH]],
            lanes=self._lanes,
            merged={key: spans for key, spans in self._merged.items() if spans},
        )
