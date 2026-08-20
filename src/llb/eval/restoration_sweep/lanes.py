"""Lane readings, the retrieval cache, and the result container for the restoration sweep.

A LANE here is anything the sweep measures on the same items: a reference lane (the clean question,
the untouched noisy one, normalization alone) or one swept setting of the restoration constants.
All of them carry the same per-item hit and reciprocal-rank vectors, which is what lets a setting be
compared with the default PAIRED rather than as two pooled averages.
"""

from dataclasses import dataclass, field
from typing import Any

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.eval.restoration_sweep.audit import AuditCounts, EditRecord
from llb.rag import retrieval
from llb.rag.query_prep.base import QueryPrepResult
from llb.rag.query_prep.restore_policy import RestorationPolicy
from llb.rag.query_prep.retrieval import retrieve_prepared


@dataclass(frozen=True)
class LaneReading:
    """One lane (a reference lane or a swept setting) measured on one noise class."""

    lane: str
    variant_class: str
    hits: tuple[float, ...]
    reciprocal_ranks: tuple[float, ...]
    counts: AuditCounts = field(default_factory=AuditCounts)

    @property
    def n(self) -> int:
        return len(self.hits)

    @property
    def recall_at_k(self) -> float:
        return sum(self.hits) / len(self.hits) if self.hits else 0.0

    def as_row(self) -> dict[str, object]:
        """This lane's metrics for one class (or `pooled`), audit tallies included."""
        return {
            "variant_class": self.variant_class,
            "n": self.n,
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            **self.counts.as_row(),
        }

    @property
    def mrr(self) -> float:
        return (
            sum(self.reciprocal_ranks) / len(self.reciprocal_ranks)
            if self.reciprocal_ranks
            else 0.0
        )


@dataclass(frozen=True)
class SweepResult:
    """Every lane reading, the per-edit audit rows, and what the run was measured over."""

    policies: tuple[RestorationPolicy, ...]
    variant_classes: tuple[str, ...]
    settings: tuple[LaneReading, ...]
    references: tuple[LaneReading, ...]
    edits: tuple[EditRecord, ...]
    item_ids: tuple[str, ...]
    top_k: int

    def reading(self, lane: str, variant_class: str) -> LaneReading | None:
        for row in (*self.settings, *self.references):
            if row.lane == lane and row.variant_class == variant_class:
                return row
        return None

    def pooled(self, lane: str) -> LaneReading:
        """One lane's readings concatenated across noise classes, in the swept class order."""
        rows = [self.reading(lane, name) for name in self.variant_classes]
        present = [row for row in rows if row is not None]
        counts = AuditCounts()
        for row in present:
            counts = counts + row.counts
        return LaneReading(
            lane=lane,
            variant_class="pooled",
            hits=tuple(value for row in present for value in row.hits),
            reciprocal_ranks=tuple(value for row in present for value in row.reciprocal_ranks),
            counts=counts,
        )

    def metric_rows(self) -> list[dict[str, object]]:
        """Every lane's metrics as flat JSONL rows: one per class, plus its pooled row."""
        rows: list[dict[str, object]] = []
        for policy in self.policies:
            for reading in self.lane_readings(policy.label):
                rows.append(
                    {
                        "setting": policy.label,
                        **policy.as_metadata(),
                        **reading.as_row(),
                    }
                )
        for lane in dict.fromkeys(reading.lane for reading in self.references):
            rows.extend(
                {"setting": lane, **reading.as_row()} for reading in self.lane_readings(lane)
            )
        return rows

    def lane_readings(self, lane: str) -> list[LaneReading]:
        """One lane's per-class readings in the swept class order, then its pooled reading."""
        readings = [
            reading
            for name in self.variant_classes
            if (reading := self.reading(lane, name)) is not None
        ]
        return [*readings, self.pooled(lane)]


class RetrievalCache:
    """Memoize the store call per (dense, lexical) query pair.

    Two settings differ on a handful of tokens across a whole split, so most prepared queries are
    byte-identical between settings; without this the sweep would re-encode each of them once per
    setting for an identical ranking.
    """

    def __init__(self, store: Any, top_k: int) -> None:
        self.store = store
        self.top_k = top_k
        self.cache: dict[tuple[str, str], list[ChunkRecord]] = {}
        self.calls = 0

    def retrieve(self, prepared: QueryPrepResult) -> list[ChunkRecord]:
        key = (prepared.dense_query, prepared.processed)
        hit = self.cache.get(key)
        if hit is None:
            self.calls += 1
            hit = retrieve_prepared(self.store, prepared, self.top_k)
            self.cache[key] = hit
        return hit


@dataclass
class LaneAccumulator:
    """Per-class hit/rank vectors and audit tallies for one lane, in item order."""

    hits: dict[str, list[float]] = field(default_factory=dict)
    ranks: dict[str, list[float]] = field(default_factory=dict)
    counts: dict[str, AuditCounts] = field(default_factory=dict)

    def add(
        self,
        variant_class: str,
        hit: float,
        reciprocal_rank: float,
        counts: AuditCounts | None = None,
    ) -> None:
        self.hits.setdefault(variant_class, []).append(hit)
        self.ranks.setdefault(variant_class, []).append(reciprocal_rank)
        self.counts[variant_class] = self.counts.get(variant_class, AuditCounts()) + (
            counts or AuditCounts()
        )

    def readings(self, lane: str) -> list[LaneReading]:
        return [
            LaneReading(
                lane=lane,
                variant_class=variant_class,
                hits=tuple(values),
                reciprocal_ranks=tuple(self.ranks[variant_class]),
                counts=self.counts[variant_class],
            )
            for variant_class, values in self.hits.items()
        ]


def score_prepared(
    cache: RetrievalCache,
    prepared: QueryPrepResult,
    spans: list[SourceSpanRecord],
) -> tuple[float, float]:
    """One lane's (hit, reciprocal rank) for one item, both paired by construction."""
    chunks = cache.retrieve(prepared)
    rank = retrieval.first_hit_rank(chunks, spans)
    return (1.0 if rank is not None else 0.0, 1.0 / rank if rank else 0.0)
