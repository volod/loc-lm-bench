"""Retrieval-rank annotation for citation-valid needle items."""

from typing import Protocol, cast

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.goldset.schema import GoldItem
from llb.rag.retrieval import first_hit_rank


class NeedleRetriever(Protocol):
    """Minimal store interface needed to score drafted needles."""

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        """Return top-k retrieval hits for `question`."""


def _source_span_records(item: GoldItem) -> list[SourceSpanRecord]:
    return [cast(SourceSpanRecord, span.model_dump()) for span in item.source_spans]


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def annotate_needle_retrieval(
    items: list[GoldItem],
    retriever: NeedleRetriever,
    *,
    k: int,
    drop_nonretrievable: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Annotate needle rows with their first source-span retrieval rank.

    A row is retrieval-unique when the full-corpus retriever returns a chunk overlapping one of
    the row's gold source spans within top-k. Misses keep `retrieval_rank: null` unless the caller
    explicitly opts into dropping them from the written review artifact.
    """
    if k < 1:
        raise ValueError("retrieval k must be >= 1")

    rows: list[dict[str, object]] = []
    retrievable = 0
    missed_ids: list[str] = []

    for item in items:
        hits = retriever.retrieve(item.question, k)
        rank = first_hit_rank(hits[:k], _source_span_records(item))
        if rank is None:
            missed_ids.append(item.id)
        else:
            retrievable += 1

        row = cast(dict[str, object], item.model_dump())
        row["retrieval_rank"] = rank
        row["retrieval_k"] = k
        if rank is not None or not drop_nonretrievable:
            rows.append(row)

    report: dict[str, object] = {
        "enabled": True,
        "k": k,
        "items": len(items),
        "retrievable_items": retrievable,
        "missed_items": len(items) - retrievable,
        "retrievable_fraction": _ratio(retrievable, len(items)),
        "dropped_items": len(items) - len(rows),
        "missed_ids": missed_ids,
    }
    return rows, report
