"""Source-span audit for external RAG answer logs (external-rag-source-mapping).

`score-external-rag` scores ANSWER text; the provider's returned source records stay opaque
context because their ids live in the provider's namespace, not the benchmark corpus. This
module joins them: an operator-supplied mapping sidecar translates provider `article_id`,
`url`, or `article_title` keys into corpus `doc_id` plus an optional character range, and the
audit then scores retrieval evidence with the SAME source-span metric local retrieval uses
(`llb.rag.retrieval`): a mapped source is a hit when its span overlaps the item's gold spans.

Evidence strength is explicit: a mapping that carries `char_start`/`char_end` supports a
span-overlap hit (strong, counted into recall@3 / MRR); a mapping with only a `doc_id`
supports a document-level match (WEAK, flagged and never counted as span proof); a returned
source with no mapping at all is reported separately as unmapped -- an audit gap, not a
retrieval miss.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.rag.retrieval import first_hit_rank
from llb.scoring.external_rag_source_map import SOURCE_RECALL_K, SourceMap, map_source

# Mapping-key precedence: an exact provider id beats a URL beats a title. Title keys are the
# most collision-prone, but with spans present the hit is still span-proof.

# recall@k over mapped source hits mirrors the local retrieval report's headline depth.

# Additive CSV columns the audit contributes (appended after the flattened source columns).


def _gold_span_records(gold_spans: Sequence[dict[str, Any]]) -> list[SourceSpanRecord]:
    return [
        {
            "doc_id": str(span.get("doc_id") or ""),
            "char_start": int(span.get("char_start") or 0),
            "char_end": int(span.get("char_end") or 0),
            "text": str(span.get("text") or ""),
        }
        for span in gold_spans
        if isinstance(span, dict)
    ]


@dataclass
class _MappedSources:
    """The chunk view of one row's returned sources plus mapping/weak-hit bookkeeping."""

    chunks: list[ChunkRecord]
    weak_rank: int | None
    mapped: int
    unmapped: int


def _map_row_sources(
    sources: Sequence[dict[str, Any]], source_map: SourceMap, gold_docs: set[str]
) -> _MappedSources:
    """Translate returned sources into rank-preserving chunks; track weak doc-level matches."""
    chunks: list[ChunkRecord] = []
    weak_rank: int | None = None
    mapped = unmapped = 0
    for position, source in enumerate(sources, 1):
        entry = map_source(source, source_map)
        if entry is None:
            unmapped += 1
            chunks.append(_MISS_CHUNK)
            continue
        mapped += 1
        if entry.has_span:
            chunks.append(
                {
                    "doc_id": entry.doc_id,
                    "char_start": int(entry.char_start or 0),
                    "char_end": int(entry.char_end or 0),
                    "text": "",
                }
            )
        else:
            chunks.append(_MISS_CHUNK)
            if weak_rank is None and entry.doc_id in gold_docs:
                weak_rank = position
    return _MappedSources(chunks=chunks, weak_rank=weak_rank, mapped=mapped, unmapped=unmapped)


def _hit_outcome(strong_rank: int | None, weak_rank: int | None) -> tuple[float, int | None, bool]:
    """(hit, rank, weak): a strong span hit wins; a weak doc-level match is flagged as weak."""
    if strong_rank is not None:
        return 1.0, strong_rank, False
    if weak_rank is not None:
        return 1.0, weak_rank, True
    return 0.0, None, False


def audit_row_sources(
    sources: Sequence[dict[str, Any]],
    gold_spans: Sequence[dict[str, Any]],
    source_map: SourceMap,
) -> dict[str, object]:
    """Audit one row's returned sources against its gold spans; returns the CSV column values.

    A STRONG hit is a mapped source whose char range overlaps a gold span (rank in returned
    order, the same `first_hit_rank` local retrieval uses). A WEAK hit is a span-less mapping
    whose doc matches a gold span's doc -- reported with `source_hit_weak=true` and never
    counted as span proof. Unmapped sources count separately.
    """
    spans = _gold_span_records(gold_spans)
    resolved = _map_row_sources(sources, source_map, {span["doc_id"] for span in spans})
    strong_rank = first_hit_rank(resolved.chunks, spans) if spans else None
    hit, rank, weak = _hit_outcome(strong_rank, resolved.weak_rank)
    return {
        "source_hit": hit if sources else "",
        "source_first_hit_rank": rank if rank is not None else "",
        "source_hit_weak": ("true" if weak else "false") if sources else "",
        "source_mapped_count": resolved.mapped,
        "source_unmapped_count": resolved.unmapped,
        "_source_strong_rank": strong_rank,  # summary-only; stripped from the CSV
    }


# A placeholder that can never overlap a gold span, holding the returned-order position.
_MISS_CHUNK: ChunkRecord = {"doc_id": "", "char_start": 0, "char_end": 0, "text": ""}


def summarize_source_audit(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Aggregate the audit over rows that returned at least one source.

    recall@3 and MRR count STRONG (span-proof) hits only, matching local retrieval's metric;
    weak hits and unmapped sources are reported beside them, never folded in.
    """
    audited = [row for row in rows if row.get("source_hit") != ""]
    strong_ranks = [row.get("_source_strong_rank") for row in audited]
    n = len(audited)
    recall = (
        sum(1 for r in strong_ranks if isinstance(r, int) and r <= SOURCE_RECALL_K) / n
        if n
        else 0.0
    )
    mrr = sum(1.0 / r for r in strong_ranks if isinstance(r, int)) / n if n else 0.0
    weak_hits = sum(1 for row in audited if row.get("source_hit_weak") == "true")
    total_mapped = sum(int(str(row.get("source_mapped_count") or 0)) for row in audited)
    total_unmapped = sum(int(str(row.get("source_unmapped_count") or 0)) for row in audited)
    total_sources = total_mapped + total_unmapped
    return {
        "rows_audited": n,
        "source_recall_at_3": round(recall, 4),
        "source_mrr": round(mrr, 4),
        "weak_hit_rows": weak_hits,
        "mapped_sources": total_mapped,
        "unmapped_sources": total_unmapped,
        "unmapped_rate": round(total_unmapped / total_sources, 4) if total_sources else 0.0,
    }
