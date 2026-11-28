"""Read paired graph/vector retrieval on the same verified book evidence."""

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.executor.cases import spans_as_dicts
from llb.goldset.schema import GoldItem
from llb.graph.book_inputs import BookSources
from llb.rag.book_comparison.inputs import pages_at_span
from llb.rag.book_comparison.metrics import ReadingSettings, first_page_rank, paired_metric_reading
from llb.rag.cited_corpus import CitedDocument
from llb.rag.comparison.models import Retriever
from llb.rag.duplicates.collapse import occurrence_spans
from llb.rag.fusion_evidence.models import METRIC_ALL_SPANS, VECTOR_ROW
from llb.rag.fusion_evidence.stats import (
    bootstrap_index_sets,
    bootstrap_interval,
)
from llb.rag.retrieval import all_spans_at_k, first_hit_rank, span_coverage_at_k, spans_overlap

TECHNICAL = "technical_evidence"
TEXT = "text_only"
PRIMARY_METRIC = METRIC_ALL_SPANS


@dataclass(frozen=True)
class GraphQuestion:
    item_id: str
    question: str
    spans: list[SourceSpanRecord]
    pages: frozenset[int]
    technical_kinds: tuple[str, ...]


def book_graph_questions(items: list[GoldItem], sources: BookSources) -> list[GraphQuestion]:
    def labelled(item: GoldItem) -> GraphQuestion:
        pages: set[int] = set()
        kinds: set[str] = set()
        for span in item.source_spans:
            doc = sources.documents[span.doc_id]
            pages.update(pages_at_span(doc.source, span.char_start, span.char_end))
            kinds.update(
                block.kind
                for block in doc.references.technical_blocks
                if spans_overlap(
                    span.char_start, span.char_end, block.span.char_start, block.span.char_end
                )
            )
        return GraphQuestion(
            item.id, item.question, spans_as_dicts(item), frozenset(pages), tuple(sorted(kinds))
        )

    return [labelled(item) for item in items]


@dataclass(frozen=True)
class ItemReading:
    """Scored ranks plus the retrieved spans and scores needed to replay them."""

    item: GraphQuestion
    rows: dict[str, dict[str, float]]
    hits: dict[str, list[list[SourceSpanRecord]]]
    scores: dict[str, list[float | None]]


def read_book_graph(
    lanes: dict[str, Retriever], items: list[GraphQuestion], docs: dict[str, CitedDocument], k: int
) -> list[ItemReading]:
    readings = []
    for item in items:
        hits = {name: store.retrieve(item.question, k)[:k] for name, store in lanes.items()}
        readings.append(
            ItemReading(
                item,
                {name: _scores(chunks, item, docs, k) for name, chunks in hits.items()},
                {
                    name: [occurrence_spans(chunk) for chunk in chunks]
                    for name, chunks in hits.items()
                },
                {
                    name: [chunk.get("retrieval_score") for chunk in chunks]
                    for name, chunks in hits.items()
                },
            )
        )
    return readings


def _scores(
    chunks: list[ChunkRecord], item: GraphQuestion, docs: dict[str, CitedDocument], k: int
) -> dict[str, float]:
    span = first_hit_rank(chunks, item.spans)
    page = first_page_rank(chunks, item.pages, docs)
    return {
        "span_recall_at_k": float(span is not None),
        "span_mrr": 1.0 / span if span else 0.0,
        "page_recall_at_k": float(page is not None),
        "page_mrr": 1.0 / page if page else 0.0,
        METRIC_ALL_SPANS: all_spans_at_k(chunks, item.spans, k),
        "span_coverage": span_coverage_at_k(chunks, item.spans, k),
    }


def graph_slices(items: list[ItemReading]) -> dict[str, list[ItemReading]]:
    return {
        "all": items,
        TECHNICAL: [item for item in items if item.item.technical_kinds],
        TEXT: [item for item in items if not item.item.technical_kinds],
    }


def summarize_graph_slices(
    groups: dict[str, list[ItemReading]], settings: ReadingSettings
) -> dict[str, dict[str, object]]:
    return {name: _read_slice(items, settings) for name, items in groups.items()}


def _vectors(items: Iterable[ItemReading], label: str) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {}
    for item in items:
        for name, value in item.rows[label].items():
            vectors.setdefault(name, []).append(value)
    return vectors


def _read_slice(items: list[ItemReading], settings: ReadingSettings) -> dict[str, object]:
    if not items:
        return {"n": 0, "rows": {}}
    draw = bootstrap_index_sets(len(items), settings.resamples, settings.seed)
    baseline = _vectors(items, VECTOR_ROW)
    rows = {
        label: {
            name: _read_metric(values, baseline[name], draw, settings, label)
            for name, values in _vectors(items, label).items()
        }
        for label in items[0].rows
    }
    return {"n": len(items), "rows": rows}


def _read_metric(
    values: list[float],
    baseline: list[float],
    draw: list[list[int]],
    settings: ReadingSettings,
    label: str,
) -> dict[str, object]:
    reading: dict[str, object] = {
        "interval": bootstrap_interval(values, draw, settings.confidence),
    }
    if label != VECTOR_ROW:
        reading.update(paired_metric_reading(values, baseline, draw, settings))
        reading["reaches_declared_mde"] = (
            abs(mean(values) - mean(baseline)) >= settings.minimum_detectable_delta
        )
    return reading


def replay_item(reading: ItemReading) -> dict[str, object]:
    return {
        "item_id": reading.item.item_id,
        "pdf_pages": sorted(reading.item.pages),
        "technical_kinds": list(reading.item.technical_kinds),
        "rows": reading.rows,
        "retrieved": {
            label: [
                {
                    "score": score,
                    "occurrences": [
                        {
                            "doc_id": span["doc_id"],
                            "char_start": span["char_start"],
                            "char_end": span["char_end"],
                        }
                        for span in hits
                    ],
                }
                for hits, score in zip(spans, reading.scores[label], strict=True)
            ]
            for label, spans in reading.hits.items()
        },
    }
