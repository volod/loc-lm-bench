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

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.core.contracts import ChunkRecord, SourceSpanRecord
from llb.rag.retrieval import first_hit_rank

# Mapping-key precedence: an exact provider id beats a URL beats a title. Title keys are the
# most collision-prone, but with spans present the hit is still span-proof.
KEY_ARTICLE_ID = "article_id"
KEY_URL = "url"
KEY_TITLE = "article_title"

# recall@k over mapped source hits mirrors the local retrieval report's headline depth.
SOURCE_RECALL_K = 3

# Additive CSV columns the audit contributes (appended after the flattened source columns).
SOURCE_AUDIT_COLUMNS = [
    "source_hit",
    "source_first_hit_rank",
    "source_hit_weak",
    "source_mapped_count",
    "source_unmapped_count",
]


@dataclass(frozen=True)
class SourceMapEntry:
    """One provider-source -> corpus location mapping."""

    doc_id: str
    char_start: int | None = None
    char_end: int | None = None

    @property
    def has_span(self) -> bool:
        return self.char_start is not None and self.char_end is not None


@dataclass(frozen=True)
class SourceMap:
    """Mapping indexes by provider key kind (see `KEY_*` precedence)."""

    by_article_id: dict[str, SourceMapEntry]
    by_url: dict[str, SourceMapEntry]
    by_title: dict[str, SourceMapEntry]

    def __len__(self) -> int:
        return len(self.by_article_id) + len(self.by_url) + len(self.by_title)


def load_source_map(path: Path | str) -> SourceMap:
    """Load a mapping sidecar (.json list, .jsonl, or .csv) into keyed indexes.

    Each record needs `doc_id` plus at least one provider key (`article_id`, `url`,
    `article_title`); `char_start`/`char_end` are optional and enable span-proof hits.
    """
    path = Path(path)
    rows = _read_mapping_rows(path)
    by_id: dict[str, SourceMapEntry] = {}
    by_url: dict[str, SourceMapEntry] = {}
    by_title: dict[str, SourceMapEntry] = {}
    for index, row in enumerate(rows, 1):
        doc_id = str(row.get("doc_id") or "").strip()
        if not doc_id:
            raise ValueError(f"{path}: mapping record {index} lacks doc_id")
        entry = SourceMapEntry(
            doc_id=doc_id,
            char_start=_int_or_none(row.get("char_start")),
            char_end=_int_or_none(row.get("char_end")),
        )
        keyed = False
        for key_field, index_map in (
            (KEY_ARTICLE_ID, by_id),
            (KEY_URL, by_url),
            (KEY_TITLE, by_title),
        ):
            key = str(row.get(key_field) or "").strip()
            if key:
                index_map[key] = entry
                keyed = True
        if not keyed:
            raise ValueError(
                f"{path}: mapping record {index} has no provider key "
                f"({KEY_ARTICLE_ID} / {KEY_URL} / {KEY_TITLE})"
            )
    return SourceMap(by_article_id=by_id, by_url=by_url, by_title=by_title)


def _read_mapping_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("mappings", [])
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON list of mapping records")
    return [row for row in payload if isinstance(row, dict)]


def map_source(source: dict[str, Any], source_map: SourceMap) -> SourceMapEntry | None:
    """Resolve one returned source record through the map (id > url > title precedence)."""
    article_id = str(source.get("article_id") or source.get("id") or "").strip()
    if article_id and article_id in source_map.by_article_id:
        return source_map.by_article_id[article_id]
    url = str(source.get("url") or source.get("uri") or "").strip()
    if url and url in source_map.by_url:
        return source_map.by_url[url]
    title = str(source.get("article_title") or source.get("title") or source.get("name") or "")
    title = title.strip()
    if title and title in source_map.by_title:
        return source_map.by_title[title]
    return None


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
    spans: list[SourceSpanRecord] = [
        {
            "doc_id": str(span.get("doc_id") or ""),
            "char_start": int(span.get("char_start") or 0),
            "char_end": int(span.get("char_end") or 0),
            "text": str(span.get("text") or ""),
        }
        for span in gold_spans
        if isinstance(span, dict)
    ]
    gold_docs = {span["doc_id"] for span in spans}
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
    strong_rank = first_hit_rank(chunks, spans) if spans else None
    if strong_rank is not None:
        hit, rank, weak = 1.0, strong_rank, False
    elif weak_rank is not None:
        hit, rank, weak = 1.0, weak_rank, True
    else:
        hit, rank, weak = 0.0, None, False
    return {
        "source_hit": hit if sources else "",
        "source_first_hit_rank": rank if rank is not None else "",
        "source_hit_weak": ("true" if weak else "false") if sources else "",
        "source_mapped_count": mapped,
        "source_unmapped_count": unmapped,
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


def _int_or_none(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(str(value))
