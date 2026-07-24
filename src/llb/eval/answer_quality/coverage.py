"""Multi-span coverage columns, recomputed from a run bundle's retrieval sidecar.

A case score row carries `retrieval_hit`, which is `recall@k`: an item counts as a hit as soon as
ANY labeled span is retrieved -- a two-hop question satisfies it by returning one hop. That is the
exact blind spot the fusion-evidence lane was built to remove, so a comparison that only reads
`retrieval_hit` could not see a multi-hop coverage gain even when one happened.

`run-eval` already persists every case's retrieved spans and gold spans in `retrieval.jsonl`, so
the two multi-span metrics are recomputed here from that sidecar with the SAME functions the
retrieval sweep uses. The lane then pairs the coverage the sweep measures against the answers the
model produced from it.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.rag.retrieval import all_spans_at_k, span_coverage_at_k
from llb.rag.retrieval_records import record_as_chunk

RETRIEVAL_FILENAME = "retrieval.jsonl"
METRIC_ALL_SPANS = "all_spans_at_k"
METRIC_SPAN_COVERAGE = "span_coverage"


def _as_chunks(records: list[JsonObject]) -> list[ChunkRecord]:
    """Persisted rows back as chunks, with the occurrences of a collapsed chunk restored, so a
    span carried by a duplicate copy counts as covered here exactly as it did in the run."""
    return [record_as_chunk(record) for record in records]  # type: ignore[arg-type]


def _as_spans(records: list[JsonObject]) -> list[SourceSpanRecord]:
    return [
        {
            "doc_id": str(record.get("doc_id", "")),
            "char_start": int(record.get("char_start", 0)),
            "char_end": int(record.get("char_end", 0)),
            "text": str(record.get("text", "")),
        }
        for record in records
    ]


def read_case_coverage(run_dir: Path, k: int) -> dict[str, dict[str, float]]:
    """Per item id, the multi-span coverage of its scored context ({} when no sidecar exists)."""
    path = Path(run_dir) / RETRIEVAL_FILENAME
    if not path.is_file():
        return {}
    coverage: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            retrieved = _as_chunks(list(record.get("retrieved") or []))
            spans = _as_spans(list(record.get("gold_spans") or []))
            coverage[str(record["item_id"])] = {
                METRIC_ALL_SPANS: all_spans_at_k(retrieved, spans, k),
                METRIC_SPAN_COVERAGE: span_coverage_at_k(retrieved, spans, k),
            }
    return coverage


def with_coverage(
    rows: list[Mapping[str, Any]], coverage: dict[str, dict[str, float]]
) -> list[Mapping[str, Any]]:
    """Attach the multi-span coverage columns to each case row it was measured for."""
    if not coverage:
        return rows
    return [{**row, **coverage.get(str(row["item_id"]), {})} for row in rows]
