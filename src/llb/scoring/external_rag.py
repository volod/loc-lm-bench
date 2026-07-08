"""Score answered JSONL exports from an external or closed RAG system.

The normal `run-eval` path owns retrieval and generation locally. This module covers the other
operator workflow: a RAG system outside the benchmark has already answered each gold question, and
the benchmark should produce objective estimates plus final human-reviewed CSV/report artifacts.
"""

import csv
import io
import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.core.contracts import CorrectnessScores
from llb.core.fsutil import atomic_write_text
from llb.eval import common as eval_common
from llb.scoring.correctness import answer_correctness

ANSWER_FIELD_CANDIDATES = ("llm_answer", "predicted_answer", "model_answer", "answer")
ERROR_FIELD_CANDIDATES = ("llm_error", "error")
MODEL_FIELD_CANDIDATES = ("llm_model", "model", "service")
PROVIDER_FIELD_CANDIDATES = ("llm_provider", "provider")
ROUTE_FIELD_CANDIDATES = ("llm_route", "route")
SOURCES_FIELD_CANDIDATES = ("llm_sources", "sources", "retrieved_sources")
DEFAULT_SOURCE_LIMIT = 3
ANSWER_PREVIEW_CHARS = 180
WORST_CASE_REPORT_LIMIT = 12
SOURCE_REPORT_LIMIT = 8
HUMAN_SCORE_FIELD = "human_score_0_1"
HUMAN_DECISION_FIELD = "human_decision"
HUMAN_NOTES_FIELD = "human_notes"
HUMAN_CORRECTED_ANSWER_FIELD = "human_corrected_answer"
HUMAN_STATUS_FIELD = "human_status"
HUMAN_FIELDS = (
    HUMAN_SCORE_FIELD,
    HUMAN_DECISION_FIELD,
    HUMAN_NOTES_FIELD,
    HUMAN_CORRECTED_ANSWER_FIELD,
    HUMAN_STATUS_FIELD,
)
HUMAN_STATUS_SCORED = "scored"
HUMAN_DECISION_ACCEPT = "accept"
HUMAN_DECISION_PARTIAL = "partial"
HUMAN_DECISION_REJECT = "reject"
HUMAN_DECISIONS = (HUMAN_DECISION_ACCEPT, HUMAN_DECISION_PARTIAL, HUMAN_DECISION_REJECT)

STATUS_ERROR = "error"
STATUS_EMPTY = "empty"
STATUS_ABSTAINED = "abstained"
STATUS_REFUSAL = "refusal"
STATUS_OK = "ok"

ABSTENTION_MARKERS = (
    "context does not contain",
    "not found",
    "no relevant",
    "unknown",
    "немає відповіді",
    "не знайдено",
    "не містить",
    "контекст не містить",
    "у базі знань немає",
    "невідомо",
)

_SOURCE_FOOTER_RE = re.compile(
    r"(?im)^\s*(джерело|джерела|dzherelo|dzherela|source|sources)\s*:\s*.*\Z",
    re.DOTALL,
)


@dataclass(frozen=True)
class ExternalRagPaths:
    """Artifacts written by `score_external_rag_file`."""

    csv: Path
    report: Path


@dataclass(frozen=True)
class ExternalRagResult:
    """In-memory result returned after scoring an answered export."""

    rows: list[dict[str, object]]
    summary: dict[str, object]
    paths: ExternalRagPaths


def score_external_rag_file(
    answers_path: Path,
    *,
    csv_out: Path | None = None,
    report_out: Path | None = None,
    answer_field: str | None = None,
    sources_field: str | None = None,
    error_field: str | None = None,
    source_limit: int = DEFAULT_SOURCE_LIMIT,
    strip_source_footer: bool = True,
    label: str | None = None,
) -> ExternalRagResult:
    """Read an answered JSONL file and write the detailed CSV plus Markdown report."""
    if source_limit < 0:
        raise ValueError("source_limit must be >= 0")
    records = load_jsonl(answers_path)
    if not records:
        raise ValueError(f"{answers_path}: no JSONL records found")

    scored = score_records(
        records,
        answer_field=answer_field,
        sources_field=sources_field,
        error_field=error_field,
        source_limit=source_limit,
        strip_source_footer=strip_source_footer,
    )
    csv_path = csv_out or answers_path.with_suffix(".csv")
    report_path = report_out or answers_path.with_name(f"{answers_path.stem}.report.md")
    summary = summarize(scored, answers_path=answers_path, label=label)
    write_csv(scored, csv_path, source_limit=source_limit)
    write_report(
        scored,
        summary,
        report_path,
        answers_path=answers_path,
        csv_path=csv_path,
        source_limit=source_limit,
    )
    return ExternalRagResult(
        rows=scored, summary=summary, paths=ExternalRagPaths(csv_path, report_path)
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows with file:line context on parse failures."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(item)
    return rows


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Atomically write JSONL records, preserving Unicode text for human review."""
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    atomic_write_text(path, text)


def ensure_human_fields(records: Sequence[dict[str, Any]]) -> bool:
    """Ensure every record has the JSONL-backed human review fields.

    Returns true when at least one record changed.
    """
    changed = False
    for record in records:
        for field in HUMAN_FIELDS:
            if field not in record:
                record[field] = ""
                changed = True
    return changed


def clear_human_fields(records: Sequence[dict[str, Any]]) -> None:
    """Clear JSONL-backed human review state in place."""
    for record in records:
        for field in HUMAN_FIELDS:
            record[field] = ""


def is_human_scored(record: dict[str, Any]) -> bool:
    """Whether a record carries the required human scoring fields."""
    decision = _string(record.get(HUMAN_DECISION_FIELD)).strip().lower()
    score_text = _string(record.get(HUMAN_SCORE_FIELD)).strip()
    if decision not in HUMAN_DECISIONS or not score_text:
        return False
    try:
        score = float(score_text)
    except ValueError:
        return False
    return 0.0 <= score <= 1.0


def human_reviewed_count(records: Sequence[dict[str, Any]]) -> int:
    """Number of records with complete human review state."""
    return sum(1 for record in records if is_human_scored(record))


def score_records(
    records: list[dict[str, Any]],
    *,
    answer_field: str | None = None,
    sources_field: str | None = None,
    error_field: str | None = None,
    source_limit: int = DEFAULT_SOURCE_LIMIT,
    strip_source_footer: bool = True,
) -> list[dict[str, object]]:
    """Score records and return CSV-ready rows sorted by human review priority."""
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records, 1):
        raw_answer, answer_field_used = _field_value(record, answer_field, ANSWER_FIELD_CANDIDATES)
        raw_error, error_field_used = _field_value(record, error_field, ERROR_FIELD_CANDIDATES)
        raw_sources, sources_field_used = _field_value(
            record, sources_field, SOURCES_FIELD_CANDIDATES
        )
        answer = _string(raw_answer)
        scored_answer = clean_answer_for_scoring(answer, strip_source_footer=strip_source_footer)
        error = _string(raw_error)
        status = classify_external_answer(scored_answer, error)
        scores = answer_correctness(scored_answer, _string(record.get("reference_answer")))
        row = _base_row(
            record,
            input_index=index,
            status=status,
            scores=scores,
            answer=answer,
            scored_answer=scored_answer,
            error=error,
            answer_field=answer_field_used,
            error_field=error_field_used,
            sources_field=sources_field_used,
            source_limit=source_limit,
        )
        row.update(_source_columns(raw_sources, source_limit))
        rows.append(row)

    _attach_ranks(rows)
    rows.sort(
        key=lambda row: (
            _as_int(row.get("review_priority_rank")),
            -_as_float(row.get("objective_score")),
            _as_int(row.get("input_index")),
        )
    )
    return rows


def clean_answer_for_scoring(answer: str, *, strip_source_footer: bool = True) -> str:
    """Remove transport-only decorations before objective answer scoring."""
    text = _strip_think(answer).strip()
    if strip_source_footer:
        text = _SOURCE_FOOTER_RE.sub("", text).strip()
    return text


def classify_external_answer(answer: str, error: str) -> str:
    """Classify the answer for reliability and human review grouping."""
    if error.strip():
        return STATUS_ERROR
    if not answer.strip():
        return STATUS_EMPTY
    if eval_common.is_refusal(answer):
        return STATUS_REFUSAL
    normalized = eval_common.normalize_refusal_text(answer)
    if any(marker in normalized for marker in ABSTENTION_MARKERS):
        return STATUS_ABSTAINED
    return STATUS_OK


def summarize(
    rows: list[dict[str, object]], *, answers_path: Path, label: str | None
) -> dict[str, object]:
    """Aggregate headline estimates and review counts."""
    n = len(rows)
    by_status = Counter(_string(row.get("status")) for row in rows)
    by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_split[_string(row.get("split"))].append(row)
    objective_values = [_as_float(row.get("objective_score")) for row in rows]
    exact_values = [_as_float(row.get("exact")) for row in rows]
    contains_values = [_as_float(row.get("contains")) for row in rows]
    source_counts = [_as_int(row.get("source_count")) for row in rows]
    human_rows = [row for row in rows if _row_human_scored(row)]
    human_scores = [_as_float(row.get(HUMAN_SCORE_FIELD)) for row in human_rows]
    human_decisions = Counter(
        _string(row.get(HUMAN_DECISION_FIELD)).strip().lower() for row in human_rows
    )
    return {
        "label": label or _infer_label(rows, answers_path),
        "n": n,
        "objective_mean": _mean(objective_values),
        "exact_rate": _mean(exact_values),
        "contains_rate": _mean(contains_values),
        "status_counts": dict(sorted(by_status.items())),
        "split_metrics": _split_metrics(by_split),
        "verified_count": sum(1 for row in rows if _string(row.get("verified")).lower() == "true"),
        "mean_sources": _mean([float(value) for value in source_counts]),
        "source_title_counts": _source_title_counts(rows),
        "human_reviewed_count": len(human_rows),
        "human_pending_count": n - len(human_rows),
        "human_score_mean": _mean(human_scores),
        "human_decision_counts": dict(sorted(human_decisions.items())),
        "answer_fields": sorted({_string(row.get("answer_field")) for row in rows}),
        "sources_fields": sorted({_string(row.get("sources_field")) for row in rows}),
        "error_fields": sorted({_string(row.get("error_field")) for row in rows}),
    }


def write_csv(rows: list[dict[str, object]], path: Path, *, source_limit: int) -> None:
    """Write the detailed per-row worksheet CSV."""
    fieldnames = csv_columns(source_limit)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(_csv_row(row, fieldnames) for row in rows)
    atomic_write_text(path, out.getvalue())


def write_report(
    rows: list[dict[str, object]],
    summary: dict[str, object],
    path: Path,
    *,
    answers_path: Path,
    csv_path: Path,
    source_limit: int,
) -> None:
    """Write the Markdown score estimate and recommendation report."""
    lines = [
        "# External RAG score report",
        "",
        "This is an answer-log diagnostic for an external RAG system. It does not launch the "
        "project RAG backend, and it does not certify unverified gold rows as a benchmark "
        "leaderboard.",
        "",
        "## Inputs and outputs",
        "",
        f"- Answer log: `{answers_path}`",
        f"- Detailed CSV: `{csv_path}`",
        f"- System label: `{summary['label']}`",
        "- Scoring: normalized exact match, token F1 objective, and contains.",
        "- Source footer handling: a trailing `Source:` / `Dzherelo:` answer footer is stripped "
        "for scoring; the raw answer remains in the CSV.",
        f"- Source columns: first {source_limit} returned source record(s).",
        "- Answer field(s): "
        + _field_set(summary.get("answer_fields"))
        + "; source field(s): "
        + _field_set(summary.get("sources_fields"))
        + "; error field(s): "
        + _field_set(summary.get("error_fields"))
        + ".",
        "",
        "## Score estimates",
        "",
        _summary_table(summary),
        "",
        "## Human decisions",
        "",
        _human_table(summary),
        "",
        "## Split estimates",
        "",
        _split_table(summary["split_metrics"]),
        "",
        "## Human review workflow",
        "",
        "The interactive command stores human review state in the JSONL answer log. The CSV and "
        "this report are generated only after all rows have `human_score_0_1` and "
        "`human_decision`.",
        "",
        "## Highest-priority rows",
        "",
        _priority_table(rows),
        "",
        "## Common returned sources",
        "",
        _source_table(summary["source_title_counts"]),
        "",
        "## Improvement recommendations",
        "",
        *_recommendation_lines(rows, summary),
        "",
        "## Project tuning map",
        "",
        "- Validate the gold data shape before headline use: "
        "`make validate-goldset GOLDSET=<goldset.jsonl> CORPUS=<corpus-dir>`; guide: "
        "`docs/guides/data-prep/goldset-from-scratch.md`.",
        "- Build a local retrieval baseline over the same corpus: "
        "`make build-index CORPUS=<corpus-dir>` then "
        "`make validate-retrieval GOLDSET=<goldset.jsonl> RAG_K=10`.",
        "- Compare Ukrainian embedders on this corpus: "
        "`make compare-embeddings GOLDSET=<goldset.jsonl> RAG_K=20`.",
        "- Test chunking and retrieval-mode knobs locally: "
        "`llb build-index --corpus-root <corpus-dir> --strategy markdown --size 800 "
        "--overlap 120 --mode parent_child`.",
        "- Sweep model and retrieval depth: "
        "`make sweep GOLDSET=<goldset.jsonl> SWEEP_RAG_GRID=top_k=3,5,8` then "
        "`make recommend`.",
        "- Tune RAG parameters on the tuning split and score only the final split winner: "
        "`llb tune --model <model> --backend <backend> --goldset <goldset.jsonl>`.",
        "- Generate and compare prompt packages: "
        "`make prompt-system-prepare PROMPT_SYSTEM_CORPUS=<corpus-dir>`, then run "
        "`make run-eval PROMPT_SYSTEM_ID=<id>`.",
        "- After a local run, classify misses and get evidence-backed actions: "
        "`make analyze-misses RUN_DIR=<run-eval-bundle> PROBE_TOP_K=3,8`.",
        "- External-service artifact manual: "
        "`docs/guides/data-prep/external-ai-service-artifacts.md`.",
    ]
    atomic_write_text(path, "\n".join(lines) + "\n")


def csv_columns(source_limit: int) -> list[str]:
    """Stable CSV column order for human review and downstream analysis."""
    columns = [
        "review_priority_rank",
        "score_rank",
        "input_index",
        "id",
        "split",
        "verified",
        "status",
        "objective_score",
        "token_f1",
        "exact",
        "contains",
        "question",
        "reference_answer",
        "scored_answer",
        "llm_answer",
        "llm_model",
        "llm_provider",
        "llm_route",
        "llm_error",
        "answer_field",
        "error_field",
        "sources_field",
        "source_doc_id",
        "source_span_1_doc_id",
        "source_span_1_char_start",
        "source_span_1_char_end",
        "source_span_1_text",
        "source_count",
    ]
    for index in range(1, source_limit + 1):
        columns.extend(
            [
                f"source_{index}_article_id",
                f"source_{index}_doc_id",
                f"source_{index}_title",
                f"source_{index}_score",
                f"source_{index}_url",
            ]
        )
    columns.extend(
        [
            HUMAN_SCORE_FIELD,
            HUMAN_DECISION_FIELD,
            HUMAN_NOTES_FIELD,
            HUMAN_CORRECTED_ANSWER_FIELD,
            HUMAN_STATUS_FIELD,
        ]
    )
    return columns


def _base_row(
    record: dict[str, Any],
    *,
    input_index: int,
    status: str,
    scores: CorrectnessScores,
    answer: str,
    scored_answer: str,
    error: str,
    answer_field: str,
    error_field: str,
    sources_field: str,
    source_limit: int,
) -> dict[str, object]:
    span = _first_span(record)
    return {
        "review_priority_rank": 0,
        "score_rank": 0,
        "input_index": input_index,
        "id": _string(record.get("id")),
        "split": _string(record.get("split")),
        "verified": str(bool(record.get("verified", False))).lower(),
        "status": status,
        "objective_score": _round(scores["score"]),
        "token_f1": _round(scores["token_f1"]),
        "exact": _round(scores["exact"]),
        "contains": _round(scores["contains"]),
        "question": _string(record.get("question")),
        "reference_answer": _string(record.get("reference_answer")),
        "scored_answer": scored_answer,
        "llm_answer": answer,
        "llm_model": _field_string(record, None, MODEL_FIELD_CANDIDATES),
        "llm_provider": _field_string(record, None, PROVIDER_FIELD_CANDIDATES),
        "llm_route": _field_string(record, None, ROUTE_FIELD_CANDIDATES),
        "llm_error": error,
        "answer_field": answer_field,
        "error_field": error_field,
        "sources_field": sources_field,
        "source_doc_id": _string(record.get("source_doc_id")),
        "source_span_1_doc_id": _string(span.get("doc_id")),
        "source_span_1_char_start": _string(span.get("char_start")),
        "source_span_1_char_end": _string(span.get("char_end")),
        "source_span_1_text": _string(span.get("text")),
        "source_count": _source_count(record, sources_field),
        HUMAN_SCORE_FIELD: _string(record.get(HUMAN_SCORE_FIELD)),
        HUMAN_DECISION_FIELD: _string(record.get(HUMAN_DECISION_FIELD)),
        HUMAN_NOTES_FIELD: _string(record.get(HUMAN_NOTES_FIELD)),
        HUMAN_CORRECTED_ANSWER_FIELD: _string(record.get(HUMAN_CORRECTED_ANSWER_FIELD)),
        HUMAN_STATUS_FIELD: _string(record.get(HUMAN_STATUS_FIELD)),
        **{key: "" for key in _empty_source_column_names(source_limit)},
    }


def _attach_ranks(rows: list[dict[str, object]]) -> None:
    score_order = sorted(
        range(len(rows)),
        key=lambda index: (
            -_as_float(rows[index].get("objective_score")),
            _as_int(rows[index].get("input_index")),
        ),
    )
    for rank, index in enumerate(score_order, 1):
        rows[index]["score_rank"] = rank

    review_order = sorted(
        range(len(rows)),
        key=lambda index: (
            _status_priority(_string(rows[index].get("status"))),
            _as_float(rows[index].get("objective_score")),
            _as_int(rows[index].get("source_count")),
            _as_int(rows[index].get("input_index")),
        ),
    )
    for rank, index in enumerate(review_order, 1):
        rows[index]["review_priority_rank"] = rank


def _status_priority(status: str) -> int:
    priorities = {
        STATUS_ERROR: 0,
        STATUS_EMPTY: 1,
        STATUS_ABSTAINED: 2,
        STATUS_REFUSAL: 3,
        STATUS_OK: 4,
    }
    return priorities.get(status, 5)


def _source_columns(raw_sources: object, source_limit: int) -> dict[str, object]:
    sources = _source_list(raw_sources)
    out: dict[str, object] = {}
    for index, source in enumerate(sources[:source_limit], 1):
        out[f"source_{index}_article_id"] = _string(source.get("article_id") or source.get("id"))
        out[f"source_{index}_doc_id"] = _string(source.get("doc_id") or source.get("document_id"))
        out[f"source_{index}_title"] = _string(
            source.get("article_title") or source.get("title") or source.get("name")
        )
        out[f"source_{index}_score"] = _string(source.get("score"))
        out[f"source_{index}_url"] = _string(source.get("url") or source.get("uri"))
    return out


def _source_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _source_count(record: dict[str, Any], sources_field: str) -> int:
    sources = record.get(sources_field) if sources_field else None
    return len(_source_list(sources))


def _empty_source_column_names(source_limit: int) -> list[str]:
    names: list[str] = []
    for index in range(1, source_limit + 1):
        names.extend(
            [
                f"source_{index}_article_id",
                f"source_{index}_doc_id",
                f"source_{index}_title",
                f"source_{index}_score",
                f"source_{index}_url",
            ]
        )
    return names


def _first_span(record: dict[str, Any]) -> dict[str, Any]:
    spans = record.get("source_spans")
    if isinstance(spans, list) and spans and isinstance(spans[0], dict):
        return spans[0]
    return {}


def _field_value(
    record: dict[str, Any], requested: str | None, candidates: tuple[str, ...]
) -> tuple[object, str]:
    if requested is not None:
        return record.get(requested), requested
    for field in candidates:
        if field in record:
            return record.get(field), field
    return "", ""


def _field_string(
    record: dict[str, Any], requested: str | None, candidates: tuple[str, ...]
) -> str:
    value, _field = _field_value(record, requested, candidates)
    return _string(value)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int | float | bool):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _round(value: float) -> float:
    return round(float(value), 4)


def _as_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value:
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _infer_label(rows: list[dict[str, object]], answers_path: Path) -> str:
    models = sorted({_string(row.get("llm_model")) for row in rows if row.get("llm_model")})
    routes = sorted({_string(row.get("llm_route")) for row in rows if row.get("llm_route")})
    if models or routes:
        return "/".join(part for part in [",".join(models), ",".join(routes)] if part)
    return answers_path.stem


def _split_metrics(by_split: dict[str, list[dict[str, object]]]) -> dict[str, dict[str, object]]:
    metrics: dict[str, dict[str, object]] = {}
    for split, split_rows in sorted(by_split.items()):
        metrics[split or "unknown"] = {
            "n": len(split_rows),
            "objective_mean": _mean([_as_float(row.get("objective_score")) for row in split_rows]),
            "exact_rate": _mean([_as_float(row.get("exact")) for row in split_rows]),
            "contains_rate": _mean([_as_float(row.get("contains")) for row in split_rows]),
            "status_counts": dict(
                sorted(Counter(_string(row.get("status")) for row in split_rows).items())
            ),
        }
    return metrics


def _source_title_counts(rows: list[dict[str, object]]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for row in rows:
        for key, value in row.items():
            if key.startswith("source_") and key.endswith("_title") and _string(value):
                counts[_string(value)] += 1
    return counts.most_common(SOURCE_REPORT_LIMIT)


def _summary_table(summary: dict[str, object]) -> str:
    status_counts = summary["status_counts"]
    assert isinstance(status_counts, dict)
    statuses = ", ".join(f"{key}={value}" for key, value in status_counts.items()) or "none"
    rows = [
        ("rows", str(summary["n"])),
        ("verified rows", str(summary["verified_count"])),
        ("objective mean", f"{_as_float(summary.get('objective_mean')):.4f}"),
        ("exact rate", f"{_as_float(summary.get('exact_rate')):.4f}"),
        ("contains rate", f"{_as_float(summary.get('contains_rate')):.4f}"),
        ("mean returned sources", f"{_as_float(summary.get('mean_sources')):.2f}"),
        ("human reviewed rows", str(summary.get("human_reviewed_count", 0))),
        ("human mean score", f"{_as_float(summary.get('human_score_mean')):.4f}"),
        ("status counts", statuses),
    ]
    return _md_table(["metric", "value"], rows)


def _human_table(summary: dict[str, object]) -> str:
    decisions = summary.get("human_decision_counts")
    if not isinstance(decisions, dict) or not decisions:
        return "No human decisions were recorded."
    rows = [(str(key), str(value)) for key, value in sorted(decisions.items())]
    return _md_table(["decision", "rows"], rows)


def _field_set(value: object) -> str:
    if not isinstance(value, list):
        return "(none)"
    fields = [item for item in (_string(item) for item in value) if item]
    return ", ".join(fields) if fields else "(none)"


def _split_table(split_metrics: object) -> str:
    assert isinstance(split_metrics, dict)
    rows = []
    for split, metrics in split_metrics.items():
        assert isinstance(metrics, dict)
        rows.append(
            (
                str(split),
                str(metrics["n"]),
                f"{_as_float(metrics.get('objective_mean')):.4f}",
                f"{_as_float(metrics.get('exact_rate')):.4f}",
                f"{_as_float(metrics.get('contains_rate')):.4f}",
                ", ".join(
                    f"{key}={value}" for key, value in dict(metrics["status_counts"]).items()
                ),
            )
        )
    return _md_table(["split", "n", "objective", "exact", "contains", "statuses"], rows)


def _priority_table(rows: list[dict[str, object]]) -> str:
    table_rows = []
    for row in rows[:WORST_CASE_REPORT_LIMIT]:
        table_rows.append(
            (
                str(row["review_priority_rank"]),
                _string(row["id"]),
                _string(row["status"]),
                f"{_as_float(row.get('objective_score')):.4f}",
                _ellipsize(_string(row["question"]), ANSWER_PREVIEW_CHARS),
            )
        )
    return _md_table(["priority", "id", "status", "score", "question"], table_rows)


def _source_table(source_counts: object) -> str:
    assert isinstance(source_counts, list)
    rows = [
        (_string(item[0]), str(_as_int(item[1])))
        for item in source_counts
        if isinstance(item, tuple) and len(item) == 2
    ]
    return _md_table(["source title", "rows"], rows) if rows else "No sources were returned."


def _recommendation_lines(rows: list[dict[str, object]], summary: dict[str, object]) -> list[str]:
    n = _as_int(summary.get("n"))
    objective = _as_float(summary.get("objective_mean"))
    exact = _as_float(summary.get("exact_rate"))
    contains = _as_float(summary.get("contains_rate"))
    statuses = summary["status_counts"]
    assert isinstance(statuses, dict)
    abstained = int(statuses.get(STATUS_ABSTAINED, 0))
    empty = int(statuses.get(STATUS_EMPTY, 0))
    errors = int(statuses.get(STATUS_ERROR, 0))
    no_sources = len([row for row in rows if _as_int(row.get("source_count")) == 0])
    human_reviewed = _as_int(summary.get("human_reviewed_count"))
    human_pending = _as_int(summary.get("human_pending_count"))
    if human_reviewed == n and n:
        decisions = summary.get("human_decision_counts")
        decision_text = (
            ", ".join(f"{key}={value}" for key, value in dict(decisions).items())
            if isinstance(decisions, dict)
            else "none"
        )
        lines = [
            "- Human review is complete. Treat the human mean score and decision split as the "
            "primary quality estimate; use objective scores as triage signals. Decisions: "
            f"{decision_text}.",
        ]
    else:
        lines = [
            "- Treat this as an estimate until the JSONL human fields are complete. The input "
            f"contains {summary['verified_count']} verified rows out of {n}; "
            f"{human_pending} rows still need human decisions.",
        ]
    if objective < 0.35:
        lines.append(
            "- The objective score is low. Start with retrieval and corpus alignment: confirm the "
            "external RAG is indexing the same staged corpus text as the goldset, then compare a "
            "local baseline with `make validate-retrieval`."
        )
    elif objective < 0.65:
        lines.append(
            "- The objective score is mixed. Review the priority rows to separate retrieval misses "
            "from answer-generation misses before tuning prompts or models."
        )
    else:
        lines.append(
            "- The objective score is relatively strong. Use human review to catch paraphrases, "
            "overlong answers, and unsupported statements that token F1 cannot judge."
        )
    if abstained / max(n, 1) >= 0.2:
        lines.append(
            "- Abstentions are common. Raise retrieval coverage first: increase or sweep `top_k`, "
            "try `parent_child` mode, and test markdown/recursive chunk sizes before changing the "
            "answer model."
        )
    if empty or errors:
        lines.append(
            f"- Transport or empty-answer failures exist (`error={errors}`, `empty={empty}`). Fix "
            "API reliability before interpreting quality deltas."
        )
    if no_sources / max(n, 1) >= 0.2:
        lines.append(
            "- Many rows have no returned sources. Configure the external API to return at least "
            "the top three source records; ideally include corpus `doc_id`, `char_start`, and "
            "`char_end` so source-span recall can be audited."
        )
    if contains > objective + 0.2 or (contains >= 0.5 and exact < 0.2):
        lines.append(
            "- The system often mentions the reference tokens but fails exact concise answering. "
            "Tighten the generation prompt to return a short direct answer and keep citations in "
            "structured metadata rather than the answer text."
        )
    return lines


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_escape_md_cell(value) for value in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _escape_md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _csv_row(row: dict[str, object], fieldnames: list[str]) -> dict[str, str]:
    return {field: _one_line(_string(row.get(field))) for field in fieldnames}


def _one_line(value: str) -> str:
    return " ".join(value.splitlines())


def _ellipsize(text: str, limit: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)] + "..."


def _row_human_scored(row: dict[str, object]) -> bool:
    decision = _string(row.get(HUMAN_DECISION_FIELD)).strip().lower()
    score_text = _string(row.get(HUMAN_SCORE_FIELD)).strip()
    if decision not in HUMAN_DECISIONS or not score_text:
        return False
    try:
        score = float(score_text)
    except ValueError:
        return False
    return 0.0 <= score <= 1.0
