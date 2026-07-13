"""Headline aggregation for external-RAG scoring: per-run and per-split summary estimates.

The `summarize` step turns scored rows into the summary dict that `external_rag_report.py` renders.
Pure over the CSV-ready rows; no I/O.
"""

from collections import Counter, defaultdict
from pathlib import Path

from llb.scoring.external_rag_common import (
    HUMAN_DECISION_FIELD,
    HUMAN_DECISIONS,
    HUMAN_SCORE_FIELD,
    SOURCE_REPORT_LIMIT,
    _as_float,
    _as_int,
    _mean,
    _string,
)


def summarize(
    rows: list[dict[str, object]], *, answers_path: Path, label: str | None
) -> dict[str, object]:
    """Aggregate headline estimates and review counts."""
    n = len(rows)
    by_status = Counter(_string(row.get("status")) for row in rows)
    by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_split[_string(row.get("split"))].append(row)
    source_counts = [_as_int(row.get("source_count")) for row in rows]
    return {
        "label": label or _infer_label(rows, answers_path),
        "n": n,
        "objective_mean": _mean([_as_float(row.get("objective_score")) for row in rows]),
        "exact_rate": _mean([_as_float(row.get("exact")) for row in rows]),
        "contains_rate": _mean([_as_float(row.get("contains")) for row in rows]),
        "status_counts": dict(sorted(by_status.items())),
        "split_metrics": _split_metrics(by_split),
        "verified_count": sum(1 for row in rows if _string(row.get("verified")).lower() == "true"),
        "mean_sources": _mean([float(value) for value in source_counts]),
        "source_title_counts": _source_title_counts(rows),
        **_human_review_fields(rows, n),
        "answer_fields": sorted({_string(row.get("answer_field")) for row in rows}),
        "sources_fields": sorted({_string(row.get("sources_field")) for row in rows}),
        "error_fields": sorted({_string(row.get("error_field")) for row in rows}),
    }


def _human_review_fields(rows: list[dict[str, object]], n: int) -> dict[str, object]:
    """The human-review slice of the summary: reviewed/pending counts, mean score, decisions."""
    human_rows = [row for row in rows if _row_human_scored(row)]
    human_decisions = Counter(
        _string(row.get(HUMAN_DECISION_FIELD)).strip().lower() for row in human_rows
    )
    return {
        "human_reviewed_count": len(human_rows),
        "human_pending_count": n - len(human_rows),
        "human_score_mean": _mean([_as_float(row.get(HUMAN_SCORE_FIELD)) for row in human_rows]),
        "human_decision_counts": dict(sorted(human_decisions.items())),
    }


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
