"""Focused external rag sections implementation."""

from collections.abc import Sequence
from llb.scoring.external_rag_common import (
    ANSWER_PREVIEW_CHARS,
    STATUS_ABSTAINED,
    STATUS_EMPTY,
    STATUS_ERROR,
    WORST_CASE_REPORT_LIMIT,
    _as_float,
    _as_int,
    _ellipsize,
    _string,
)


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


def _source_audit_section(summary: dict[str, object]) -> list[str]:
    """The source-span audit block (present only when a --source-map was supplied)."""
    audit = summary.get("source_audit")
    if not isinstance(audit, dict):
        return []
    rows = [
        ("rows audited (returned >= 1 source)", str(audit.get("rows_audited", 0))),
        ("source recall@3 (span-proof)", f"{_as_float(audit.get('source_recall_at_3')):.4f}"),
        ("source MRR (span-proof)", f"{_as_float(audit.get('source_mrr')):.4f}"),
        ("weak (doc-level only) hit rows", str(audit.get("weak_hit_rows", 0))),
        ("mapped sources", str(audit.get("mapped_sources", 0))),
        ("unmapped sources", str(audit.get("unmapped_sources", 0))),
        ("unmapped rate", f"{_as_float(audit.get('unmapped_rate')):.4f}"),
    ]
    return [
        "## Source-span audit",
        "",
        "Provider sources joined onto corpus spans via the operator --source-map. recall@3 and "
        "MRR count SPAN-PROOF hits only (the same source-span metric as local retrieval); a "
        "doc-level match from a span-less mapping is flagged weak evidence, and unmapped "
        "returned sources are an audit gap, not a retrieval miss.",
        "",
        _md_table(["metric", "value"], rows),
        "",
    ]


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
    lines = [_review_status_line(summary, n), _objective_advice_line(objective)]
    lines.extend(_reliability_advice_lines(rows, summary, n))
    contains = _as_float(summary.get("contains_rate"))
    exact = _as_float(summary.get("exact_rate"))
    if contains > objective + 0.2 or (contains >= 0.5 and exact < 0.2):
        lines.append(
            "- The system often mentions the reference tokens but fails exact concise answering. "
            "Tighten the generation prompt to return a short direct answer and keep citations in "
            "structured metadata rather than the answer text."
        )
    return lines


def _review_status_line(summary: dict[str, object], n: int) -> str:
    """The lead line: human review complete (trust human scores) vs pending (an estimate)."""
    human_reviewed = _as_int(summary.get("human_reviewed_count"))
    if human_reviewed == n and n:
        decisions = summary.get("human_decision_counts")
        decision_text = (
            ", ".join(f"{key}={value}" for key, value in dict(decisions).items())
            if isinstance(decisions, dict)
            else "none"
        )
        return (
            "- Human review is complete. Treat the human mean score and decision split as the "
            "primary quality estimate; use objective scores as triage signals. Decisions: "
            f"{decision_text}."
        )
    human_pending = _as_int(summary.get("human_pending_count"))
    return (
        "- Treat this as an estimate until the JSONL human fields are complete. The input "
        f"contains {summary['verified_count']} verified rows out of {n}; "
        f"{human_pending} rows still need human decisions."
    )


def _objective_advice_line(objective: float) -> str:
    """One line of next-step advice for the low / mixed / strong objective-score band."""
    if objective < 0.35:
        return (
            "- The objective score is low. Start with retrieval and corpus alignment: confirm the "
            "external RAG is indexing the same staged corpus text as the goldset, then compare a "
            "local baseline with `make validate-retrieval`."
        )
    if objective < 0.65:
        return (
            "- The objective score is mixed. Review the priority rows to separate retrieval misses "
            "from answer-generation misses before tuning prompts or models."
        )
    return (
        "- The objective score is relatively strong. Use human review to catch paraphrases, "
        "overlong answers, and unsupported statements that token F1 cannot judge."
    )


def _reliability_advice_lines(
    rows: list[dict[str, object]], summary: dict[str, object], n: int
) -> list[str]:
    """Advice triggered by abstentions, transport failures, or missing source records."""
    statuses = summary["status_counts"]
    assert isinstance(statuses, dict)
    abstained = int(statuses.get(STATUS_ABSTAINED, 0))
    empty = int(statuses.get(STATUS_EMPTY, 0))
    errors = int(statuses.get(STATUS_ERROR, 0))
    no_sources = len([row for row in rows if _as_int(row.get("source_count")) == 0])
    lines: list[str] = []
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
    return lines


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_escape_md_cell(value) for value in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _escape_md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
