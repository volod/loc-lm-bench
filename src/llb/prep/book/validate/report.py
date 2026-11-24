"""`metadata/processing_report.md` and `metadata/validation.json` for one vault."""

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.core.contracts.data_prep.book import BookStageSpec
from llb.core.contracts.data_prep.book_blocks import UNCERTAINTY_CLASSES
from llb.core.contracts.data_prep.book_review import MARKER_KINDS
from llb.core.fsutil import atomic_write_text
from llb.prep.book.constants import METADATA_DIRNAME, PAGE_STATUSES, PROCESSING_REPORT_FILENAME
from llb.prep.book.ledger.manifest import REUSED
from llb.prep.book.validate.constants import (
    CONFIDENCE_BUCKETS,
    CONFIDENCE_NOTE,
    VALIDATION_FILENAME,
)
from llb.prep.book.validate.references import reference_payload, reference_section
from llb.prep.book.validate.types import (
    EngineScores,
    OpenMark,
    PageRow,
    ValidationReport,
)


def write_report(report: ValidationReport, open_marks: list[OpenMark]) -> tuple[Path, Path]:
    """Write the human report and the machine-readable result. Both are deterministic."""
    directory = report.context.book_dir / METADATA_DIRNAME
    markdown = directory / PROCESSING_REPORT_FILENAME
    payload = directory / VALIDATION_FILENAME
    atomic_write_text(markdown, render_markdown(report, open_marks))
    atomic_write_text(payload, render_json(report, open_marks))
    return markdown, payload


def render_markdown(report: ValidationReport, open_marks: list[OpenMark]) -> str:
    """The processing report. It names pages and counts, never recovered text."""
    ctx = report.context
    source = ctx.config.source
    lines = [
        "# Book vault validation",
        "",
        "## Source",
        "",
        f"- File: `{Path(source.original_path).name}` ({source.size_bytes} bytes, "
        f"{source.page_count} PDF pages)",
        f"- SHA-256: `{source.sha256}`",
        f"- Selected: `{ctx.config.pages_spec}` -> {len(ctx.config.pages)} page(s)",
        "",
        "## Result",
        "",
        _result(report),
        "",
        "## Engines, models, and parameters",
        "",
        *_engines(ctx.config.stages),
        "",
        "## Readings",
        "",
        *_readings(report.facts.readings),
        "",
        "## Per-page status",
        "",
        _status_note(),
        "",
        *_page_table(report.pages),
        "",
        *_stage_section(report),
        "## Warnings",
        "",
        *_bullets(report.warnings),
        "",
        "## Confidence",
        "",
        CONFIDENCE_NOTE,
        "",
        *_confidence(report.facts.scores),
        "",
        "## Review queue",
        "",
        *_queue(open_marks, report),
        "",
        "## Manual verification",
        "",
        *_manual(report.pages, open_marks),
        "",
        *reference_section(ctx.book_dir),
        "## Checks",
        "",
        *_checks(report),
        "",
    ]
    return "\n".join(lines)


def render_json(report: ValidationReport, open_marks: list[OpenMark]) -> str:
    """The same outcome as JSON, with one record per PDF page."""
    source = report.context.config.source
    payload = {
        "checks": [
            {"check": item.check, "detail": item.detail, "file": item.file}
            for item in report.findings
        ],
        "confidence": _confidence_json(report.facts.scores),
        "engines": [
            {"model": spec.model, "params": spec.params, "stage": spec.name}
            for spec in report.context.config.stages
        ],
        "manual_verification": _manual(report.pages, open_marks),
        "ok": report.ok,
        "orphaned_by_reason": report.facts.orphaned_by_reason,
        "orphaned_overlays": report.facts.orphans,
        "pages": [
            {
                "page_id": row.page_id,
                "pdf_page": row.pdf_page,
                "reason": row.reason,
                "status": row.status,
            }
            for row in report.pages
        ],
        "references": reference_payload(report.context.book_dir),
        "readings": [
            {"blocks": count, "engine": engine, "model": model, "variant": variant}
            for engine, model, variant, count in report.facts.readings
        ],
        "review_by_uncertainty_class": _class_counts(open_marks),
        "review_queue": _kind_counts(open_marks),
        "source": {
            "name": Path(source.original_path).name,
            "page_count": source.page_count,
            "pages_spec": report.context.config.pages_spec,
            "sha256": source.sha256,
        },
        "warnings": list(report.warnings),
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _result(report: ValidationReport) -> str:
    counts = Counter(row.status for row in report.pages)
    summary = ", ".join(
        f"{counts.get(name, 0)} {name}"
        for name in ("completed", "failed", "skipped", "unaccounted")
    )
    if report.ok:
        return f"Passed. {summary}."
    return f"Failed. {len(report.findings)} check(s). {summary}."


def _status_note() -> str:
    return "Consecutive skipped pages share a row. `validation.json` lists every page."


def _engines(stages: Sequence[BookStageSpec]) -> list[str]:
    lines = ["| Stage | Model | Parameters |", "| --- | --- | --- |"]
    for spec in stages:
        model = spec.model or "none"
        params = _params(spec.params)
        lines.append(f"| `{spec.name}` | {model} | {params} |")
    return lines


def _params(params: JsonObject) -> str:
    if not params:
        return "none"
    parts = [f"{key}={json.dumps(params[key], ensure_ascii=True)}" for key in sorted(params)]
    return ", ".join(parts)


def _readings(rows: list[tuple[str, str, str, int]]) -> list[str]:
    if not rows:
        return ["No block readings were recorded."]
    lines = ["| Engine | Model | Variant | Blocks |", "| --- | --- | --- | --- |"]
    for engine, model, variant, count in rows:
        lines.append(f"| `{engine}` | {model} | `{variant}` | {count} |")
    return lines


def _page_table(pages: tuple[PageRow, ...] | list[PageRow]) -> list[str]:
    lines = ["| PDF page | Page | Status | Reason |", "| --- | --- | --- | --- |"]
    index = 0
    rows = list(pages)
    while index < len(rows):
        row = rows[index]
        end = _skip_end(rows, index) if row.status == "skipped" else index
        lines.append(_page_line(rows[index], rows[end]))
        index = end + 1
    return lines


def _skip_end(rows: list[PageRow], index: int) -> int:
    end = index
    reason = rows[index].reason
    while end + 1 < len(rows) and _continues(rows[end], rows[end + 1], reason):
        end += 1
    return end


def _continues(current: PageRow, nxt: PageRow, reason: str) -> bool:
    return nxt.status == "skipped" and nxt.reason == reason and nxt.pdf_page == current.pdf_page + 1


def _page_line(start: PageRow, end: PageRow) -> str:
    if start.pdf_page == end.pdf_page:
        span, pid = str(start.pdf_page), f"`{start.page_id}`"
    else:
        span = f"{start.pdf_page}-{end.pdf_page}"
        pid = f"`{start.page_id}`-`{end.page_id}`"
    return f"| {span} | {pid} | {start.status} | {start.reason} |"


def _stage_section(report: ValidationReport) -> list[str]:
    manifest = report.context.manifest
    if manifest is None:
        return ["## Stage status", "", "No page manifest was recorded.", ""]
    columns = [*PAGE_STATUSES, REUSED]
    lines = [
        "## Stage status",
        "",
        "| Stage | " + " | ".join(columns) + " |",
        "| --- |" + " --- |" * len(columns),
    ]
    for name in manifest.stage_names:
        counts = manifest.summary.get(name, {})
        cells = " | ".join(str(counts.get(column, 0)) for column in columns)
        lines.append(f"| `{name}` | {cells} |")
    lines.append("")
    reused = any(manifest.summary.get(name, {}).get(REUSED, 0) for name in manifest.stage_names)
    if reused:
        lines.append("A reused page is also counted under its status.")
        lines.append("")
    return lines


def _bullets(items: tuple[str, ...] | list[str]) -> list[str]:
    return [f"- {item}" for item in items] or ["None."]


def _confidence(scores: dict[str, EngineScores]) -> list[str]:
    if not scores:
        return ["No confidence was recorded."]
    header = (
        "| Engine | Count | Missing | Min | Max | "
        + " | ".join(name for name, _low, _high in CONFIDENCE_BUCKETS)
        + " |"
    )
    rule = (
        "| --- | --- | --- | --- | --- | "
        + " | ".join("---" for _item in CONFIDENCE_BUCKETS)
        + " |"
    )
    lines = [header, rule]
    for engine in sorted(scores):
        lines.append(_score_line(engine, scores[engine]))
    return lines


def _score_line(engine: str, scores: EngineScores) -> str:
    low = "none" if scores.minimum is None else f"{scores.minimum:.3f}"
    high = "none" if scores.maximum is None else f"{scores.maximum:.3f}"
    buckets = " | ".join(
        str(scores.buckets.get(name, 0)) for name, _low, _high in CONFIDENCE_BUCKETS
    )
    return f"| `{engine}` | {scores.count} | {scores.missing} | {low} | {high} | {buckets} |"


def _queue(marks: list[OpenMark], report: ValidationReport) -> list[str]:
    kinds = _kind_counts(marks)
    classes = _class_counts(marks)
    lines = ["| Kind | Open |", "| --- | --- |"]
    lines.extend(f"| `{kind}` | {kinds[kind]} |" for kind in MARKER_KINDS)
    lines.extend(["", "| Uncertainty class | Open |", "| --- | --- |"])
    lines.extend(f"| `{name}` | {classes[name]} |" for name in (*UNCERTAINTY_CLASSES, "none"))
    lines.append("")
    lines.append(f"Orphaned overlay entries: {report.facts.orphans}.")
    for reason, count in report.facts.orphaned_by_reason.items():
        lines.append(f"- `{reason}`: {count}")
    return lines


def _kind_counts(marks: list[OpenMark]) -> dict[str, int]:
    counts = Counter(mark.kind for mark in marks)
    return {kind: counts.get(kind, 0) for kind in MARKER_KINDS}


def _class_counts(marks: list[OpenMark]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for mark in marks:
        if mark.classes:
            counts.update(mark.classes)
        else:
            counts["none"] += 1
    return {name: counts.get(name, 0) for name in (*UNCERTAINTY_CLASSES, "none")}


def _manual(pages: tuple[PageRow, ...] | list[PageRow], marks: list[OpenMark]) -> list[str]:
    lines = [
        f"- PDF {row.pdf_page} `{row.page_id}`: failed ({row.reason})"
        for row in pages
        if row.status == "failed"
    ]
    grouped: Counter[tuple[int, str, str, str]] = Counter()
    page_ids = {row.pdf_page: row.page_id for row in pages}
    for mark in marks:
        grouped[(mark.pdf_page, page_ids.get(mark.pdf_page, ""), mark.kind, mark.block_kind)] += 1
    for (page, pid, kind, block_kind), count in sorted(grouped.items()):
        lines.append(f"- PDF {page} `{pid}`: `{kind}` {block_kind} x{count}")
    if not lines:
        return ["No page needs a person."]
    totals = Counter(mark.block_kind for mark in marks)
    if totals:
        rendered = ", ".join(f"{name} {count}" for name, count in sorted(totals.items()))
        lines.append(f"- Content types still open: {rendered}")
    return lines


def _checks(report: ValidationReport) -> list[str]:
    if not report.findings:
        return ["None."]
    return [f"- `{item.file}`: {item.check}: {item.detail}" for item in report.findings]


def _confidence_json(scores: dict[str, EngineScores]) -> dict[str, object]:
    by_engine = {}
    for engine in sorted(scores):
        item = scores[engine]
        by_engine[engine] = {
            "buckets": {
                name: item.buckets.get(name, 0) for name, _low, _high in CONFIDENCE_BUCKETS
            },
            "count": item.count,
            "maximum": item.maximum,
            "minimum": item.minimum,
            "missing": item.missing,
        }
    return {"by_engine": by_engine, "note": CONFIDENCE_NOTE}
