"""End-to-end external-RAG scoring artifact orchestration."""

from pathlib import Path

from llb.scoring.external_rag.records import load_jsonl
from llb.scoring.external_rag.score import score_records
from llb.scoring.external_rag.worksheet import write_csv
from llb.scoring.external_rag_common import (
    DEFAULT_SOURCE_LIMIT,
    ExternalRagPaths,
    ExternalRagResult,
)
from llb.scoring.external_rag_report import write_report
from llb.scoring.external_rag_source_map import load_source_map
from llb.scoring.external_rag_sources import summarize_source_audit
from llb.scoring.external_rag_summary import summarize


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
    source_map_path: Path | None = None,
) -> ExternalRagResult:
    """Score answered JSONL and write its detailed CSV and Markdown report."""
    if source_limit < 0:
        raise ValueError("source_limit must be >= 0")
    records = load_jsonl(answers_path)
    if not records:
        raise ValueError(f"{answers_path}: no JSONL records found")
    source_map = load_source_map(source_map_path) if source_map_path is not None else None
    scored = score_records(
        records,
        answer_field=answer_field,
        sources_field=sources_field,
        error_field=error_field,
        source_limit=source_limit,
        strip_source_footer=strip_source_footer,
        source_map=source_map,
    )
    csv_path = csv_out or answers_path.with_suffix(".csv")
    report_path = report_out or answers_path.with_name(f"{answers_path.stem}.report.md")
    summary = summarize(scored, answers_path=answers_path, label=label)
    if source_map is not None:
        summary["source_audit"] = summarize_source_audit(scored)
    write_csv(scored, csv_path, source_limit=source_limit, source_audit=source_map is not None)
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
