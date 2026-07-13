"""The JSONL-backed interactive review driver: navigate rows, then finalize artifacts."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from llb.scoring.external_rag import (
    DEFAULT_SOURCE_LIMIT,
    ExternalRagResult,
    ensure_human_fields,
    load_jsonl,
    score_external_rag_file,
    write_jsonl,
)
from llb.scoring.external_rag_session.handlers import (
    _emit_final_result,
    _handle_completion_screen,
    _handle_row,
    _maybe_clear,
)
from llb.scoring.external_rag_session.prompt import _Quit, _default_output
from llb.scoring.external_rag_session.records import _get_index, reviewed_count


@dataclass(frozen=True)
class ExternalRagReviewResult:
    """Result of an interactive external RAG review session."""

    reviewed: int
    total: int
    complete: bool
    score_result: ExternalRagResult | None = None


def run_external_rag_session(
    answers_path: Path | str,
    *,
    csv_out: Path | None = None,
    report_out: Path | None = None,
    answer_field: str | None = None,
    sources_field: str | None = None,
    error_field: str | None = None,
    source_limit: int = DEFAULT_SOURCE_LIMIT,
    strip_source_footer: bool = True,
    label: str | None = None,
    start: int | None = None,
    clear: bool = False,
    source_map: Path | None = None,
    inputs: Iterable[str] | None = None,
    output: Callable[[str], None] | None = None,
) -> ExternalRagReviewResult:
    """Drive the JSONL-backed interactive review and finalize artifacts when complete."""
    path = Path(answers_path)
    emit = output or _default_output
    it: Iterator[str] | None = iter(inputs) if inputs is not None else None
    if source_limit < 0:
        raise ValueError("source_limit must be >= 0")

    records = load_jsonl(path)
    if not records:
        raise ValueError(f"{path}: no JSONL records found")
    if ensure_human_fields(records):
        write_jsonl(path, records)

    if clear and not _maybe_clear(records, path, it, emit):
        count = reviewed_count(records)
        return ExternalRagReviewResult(count, len(records), count == len(records))

    total = len(records)
    index = _get_index(start, total, records)

    try:
        while True:
            index, handled = _handle_completion_screen(index, records, emit, it)
            if handled:
                continue
            index = _handle_row(
                records,
                index,
                path,
                emit,
                it,
                answer_field=answer_field,
                sources_field=sources_field,
                error_field=error_field,
                source_limit=source_limit,
                strip_source_footer=strip_source_footer,
            )
    except (_Quit, EOFError):
        pass
    except KeyboardInterrupt:
        emit("")

    write_jsonl(path, records)
    reviewed = reviewed_count(records)
    complete = reviewed == total
    if not complete:
        emit(f"[score-external-rag] saved {path}")
        emit(
            f"[score-external-rag] progress: {reviewed}/{total} scored, "
            f"{total - reviewed} remaining"
        )
        emit(
            "[score-external-rag] resume: make score-external-rag "
            "EXTERNAL_RAG_ANSWERS=<answered-jsonl>"
        )
        emit("[score-external-rag] csv/report: not written until all rows are scored")
        return ExternalRagReviewResult(reviewed, total, False)

    result = score_external_rag_file(
        path,
        csv_out=csv_out,
        report_out=report_out,
        answer_field=answer_field,
        sources_field=sources_field,
        error_field=error_field,
        source_limit=source_limit,
        strip_source_footer=strip_source_footer,
        label=label,
        source_map_path=source_map,
    )
    _emit_final_result(result, emit)
    return ExternalRagReviewResult(reviewed, total, True, result)
