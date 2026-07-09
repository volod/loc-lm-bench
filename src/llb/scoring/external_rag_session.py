"""Interactive human review session for external RAG answer logs.

The answered JSONL is the session state. Each human edit rewrites the file atomically, so a
reviewer can quit or interrupt and later resume at the first unscored row. CSV/report artifacts are
written only when every row has a human score and decision.
"""

import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.scoring.external_rag import (
    DEFAULT_SOURCE_LIMIT,
    HUMAN_CORRECTED_ANSWER_FIELD,
    HUMAN_DECISION_ACCEPT,
    HUMAN_DECISION_FIELD,
    HUMAN_DECISION_PARTIAL,
    HUMAN_DECISION_REJECT,
    HUMAN_FIELDS,
    HUMAN_NOTES_FIELD,
    HUMAN_SCORE_FIELD,
    HUMAN_STATUS_FIELD,
    HUMAN_STATUS_SCORED,
    ExternalRagResult,
    clean_answer_for_scoring,
    clear_human_fields,
    ensure_human_fields,
    human_reviewed_count,
    is_human_scored,
    load_jsonl,
    score_external_rag_file,
    score_records,
    write_jsonl,
    _field_value,
    _source_list,
    _string,
)

NEXT = "next"
PREV = "prev"
JUMP = "jump"
UNSCORED = "unscored"
CLEAR = "clear"
HELP = "help"
QUIT = "quit"
NOTE = "note"
CORRECT = "correct"
DECISION = "decision"
SCORE = "score"
UNKNOWN = "unknown"

ACCEPT_SCORE = 1.0
PARTIAL_SCORE = 0.5
REJECT_SCORE = 0.0
ACCEPT_THRESHOLD = 0.85
TEXT_PREVIEW_CHARS = 1200
SOURCE_PREVIEW_CHARS = 700
SPAN_LIMIT = 3
_ESC = "\x1b"
_ARROWS = {f"{_ESC}[A": PREV, f"{_ESC}[D": PREV, f"{_ESC}[B": NEXT, f"{_ESC}[C": NEXT}
_SIMPLE_COMMANDS = {
    "": NEXT,
    "n": NEXT,
    "b": PREV,
    "u": UNSCORED,
    "c": CLEAR,
    "q": QUIT,
    "quit": QUIT,
    "?": HELP,
    "h": HELP,
    "help": HELP,
    "o": NOTE,
    "note": NOTE,
    "w": CORRECT,
    "corr": CORRECT,
    "correct": CORRECT,
    "a": DECISION,
    "accept": DECISION,
    "p": DECISION,
    "partial": DECISION,
    "r": DECISION,
    "reject": DECISION,
}

PROMPT_HINT = (
    "score: a=accept(1.0), p=partial(0.5), r=reject(0.0), s 0..1=custom score\n"
    "edit/nav: o=note, w=corrected answer, c=clear, Enter/n=next, "
    "b=prev, u=unscored, j<N>=jump, ?=help, q=quit"
)


@dataclass(frozen=True)
class Command:
    """A parsed prompt line."""

    kind: str
    value: str | float | int | None = None
    raw: str = ""


@dataclass(frozen=True)
class ExternalRagReviewResult:
    """Result of an interactive external RAG review session."""

    reviewed: int
    total: int
    complete: bool
    score_result: ExternalRagResult | None = None


class _Quit(Exception):
    """Internal end-of-session signal."""


def parse_command(raw: str) -> Command:
    """Parse one prompt command."""
    s = raw.strip()
    if s in _ARROWS:
        return Command(_ARROWS[s])
    lower = s.lower()
    if lower.startswith("j"):
        rest = lower[1:].strip()
        if rest.isdigit():
            return Command(JUMP, int(rest))
        return Command(UNKNOWN, raw=s)
    if lower.startswith("s "):
        return _parse_score(lower[2:].strip(), s)
    if lower in _SIMPLE_COMMANDS:
        if _SIMPLE_COMMANDS[lower] == DECISION:
            return Command(DECISION, _decision_from_text(lower))
        return Command(_SIMPLE_COMMANDS[lower])
    score_command = _parse_score(lower, s)
    if score_command.kind != UNKNOWN:
        return score_command
    return Command(UNKNOWN, raw=s)


def first_unscored_index(records: Sequence[dict[str, Any]]) -> int:
    """Index of the first record without complete human score + decision; 0 if all scored."""
    for index, record in enumerate(records):
        if not is_human_scored(record):
            return index
    return 0


def reviewed_count(records: Sequence[dict[str, Any]]) -> int:
    """How many records have complete human score + decision."""
    return human_reviewed_count(records)


def format_card(
    record: dict[str, Any],
    position: int,
    total: int,
    reviewed: int,
    *,
    answer_field: str | None,
    sources_field: str | None,
    error_field: str | None,
    source_limit: int,
    strip_source_footer: bool,
) -> str:
    """Render the current row for human scoring."""
    scored = score_records(
        [record],
        answer_field=answer_field,
        sources_field=sources_field,
        error_field=error_field,
        source_limit=source_limit,
        strip_source_footer=strip_source_footer,
    )[0]
    raw_answer, _answer_field_used = _field_value(
        record,
        answer_field,
        ("llm_answer", "predicted_answer", "model_answer", "answer"),
    )
    raw_error, _error_field_used = _field_value(record, error_field, ("llm_error", "error"))
    raw_sources, _sources_field_used = _field_value(
        record, sources_field, ("llm_sources", "sources", "retrieved_sources")
    )
    remaining = total - reviewed
    answer = _string(raw_answer)
    scored_answer = clean_answer_for_scoring(answer, strip_source_footer=strip_source_footer)
    lines = [
        "===== external RAG human review =====",
        f"item {position}/{total} (reviewed {reviewed}, remaining {remaining})",
        f"== id: {_string(record.get('id'))}",
        f"== meta: split={_string(record.get('split'))} "
        f"source_doc_id={_string(record.get('source_doc_id'))} "
        f"verified={_string(record.get('verified'))}",
        f"== auto_score: status={_string(scored.get('status'))} "
        f"objective={_float_text(scored.get('objective_score'))} "
        f"exact/token/contains={_float_text(scored.get('exact'))}/"
        f"{_float_text(scored.get('token_f1'))}/{_float_text(scored.get('contains'))}",
        "",
        f"== question: {_preview_one_line(_string(record.get('question')), TEXT_PREVIEW_CHARS)}",
        f"== reference_answer: "
        f"{_preview_one_line(_string(record.get('reference_answer')), TEXT_PREVIEW_CHARS)}",
        "== gold_source_text",
        *_source_span_lines(record),
        f"== llm_answer: {_preview_one_line(answer, TEXT_PREVIEW_CHARS)}",
        f"== scored_answer: {_preview_one_line(scored_answer, TEXT_PREVIEW_CHARS)}",
        "== llm_sources",
        *_returned_source_lines(raw_sources, source_limit),
        f"== llm_error: {_preview_one_line(_string(raw_error), TEXT_PREVIEW_CHARS) or '(none)'}",
        f"== human: {HUMAN_SCORE_FIELD}={_field(record, HUMAN_SCORE_FIELD, '(unscored)')} "
        f"{HUMAN_DECISION_FIELD}={_field(record, HUMAN_DECISION_FIELD, '(unset)')}",
        f"== human_notes: {_field(record, HUMAN_NOTES_FIELD, '')}",
        f"== human_corrected_answer: {_field(record, HUMAN_CORRECTED_ANSWER_FIELD, '')}",
    ]
    return "\n".join(lines)


def help_text() -> str:
    """Command reference for the review prompt."""
    return "\n".join(
        [
            "score commands:",
            "  a        accept, score=1.0",
            "  p        partial, score=0.5",
            "  r        reject, score=0.0",
            "  s 0..1   set an explicit human score, for example: s 0.8",
            "edits:",
            "  o        edit human_notes",
            "  w        edit human_corrected_answer",
            "  c        clear this row's human fields",
            "navigation:",
            "  n/Enter  next",
            "  b        previous",
            "  j <N>    jump to row N",
            "  u        first unscored row",
            "  q        save and quit",
            "The JSONL is saved after each edit. CSV/report are written only after all rows are "
            "scored.",
        ]
    )


def completion_panel(records: Sequence[dict[str, Any]]) -> str:
    """All-reviewed screen."""
    counts = _decision_counts(records)
    decision_text = ", ".join(f"{key}={value}" for key, value in counts.items()) or "none"
    return "\n".join(
        [
            f"===== all {len(records)} rows scored ({decision_text}) =====",
            "  review/change: b = last row, j <N> = jump to row N",
            "  finish: press Enter or q to save JSONL and write CSV/report",
        ]
    )


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
    )
    _emit_final_result(result, emit)
    return ExternalRagReviewResult(reviewed, total, True, result)


def _parse_score(value: str, raw: str) -> Command:
    try:
        score = float(value)
    except ValueError:
        return Command(UNKNOWN, raw=raw)
    if 0.0 <= score <= 1.0:
        return Command(SCORE, score)
    return Command(UNKNOWN, raw=raw)


def _decision_from_text(value: str) -> str:
    if value in ("a", "accept"):
        return HUMAN_DECISION_ACCEPT
    if value in ("p", "partial"):
        return HUMAN_DECISION_PARTIAL
    return HUMAN_DECISION_REJECT


def _set_decision(record: dict[str, Any], decision: str) -> None:
    score = {
        HUMAN_DECISION_ACCEPT: ACCEPT_SCORE,
        HUMAN_DECISION_PARTIAL: PARTIAL_SCORE,
        HUMAN_DECISION_REJECT: REJECT_SCORE,
    }[decision]
    _set_score_and_decision(record, score, decision)


def _set_explicit_score(record: dict[str, Any], score: float) -> None:
    if score >= ACCEPT_THRESHOLD:
        decision = HUMAN_DECISION_ACCEPT
    elif score > 0.0:
        decision = HUMAN_DECISION_PARTIAL
    else:
        decision = HUMAN_DECISION_REJECT
    _set_score_and_decision(record, score, decision)


def _set_score_and_decision(record: dict[str, Any], score: float, decision: str) -> None:
    record[HUMAN_SCORE_FIELD] = f"{score:g}"
    record[HUMAN_DECISION_FIELD] = decision
    record[HUMAN_STATUS_FIELD] = HUMAN_STATUS_SCORED


def _clear_row(record: dict[str, Any]) -> None:
    for field in HUMAN_FIELDS:
        record[field] = ""


def _advanced_index(index: int, records: Sequence[dict[str, Any]]) -> int:
    total = len(records)
    if index < total - 1:
        return index + 1
    if reviewed_count(records) == total:
        return total
    return first_unscored_index(records)


def _get_index(start: int | None, total: int, records: Sequence[dict[str, Any]]) -> int:
    if start is not None:
        return max(0, min(start - 1, total - 1))
    if reviewed_count(records) == total:
        return total
    return first_unscored_index(records)


def _handle_completion_screen(
    index: int,
    records: Sequence[dict[str, Any]],
    emit: Callable[[str], None],
    it: Iterator[str] | None,
) -> tuple[int, bool]:
    total = len(records)
    if index < total:
        return index, False
    emit(completion_panel(records))
    cmd = parse_command(_read("review (b / j <N>) or finish (Enter / q) > ", it, emit))
    if cmd.kind in (QUIT, NEXT):
        raise _Quit
    if cmd.kind == HELP:
        emit(help_text())
    elif cmd.kind == PREV:
        index = total - 1
    elif cmd.kind == JUMP and isinstance(cmd.value, int):
        index = _jump(index, cmd.value, total, emit)
    elif cmd.kind == UNSCORED:
        emit("[score-external-rag] all rows are scored.")
    else:
        emit("[score-external-rag] all scored -- b to review, j <N> to jump, Enter/q to finish.")
    return index, True


def _handle_row(
    records: Sequence[dict[str, Any]],
    index: int,
    path: Path,
    emit: Callable[[str], None],
    it: Iterator[str] | None,
    *,
    answer_field: str | None,
    sources_field: str | None,
    error_field: str | None,
    source_limit: int,
    strip_source_footer: bool,
) -> int:
    total = len(records)
    record = records[index]
    emit(
        format_card(
            record,
            index + 1,
            total,
            reviewed_count(records),
            answer_field=answer_field,
            sources_field=sources_field,
            error_field=error_field,
            source_limit=source_limit,
            strip_source_footer=strip_source_footer,
        )
    )
    cmd = parse_command(_read(f"{PROMPT_HINT}\nscore> ", it, emit))
    if cmd.kind == QUIT:
        raise _Quit
    if cmd.kind == HELP:
        emit(help_text())
    elif cmd.kind == NEXT:
        index = _advanced_index(index, records)
    elif cmd.kind == PREV:
        index = max(index - 1, 0)
    elif cmd.kind == JUMP and isinstance(cmd.value, int):
        index = _jump(index, cmd.value, total, emit)
    elif cmd.kind == UNSCORED:
        index = first_unscored_index(records)
    elif cmd.kind == CLEAR:
        _clear_row(record)
        write_jsonl(path, records)
    elif cmd.kind == NOTE:
        record[HUMAN_NOTES_FIELD] = _read("human_notes (empty to clear): ", it, emit).strip()
        write_jsonl(path, records)
    elif cmd.kind == CORRECT:
        record[HUMAN_CORRECTED_ANSWER_FIELD] = _read(
            "human_corrected_answer (empty to clear): ", it, emit
        ).strip()
        write_jsonl(path, records)
    elif cmd.kind == DECISION and isinstance(cmd.value, str):
        _set_decision(record, cmd.value)
        write_jsonl(path, records)
        index = _advanced_index(index, records)
    elif cmd.kind == SCORE and isinstance(cmd.value, float):
        _set_explicit_score(record, cmd.value)
        write_jsonl(path, records)
        index = _advanced_index(index, records)
    else:
        _emit_unknown(cmd, emit)
    return index


def _maybe_clear(
    records: Sequence[dict[str, Any]],
    path: Path,
    it: Iterator[str] | None,
    emit: Callable[[str], None],
) -> bool:
    answer = _read(
        "clear ALL external RAG human scores in the JSONL? type 'yes' to confirm: ", it, emit
    )
    if answer.strip().lower() != "yes":
        emit("[score-external-rag] clear aborted; nothing changed.")
        return False
    clear_human_fields(records)
    write_jsonl(path, records)
    emit("[score-external-rag] cleared all human fields.")
    return True


def _jump(current: int, target: int, total: int, emit: Callable[[str], None]) -> int:
    if 1 <= target <= total:
        return target - 1
    emit(f"[score-external-rag] row out of range 1..{total}: {target}")
    return current


def _emit_final_result(result: ExternalRagResult, emit: Callable[[str], None]) -> None:
    summary = result.summary
    emit(
        f"[score-external-rag] rows={summary['n']} "
        f"objective={_float_text(summary.get('objective_mean'))} "
        f"exact={_float_text(summary.get('exact_rate'))} "
        f"contains={_float_text(summary.get('contains_rate'))} "
        f"human_mean={_float_text(summary.get('human_score_mean'))}"
    )
    emit(f"[score-external-rag] csv -> {result.paths.csv}")
    emit(f"[score-external-rag] report -> {result.paths.report}")


def _read(prompt: str, it: Iterator[str] | None, emit: Callable[[str], None]) -> str:
    if it is None:
        return input(prompt)
    emit(prompt)
    try:
        return next(it)
    except StopIteration as exc:
        raise _Quit from exc


def _default_output(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _emit_unknown(cmd: Command, emit: Callable[[str], None]) -> None:
    if cmd.raw.startswith(_ESC):
        emit("[score-external-rag] arrow keys garbled -- use n (next) / b (prev).")
    else:
        emit(f"[score-external-rag] not a command: {cmd.raw!r} (? for help).")


def _source_span_lines(record: dict[str, Any]) -> list[str]:
    spans = record.get("source_spans")
    if not isinstance(spans, list) or not spans:
        return ["  (none)"]
    lines: list[str] = []
    for index, span in enumerate(spans[:SPAN_LIMIT], 1):
        if not isinstance(span, dict):
            continue
        doc_id = _string(span.get("doc_id"))
        start = _string(span.get("char_start"))
        end = _string(span.get("char_end"))
        text = _preview(_string(span.get("text")), SOURCE_PREVIEW_CHARS)
        lines.append(f"  span {index}: doc={doc_id} chars={start}-{end}")
        lines.append(_indent(text or "(empty)", prefix="    "))
    return lines or ["  (none)"]


def _returned_source_lines(raw_sources: object, limit: int) -> list[str]:
    if limit <= 0:
        return ["  (source display disabled)"]
    sources = _source_list(raw_sources)
    if not sources:
        return ["  (none)"]
    lines: list[str] = []
    for index, source in enumerate(sources[:limit], 1):
        title = _string(source.get("article_title") or source.get("title") or source.get("name"))
        article_id = _string(source.get("article_id") or source.get("id"))
        doc_id = _string(source.get("doc_id") or source.get("document_id"))
        score = _string(source.get("score"))
        url = _string(source.get("url") or source.get("uri"))
        text = _string(source.get("text") or source.get("snippet") or source.get("content"))
        lines.append(
            f"  source {index}: title={title or '(none)'} id={article_id or '(none)'} "
            f"doc={doc_id or '(none)'} score={score or '(none)'} url={url or '(none)'}"
        )
        if text:
            lines.append(_indent(_preview(text, SOURCE_PREVIEW_CHARS), prefix="    "))
    return lines


def _field(record: dict[str, Any], name: str, blank: str) -> str:
    value = _string(record.get(name)).strip()
    return value if value else blank


def _decision_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = {
        HUMAN_DECISION_ACCEPT: 0,
        HUMAN_DECISION_PARTIAL: 0,
        HUMAN_DECISION_REJECT: 0,
    }
    for record in records:
        decision = _string(record.get(HUMAN_DECISION_FIELD)).strip().lower()
        if decision in counts:
            counts[decision] += 1
    return {key: value for key, value in counts.items() if value}


def _indent(text: str, prefix: str = "  ") -> str:
    if not text:
        return prefix.rstrip()
    return "\n".join(prefix + line for line in text.splitlines())


def _preview(text: str, limit: int) -> str:
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)] + "..."


def _preview_one_line(text: str, limit: int) -> str:
    return _preview(" ".join(text.split()), limit)


def _float_text(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    if not isinstance(value, str):
        return "0.0000"
    try:
        return f"{float(value):.4f}"
    except ValueError:
        return "0.0000"
