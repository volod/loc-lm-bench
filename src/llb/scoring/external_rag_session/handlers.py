"""Per-screen handlers for the review loop: the completion screen and single-row editing."""

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from llb.scoring.external_rag import (
    HUMAN_CORRECTED_ANSWER_FIELD,
    HUMAN_NOTES_FIELD,
    ExternalRagResult,
    clear_human_fields,
    write_jsonl,
)
from llb.scoring.external_rag_session.cards import _float_text, completion_panel, format_card
from llb.scoring.external_rag_session.commands import (
    CLEAR,
    CORRECT,
    DECISION,
    HELP,
    JUMP,
    NEXT,
    NOTE,
    PREV,
    PROMPT_HINT,
    QUIT,
    SCORE,
    UNSCORED,
    help_text,
    parse_command,
)
from llb.scoring.external_rag_session.prompt import _Quit, _emit_unknown, _read
from llb.scoring.external_rag_session.records import (
    _advanced_index,
    _clear_row,
    _set_decision,
    _set_explicit_score,
    first_unscored_index,
    reviewed_count,
)


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
