"""Injected-I/O interactive calibration rating session."""

import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

from llb.judge.calibration import load_worksheet
from llb.judge.rate.commands import (
    ANSWER,
    CLEAR,
    HELP,
    JUMP,
    NEXT,
    NOTE,
    PREV,
    PROMPT_HINT,
    QUIT,
    RATE,
    RATING_MAX,
    RATING_MIN,
    UNRATED,
    _ESC,
    parse_command,
)
from llb.judge.rate.presentation import completion_panel, format_card, help_text, summary_lines
from llb.judge.rate.state import (
    advanced_index,
    clear_human_columns,
    clear_rating,
    first_unrated_index,
    rated_count,
    save_human_columns,
    set_rating,
)


class _Quit(Exception):
    """End the session on quit, EOF, or exhausted injected input."""


def _default_output(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _read(prompt: str, inputs: Iterator[str] | None, emit: Callable[[str], None]) -> str:
    if inputs is None:
        return input(prompt)
    emit(prompt)
    try:
        return next(inputs)
    except StopIteration as exc:
        raise _Quit from exc


def _go_forward(
    index: int, total: int, rows: Sequence[dict[str, str]], emit: Callable[[str], None]
) -> int:
    new_index = advanced_index(index, total, rows)
    if index == total - 1 and new_index < total:
        emit(f"[calibration] {total - rated_count(rows)} item(s) still unrated -- jumping there.")
    return new_index


def _jump_to(index: int, target: int, total: int, emit: Callable[[str], None]) -> int:
    if 1 <= target <= total:
        return target - 1
    emit(f"[calibration] item out of range 1..{total}: {target}")
    return index


def _go_unrated(index: int, rows: Sequence[dict[str, str]], emit: Callable[[str], None]) -> int:
    next_index = first_unrated_index(rows)
    if (rows[next_index].get("human_rating") or "").strip():
        emit("[calibration] all items are rated.")
        return index
    return next_index


def _completion_action(
    index: int,
    total: int,
    rows: Sequence[dict[str, str]],
    emit: Callable[[str], None],
    inputs: Iterator[str] | None,
) -> tuple[int, bool]:
    if index < total:
        return index, False
    emit(completion_panel(rows, total))
    command = parse_command(_read("review (p / j <N> / u) or finish (Enter / q) > ", inputs, emit))
    if command.kind in (QUIT, NEXT):
        raise _Quit
    if command.kind == HELP:
        emit(help_text())
    elif command.kind == PREV:
        index = total - 1
    elif command.kind == JUMP:
        index = _jump_to(index, command.value or 0, total, emit)
    elif command.kind == UNRATED:
        index = _go_unrated(index, rows, emit)
    else:
        emit("[calibration] all rated -- p to review, j <N> to jump, Enter/q to finish.")
    return index, True


def _row_action(
    rows: Sequence[dict[str, str]],
    index: int,
    total: int,
    emit: Callable[[str], None],
    inputs: Iterator[str] | None,
    path: Path,
    fieldnames: Sequence[str],
    show_judge: bool,
) -> int:
    row = rows[index]
    emit(format_card(row, index + 1, total, rated_count(rows), show_judge=show_judge))
    command = parse_command(_read(f"{PROMPT_HINT}\nrating> ", inputs, emit))
    if command.kind == QUIT:
        raise _Quit
    if command.kind == HELP:
        emit(help_text())
    elif command.kind == NEXT:
        index = _go_forward(index, total, rows, emit)
    elif command.kind == PREV:
        index = max(index - 1, 0)
    elif command.kind == JUMP:
        index = _jump_to(index, command.value or 0, total, emit)
    elif command.kind == UNRATED:
        index = _go_unrated(index, rows, emit)
    elif command.kind == RATE:
        set_rating(row, command.value or RATING_MIN)
        save_human_columns(path, rows, fieldnames)
        index = _go_forward(index, total, rows, emit)
    elif command.kind == CLEAR:
        clear_rating(row)
        save_human_columns(path, rows, fieldnames)
    elif command.kind == ANSWER:
        row["human_answer"] = _read("your answer (empty to clear): ", inputs, emit).strip()
        save_human_columns(path, rows, fieldnames)
    elif command.kind == NOTE:
        row["human_note"] = _read("note (empty to clear): ", inputs, emit).strip()
        save_human_columns(path, rows, fieldnames)
    elif command.raw.startswith(_ESC):
        emit("[calibration] arrow keys garbled -- use n (next) / p (prev).")
    else:
        emit(f"[calibration] not a command: {command.raw!r} (? for help; 1-5 to rate).")
    return index


def _clear_if_requested(
    clear: bool,
    rows: Sequence[dict[str, str]],
    path: Path,
    fieldnames: Sequence[str],
    inputs: Iterator[str] | None,
    emit: Callable[[str], None],
) -> bool:
    if not clear:
        return True
    answer = _read(
        "clear ALL human ratings/answers and start fresh? type 'yes' to confirm: ", inputs, emit
    )
    if answer.strip().lower() != "yes":
        emit("[calibration] clear aborted; nothing changed.")
        return False
    clear_human_columns(rows)
    save_human_columns(path, rows, fieldnames)
    emit("[calibration] cleared all human columns.")
    return True


def run_session(
    worksheet_path: Path | str,
    *,
    inputs: Iterable[str] | None = None,
    output: Callable[[str], None] | None = None,
    start: int | None = None,
    show_judge: bool = False,
    clear: bool = False,
) -> int:
    """Drive the interactive rater and return the final rated count."""
    path = Path(worksheet_path)
    emit = output or _default_output
    input_iterator = iter(inputs) if inputs is not None else None
    rows, fieldnames = load_worksheet(path)
    if not rows:
        emit(f"[calibration] worksheet has no rows: {path}")
        return 0
    if not _clear_if_requested(clear, rows, path, fieldnames, input_iterator, emit):
        return rated_count(rows)
    total = len(rows)
    if start is not None:
        index = max(0, min(start - 1, total - 1))
    elif rated_count(rows) == total:
        index = total
    else:
        index = first_unrated_index(rows)
    emit(
        "judge calibration -- rate each model answer against the reference, "
        f"{RATING_MIN} (wrong) to {RATING_MAX} (fully correct)."
    )
    emit(help_text())
    try:
        while True:
            index, completed = _completion_action(index, total, rows, emit, input_iterator)
            if completed:
                continue
            index = _row_action(
                rows, index, total, emit, input_iterator, path, fieldnames, show_judge
            )
    except (_Quit, EOFError):
        pass
    except KeyboardInterrupt:
        emit("")
    save_human_columns(path, rows, fieldnames)
    for line in summary_lines(rows, path):
        emit(line)
    return rated_count(rows)
