"""Terminal I/O primitives and the navigation commands of the verify session.

The injected input iterator + output sink make the whole loop unit-testable without a terminal;
the navigation handlers (next / prev / jump / undecided, the completion screen) move the cursor and
never touch the worksheet decision state.
"""

import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from llb.goldset.verify_base import (
    ACCEPT,
    FAIL,
    HUMAN_COLS,
    PASS,
    REJECT,
    STATUS_DECIDED,
    STATUS_PENDING,
)
from llb.goldset.verify_card import (
    HELP,
    JUMP,
    NEXT,
    PREV,
    QUIT,
    UNDECIDED,
    Command,
    _is_synthetic_row,
    help_text,
    parse_command,
)
from llb.goldset.verify_session.report import (
    _advanced_index,
    clear_human_columns,
    completion_panel,
    decided_count,
    first_undecided_index,
    save_human_columns,
)


class _Quit(Exception):
    """Internal: end the session (q, EOF, or exhausted injected input)."""


def _default_output(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _stdin_reader(prompt: str) -> str:
    return input(prompt)


def _emit_intro(emit: Callable[[str], None]) -> None:
    emit(
        "human verification gate data verification -- verify each sampled item against the "
        "corpus, then accept/reject."
    )
    emit(help_text())


def _read(prompt: str, it: Iterator[str] | None, emit: Callable[[str], None]) -> str:
    if it is None:
        return _stdin_reader(prompt)
    emit(prompt)
    try:
        return next(it)
    except StopIteration as exc:
        raise _Quit from exc


def _save(path: Path, rows: Sequence[dict[str, str]], fieldnames: Sequence[str]) -> None:
    save_human_columns(path, rows, fieldnames)


def _maybe_clear_human_columns(
    clear: bool,
    rows: Sequence[dict[str, str]],
    path: Path,
    fieldnames: Sequence[str],
    it: Iterator[str] | None,
    emit: Callable[[str], None],
) -> bool:
    if not clear:
        return True
    ans = _read(
        "clear ALL human marks/decisions and start fresh? type 'yes' to confirm: ", it, emit
    )
    if ans.strip().lower() != "yes":
        emit("[verify] clear aborted; nothing changed.")
        return False
    clear_human_columns(rows)
    _save(path, rows, fieldnames)
    emit("[verify] cleared all human columns.")
    return True


def _get_idx(start: int | None, total: int, rows: Sequence[dict[str, str]]) -> int:
    if start is not None:
        return max(0, min(start - 1, total - 1))
    if decided_count(rows) == total:
        return total
    return first_undecided_index(rows)


def _set_decision(row: dict[str, str], decision: str) -> None:
    row["decision"] = decision
    row["human_status"] = STATUS_DECIDED


def _clear_row(row: dict[str, str]) -> None:
    for col in HUMAN_COLS:
        row[col] = ""
    row["human_status"] = STATUS_PENDING


def _go_forward(
    idx: int, total: int, rows: Sequence[dict[str, str]], emit: Callable[[str], None]
) -> int:
    new = _advanced_index(idx, total, rows)
    if idx == total - 1 and new < total:
        remaining = total - decided_count(rows)
        emit(f"[verify] {remaining} item(s) still undecided -- jumping there.")
    return new


def _jump_to(idx: int, target: int, total: int, emit: Callable[[str], None]) -> int:
    if 1 <= target <= total:
        return target - 1
    emit(f"[verify] item out of range 1..{total}: {target}")
    return idx


def _go_undecided(idx: int, rows: Sequence[dict[str, str]], emit: Callable[[str], None]) -> int:
    next_idx = first_undecided_index(rows)
    if (rows[next_idx].get("decision") or "").strip() in (ACCEPT, REJECT):
        emit("[verify] all items are decided.")
        return idx
    return next_idx


def _handle_completion_screen(
    idx: int,
    total: int,
    rows: Sequence[dict[str, str]],
    emit: Callable[[str], None],
    it: Iterator[str] | None,
) -> tuple[int, bool]:
    is_completion = idx >= total
    if not is_completion:
        return idx, False

    emit(completion_panel(rows, total))
    cmd = parse_command(_read("review (b / j <N> / u) or finish (Enter / q) > ", it, emit))
    if cmd.kind in (QUIT, NEXT):
        raise _Quit
    if cmd.kind == HELP:
        emit(help_text())
    elif cmd.kind == PREV:
        idx = total - 1
    elif cmd.kind == JUMP:
        idx = _jump_to(idx, cmd.value if isinstance(cmd.value, int) else 0, total, emit)
    elif cmd.kind == UNDECIDED:
        idx = _go_undecided(idx, rows, emit)
    else:
        emit("[verify] all decided -- b to review, j <N> to jump, Enter/q to finish.")
    return idx, True


def _set_check(row: dict[str, str], cmd: Command, emit: Callable[[str], None]) -> bool:
    if cmd.field == "chk_planted" and not _is_synthetic_row(row):
        emit("[verify] planted check is N/A for a real (non-synthetic) item.")
        return False
    row[cmd.field] = PASS if cmd.value else FAIL
    return True


def _handle_navigation_action(
    cmd: Command,
    idx: int,
    total: int,
    rows: Sequence[dict[str, str]],
    emit: Callable[[str], None],
) -> tuple[int, bool]:
    if cmd.kind == HELP:
        emit(help_text())
    elif cmd.kind == NEXT:
        idx = _go_forward(idx, total, rows, emit)
    elif cmd.kind == PREV:
        idx = max(idx - 1, 0)
    elif cmd.kind == JUMP:
        idx = _jump_to(idx, cmd.value if isinstance(cmd.value, int) else 0, total, emit)
    elif cmd.kind == UNDECIDED:
        idx = _go_undecided(idx, rows, emit)
    else:
        return idx, False
    return idx, True
