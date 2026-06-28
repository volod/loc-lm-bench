"""Interactive human verifier for the human verification gate sample worksheet.

A terminal session that walks a stratified sample item by item and writes the HUMAN columns
(the four checks, the accept/reject decision, a note, a status) in place. Interactive I/O lives
here, OUT of the pure `verify.py`; the two share the worksheet schema + atomic load/save. This
mirrors how `judge/rate.py` pairs with `judge/calibration.py`.

Design notes that matter for trust:
- The second-frontier `cc_*` verdict is HIDDEN by default. The human must verify INDEPENDENTLY;
  seeing the cross-check first anchors them and defeats the point of the gate. `--show-crosscheck`
  reveals it for post-hoc review only.
- The CSV IS the session state: every edit rewrites the whole file atomically, so resume and
  crash-safety are free. Samples are small by design (a few dozen across strata).
- The DECISION (accept/reject) is what advances; marking the individual checks does not, because
  an item has several checks and you set them before deciding.

The loop is driven by an injected input iterator + output sink, so it is fully unit-testable
without a terminal, model, endpoint, or GPU (it operates only on the CSV).
"""

import csv
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from llb.goldset.verify import (
    ACCEPT,
    CHECK_COLS,
    FAIL,
    HUMAN_COLS,
    PASS,
    REJECT,
    STATUS_DECIDED,
    STATUS_PENDING,
    load_worksheet,
    write_worksheet_rows,
)

# The four checks, in card order, mapped to the keystroke that marks them. Lowercase = PASS,
# uppercase = FAIL. `planted` only applies to synthetic items (blank/N/A for real ones).
CHECK_KEYS: dict[str, str] = {
    "g": "chk_grounded",
    "a": "chk_answerable",
    "r": "chk_reference",
    "p": "chk_planted",
}
CHECK_LABEL: dict[str, str] = {
    "chk_grounded": "span grounded (offsets really support the answer/label)",
    "chk_answerable": "answerable + non-circular (question doesn't leak its answer)",
    "chk_reference": "reference answer correct",
    "chk_planted": "planted labels match the doc (synthetic only)",
}

# Command kinds returned by `parse_command`.
CHECK = "check"
ACCEPT_CMD = "accept"
REJECT_CMD = "reject"
NOTE = "note"
NEXT = "next"
PREV = "prev"
JUMP = "jump"
UNDECIDED = "undecided"
CLEAR = "clear"
HELP = "help"
QUIT = "quit"
UNKNOWN = "unknown"

_ESC = "\x1b"
_ARROWS = {f"{_ESC}[A": PREV, f"{_ESC}[D": PREV, f"{_ESC}[B": NEXT, f"{_ESC}[C": NEXT}
_SIMPLE_COMMANDS = {
    "": NEXT,
    "n": NEXT,
    "b": PREV,
    "u": UNDECIDED,
    "c": CLEAR,
    "y": ACCEPT_CMD,
    "x": REJECT_CMD,
    "q": QUIT,
    "quit": QUIT,
    "?": HELP,
    "h": HELP,
    "help": HELP,
    "note": NOTE,
}

PROMPT_HINT = "g/a/r/p=check PASS  G/A/R/P=FAIL  y=accept  x=reject  n=next  b=prev  j<N>=jump  ?=help  q=quit"


@dataclass(frozen=True)
class Command:
    """A parsed prompt line. For CHECK, `field` is the column and `value` is pass/fail; for JUMP,
    `value` is the target; `raw` carries offending text for UNKNOWN so the loop can echo it."""

    kind: str
    field: str = ""
    value: bool | int | None = None
    raw: str = ""


def _parse_simple_command(s: str) -> Command | None:
    kind = _SIMPLE_COMMANDS.get(s.lower())
    return Command(kind) if kind is not None else None


def _parse_check_command(s: str) -> Command | None:
    if s in CHECK_KEYS:
        return Command(CHECK, field=CHECK_KEYS[s], value=True)
    if s.lower() in CHECK_KEYS and s.isupper():
        return Command(CHECK, field=CHECK_KEYS[s.lower()], value=False)
    return None


def _parse_jump_command(s: str) -> Command | None:
    if not s.lower().startswith("j"):
        return None
    rest = s[1:].strip()
    if rest.isdigit():
        return Command(JUMP, value=int(rest))
    return Command(UNKNOWN, raw=s)


def parse_command(raw: str) -> Command:
    """Parse one prompt line into a `Command` (pure; no I/O, no state).

    Empty / `n` = next; `b`/up/left arrow = previous; `g/a/r/p` mark a check PASS, the uppercase
    forms FAIL; `y` = accept, `x` = reject; `note` = edit a note; `j <N>`/`jN` = jump; `u` = next
    undecided; `c` = clear this item; `?`/`h` = help; `q` = save + quit.
    """
    s = raw.strip()
    if s in _ARROWS:
        return Command(_ARROWS[s])
    for parser in (_parse_simple_command, _parse_check_command, _parse_jump_command):
        cmd = parser(s)
        if cmd is not None:
            return cmd
    return Command(UNKNOWN, raw=s)


def first_undecided_index(rows: Sequence[dict[str, str]]) -> int:
    """Index of the first row without an accept/reject decision (resume point). 0 if all decided."""
    for i, row in enumerate(rows):
        if (row.get("decision") or "").strip() not in (ACCEPT, REJECT):
            return i
    return 0


def decided_count(rows: Sequence[dict[str, str]]) -> int:
    """How many rows carry an accept/reject decision."""
    return sum(1 for row in rows if (row.get("decision") or "").strip() in (ACCEPT, REJECT))


def decision_tally(rows: Sequence[dict[str, str]]) -> tuple[int, int]:
    """`(accepted, rejected)` counts over decided rows."""
    accepted = sum(1 for row in rows if (row.get("decision") or "").strip() == ACCEPT)
    rejected = sum(1 for row in rows if (row.get("decision") or "").strip() == REJECT)
    return accepted, rejected


def summary_lines(rows: Sequence[dict[str, str]], path: Path) -> list[str]:
    """The end-of-session report: progress, accept/reject split, and the next command."""
    total = len(rows)
    decided = decided_count(rows)
    accepted, rejected = decision_tally(rows)
    lines = [
        f"[verify] saved {path}",
        f"[verify] progress : {decided}/{total} decided, {total - decided} remaining "
        f"(accept {accepted}, reject {rejected})",
    ]
    if decided < total:
        lines.append(
            "[verify] resume   : re-run `make verify-review` (continues at the first undecided item)"
        )
    lines.append(f"[verify] accept   : make verify-accept WS={path} BUNDLE=<draft bundle>")
    return lines


def _advanced_index(idx: int, total: int, rows: Sequence[dict[str, str]]) -> int:
    """Where to go after deciding the item at `idx`: next item, else wrap to the first undecided,
    else the completion screen once everything is decided (never stuck re-showing the last card)."""
    if idx < total - 1:
        return idx + 1
    if decided_count(rows) == total:
        return total
    return first_undecided_index(rows)


def completion_panel(rows: Sequence[dict[str, str]], total: int) -> str:
    """The 'all items decided' review screen shown once you advance past the last item."""
    accepted, rejected = decision_tally(rows)
    return "\n".join(
        [
            f"===== all {total} items decided (accept {accepted}, reject {rejected}) =====",
            "  review/change: b = last item, j <N> = jump to item N, u = next undecided",
            "  finish: press Enter or q to save + quit (then run make verify-accept)",
        ]
    )


def clear_human_columns(rows: Sequence[dict[str, str]]) -> None:
    """Wipe every human column in place (the `--clear` start-fresh path)."""
    for row in rows:
        for col in HUMAN_COLS:
            row[col] = ""


def save_human_columns(
    path: Path, rows: Sequence[dict[str, str]], fieldnames: Sequence[str]
) -> None:
    """Persist ONLY the human columns, merged into the CURRENT on-disk worksheet by `item_id`.

    Re-reading the file on each save and overlaying only the human columns means a context column
    the sampler owns is never clobbered by the session's load-time snapshot. Falls back to a full
    write if the file is missing or unreadable.
    """
    try:
        disk_rows, disk_fields = load_worksheet(path)
    except (OSError, csv.Error):
        write_worksheet_rows(path, rows, fieldnames)
        return
    if not disk_rows:
        write_worksheet_rows(path, rows, fieldnames)
        return
    human_by_id = {
        row["item_id"]: {col: row.get(col, "") for col in HUMAN_COLS}
        for row in rows
        if row.get("item_id")
    }
    for disk_row in disk_rows:
        overlay = human_by_id.get(disk_row.get("item_id", ""))
        if overlay is not None:
            disk_row.update(overlay)
    write_worksheet_rows(path, disk_rows, disk_fields)


def _field(row: dict[str, str], name: str, blank: str) -> str:
    value = (row.get(name) or "").strip()
    return value if value else blank


def _is_synthetic_row(row: dict[str, str]) -> bool:
    return (row.get("synthetic") or "").strip().lower() == "true"


def format_card(
    row: dict[str, str],
    position: int,
    total: int,
    decided: int,
    *,
    show_crosscheck: bool = False,
) -> str:
    """Render the per-item card: the question/reference, the cited span inside its corpus window,
    the four check states, and the decision. `cc_*` shown ONLY when `show_crosscheck` is set."""
    remaining = total - decided
    lines = [
        f"item {position}/{total} (decided {decided}, remaining {remaining})",
        f"  item_id    : {row.get('item_id', '')}",
        f"  stratum    : {row.get('stratum', '')}",
        f"  question   : {row.get('question', '')}",
        f"  reference  : {row.get('reference_answer', '')}",
        f"  span doc   : {row.get('span_doc_id', '')}",
        f"  context    : {row.get('context', '')}",
        "  checks:",
    ]
    synthetic = _is_synthetic_row(row)
    for col in CHECK_COLS:
        if col == "chk_planted" and not synthetic:
            continue
        lines.append(f"    {col:<13}: {_field(row, col, '(unchecked)')}  -- {CHECK_LABEL[col]}")
    lines.append(f"  decision   : {_field(row, 'decision', '(undecided)')}")
    note = (row.get("human_note") or "").strip()
    if note:
        lines.append(f"  note       : {note}")
    if show_crosscheck:
        cc = "  ".join(
            f"{key}={_field(row, f'cc_{key}', '?')}"
            for key in ("grounded", "non_circular", "supported", "answerable")
        )
        lines.append(f"  crosscheck : {cc}  note={_field(row, 'cc_note', '')}")
    return "\n".join(lines)


def help_text() -> str:
    """The command + check reference shown by `?`."""
    checks = "\n".join(
        f"  {key} / {key.upper()}  {CHECK_LABEL[CHECK_KEYS[key]]}" for key in CHECK_KEYS
    )
    return "\n".join(
        [
            "checks (lowercase = PASS, uppercase = FAIL):",
            checks,
            "decision:",
            "  y  accept (within tolerance)            x  reject (back to the pipeline)",
            "navigation:",
            "  n/Enter  next                           b/up/left  previous",
            "  j <N>    jump to item N                 u  next undecided",
            "  note     edit a note                    c  clear this item's marks",
            "  ?/h      this help                      q  save + quit",
            "verify INDEPENDENTLY against the corpus window -- do not anchor to the cross-check.",
        ]
    )


def _default_output(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _stdin_reader(prompt: str) -> str:
    return input(prompt)


class _Quit(Exception):
    """Internal: end the session (q, EOF, or exhausted injected input)."""


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


def _handle_decision_action(
    cmd: Command,
    row: dict[str, str],
    idx: int,
    total: int,
    rows: Sequence[dict[str, str]],
    emit: Callable[[str], None],
    path: Path,
    fieldnames: Sequence[str],
) -> tuple[int, bool]:
    if cmd.kind == ACCEPT_CMD:
        decision = ACCEPT
    elif cmd.kind == REJECT_CMD:
        decision = REJECT
    else:
        return idx, False

    _set_decision(row, decision)
    _save(path, rows, fieldnames)
    return _go_forward(idx, total, rows, emit), True


def _handle_edit_action(
    cmd: Command,
    row: dict[str, str],
    rows: Sequence[dict[str, str]],
    emit: Callable[[str], None],
    it: Iterator[str] | None,
    path: Path,
    fieldnames: Sequence[str],
) -> bool:
    if cmd.kind == CHECK:
        if _set_check(row, cmd, emit):
            _save(path, rows, fieldnames)
    elif cmd.kind == CLEAR:
        _clear_row(row)
        _save(path, rows, fieldnames)
    elif cmd.kind == NOTE:
        text = _read("note (empty to clear): ", it, emit).strip()
        row["human_note"] = text
        _save(path, rows, fieldnames)
    else:
        return False
    return True


def _emit_unknown_command(cmd: Command, emit: Callable[[str], None]) -> None:
    if cmd.raw.startswith(_ESC):
        emit("[verify] arrow keys garbled -- use n (next) / b (prev).")
    else:
        emit(f"[verify] not a command: {cmd.raw!r} (? for help).")


def _handle_row_action(
    rows: Sequence[dict[str, str]],
    idx: int,
    total: int,
    emit: Callable[[str], None],
    it: Iterator[str] | None,
    path: Path,
    fieldnames: Sequence[str],
    show_crosscheck: bool = False,
) -> int:
    row = rows[idx]
    emit(format_card(row, idx + 1, total, decided_count(rows), show_crosscheck=show_crosscheck))
    cmd = parse_command(_read(f"{PROMPT_HINT}\nverify> ", it, emit))

    if cmd.kind == QUIT:
        raise _Quit
    idx, handled = _handle_navigation_action(cmd, idx, total, rows, emit)
    if handled:
        return idx

    idx, handled = _handle_decision_action(cmd, row, idx, total, rows, emit, path, fieldnames)
    if handled:
        return idx

    if _handle_edit_action(cmd, row, rows, emit, it, path, fieldnames):
        return idx

    _emit_unknown_command(cmd, emit)
    return idx


def run_session(
    worksheet_path: Path | str,
    *,
    inputs: Iterable[str] | None = None,
    output: Callable[[str], None] | None = None,
    start: int | None = None,
    show_crosscheck: bool = False,
    clear: bool = False,
) -> int:
    """Drive the interactive verification over `worksheet_path`; return the decided count.

    `inputs` / `output` are injected for testing; in a real terminal they default to `input()` /
    stdout. Every edit writes the whole CSV atomically, so a crash, EOF, or Ctrl-C never loses
    work. With no `start`, resume at the first undecided item. `clear` wipes all human columns
    first (confirmation-gated).
    """
    path = Path(worksheet_path)
    emit = output or _default_output
    it: Iterator[str] | None = iter(inputs) if inputs is not None else None

    rows, fieldnames = load_worksheet(path)
    if not rows:
        emit(f"[verify] worksheet has no rows: {path}")
        return 0

    if not _maybe_clear_human_columns(clear, rows, path, fieldnames, it, emit):
        return decided_count(rows)

    total = len(rows)
    idx = _get_idx(start, total, rows)

    _emit_intro(emit)

    try:
        while True:
            idx, is_completion = _handle_completion_screen(idx, total, rows, emit, it)
            if is_completion:
                continue

            idx = _handle_row_action(rows, idx, total, emit, it, path, fieldnames, show_crosscheck)
    except (_Quit, EOFError):
        pass
    except KeyboardInterrupt:
        emit("")

    _save(path, rows, fieldnames)
    for line in summary_lines(rows, path):
        emit(line)
    return decided_count(rows)
