"""Interactive human verifier for the MH.5 sample worksheet.

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

PROMPT_HINT = "g/a/r/p=check PASS  G/A/R/P=FAIL  y=accept  x=reject  n=next  b=prev  j<N>=jump  ?=help  q=quit"


@dataclass(frozen=True)
class Command:
    """A parsed prompt line. For CHECK, `field` is the column and `value` is pass/fail; for JUMP,
    `value` is the target; `raw` carries offending text for UNKNOWN so the loop can echo it."""

    kind: str
    field: str = ""
    value: bool | int | None = None
    raw: str = ""


def parse_command(raw: str) -> Command:
    """Parse one prompt line into a `Command` (pure; no I/O, no state).

    Empty / `n` = next; `b`/up/left arrow = previous; `g/a/r/p` mark a check PASS, the uppercase
    forms FAIL; `y` = accept, `x` = reject; `note` = edit a note; `j <N>`/`jN` = jump; `u` = next
    undecided; `c` = clear this item; `?`/`h` = help; `q` = save + quit.
    """
    s = raw.strip()
    if s == "":
        return Command(NEXT)
    if s in _ARROWS:
        return Command(_ARROWS[s])
    low = s.lower()
    if low == "n":
        return Command(NEXT)
    if low == "b":
        return Command(PREV)
    if low == "u":
        return Command(UNDECIDED)
    if low == "c":
        return Command(CLEAR)
    if low == "y":
        return Command(ACCEPT_CMD)
    if low == "x":
        return Command(REJECT_CMD)
    if low in ("q", "quit"):
        return Command(QUIT)
    if low in ("?", "h", "help"):
        return Command(HELP)
    if low == "note":
        return Command(NOTE)
    if s in CHECK_KEYS:  # lowercase letter -> mark PASS
        return Command(CHECK, field=CHECK_KEYS[s], value=True)
    if s.lower() in CHECK_KEYS and s.isupper():  # uppercase letter -> mark FAIL
        return Command(CHECK, field=CHECK_KEYS[s.lower()], value=False)
    if low[0] == "j":  # jump: "j 5" or "j5"
        rest = s[1:].strip()
        if rest.isdigit():
            return Command(JUMP, value=int(rest))
        return Command(UNKNOWN, raw=s)
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

    def read(prompt: str) -> str:
        if it is None:
            return _stdin_reader(prompt)
        emit(prompt)
        try:
            return next(it)
        except StopIteration as exc:
            raise _Quit from exc

    rows, fieldnames = load_worksheet(path)
    if not rows:
        emit(f"[verify] worksheet has no rows: {path}")
        return 0

    def save() -> None:
        save_human_columns(path, rows, fieldnames)

    if clear:
        ans = read("clear ALL human marks/decisions and start fresh? type 'yes' to confirm: ")
        if ans.strip().lower() != "yes":
            emit("[verify] clear aborted; nothing changed.")
            return decided_count(rows)
        clear_human_columns(rows)
        save()
        emit("[verify] cleared all human columns.")

    total = len(rows)
    if start is not None:
        idx = max(0, min(start - 1, total - 1))
    elif decided_count(rows) == total:
        idx = total
    else:
        idx = first_undecided_index(rows)

    emit(
        "MH.5 data verification -- verify each sampled item against the corpus, then accept/reject."
    )
    emit(help_text())

    def set_decision(row: dict[str, str], decision: str) -> None:
        row["decision"] = decision
        row["human_status"] = STATUS_DECIDED

    def go_forward() -> None:
        nonlocal idx
        new = _advanced_index(idx, total, rows)
        if idx == total - 1 and new < total:
            remaining = total - decided_count(rows)
            emit(f"[verify] {remaining} item(s) still undecided -- jumping there.")
        idx = new

    def jump_to(target: int) -> None:
        nonlocal idx
        if 1 <= target <= total:
            idx = target - 1
        else:
            emit(f"[verify] item out of range 1..{total}: {target}")

    def go_undecided() -> None:
        nonlocal idx
        nxt = first_undecided_index(rows)
        if (rows[nxt].get("decision") or "").strip() in (ACCEPT, REJECT):
            emit("[verify] all items are decided.")
        else:
            idx = nxt

    try:
        while True:
            if idx >= total:  # past the last item -> the review / finish screen
                emit(completion_panel(rows, total))
                cmd = parse_command(read("review (b / j <N> / u) or finish (Enter / q) > "))
                if cmd.kind in (QUIT, NEXT):
                    raise _Quit
                if cmd.kind == HELP:
                    emit(help_text())
                elif cmd.kind == PREV:
                    idx = total - 1
                elif cmd.kind == JUMP:
                    jump_to(cmd.value if isinstance(cmd.value, int) else 0)
                elif cmd.kind == UNDECIDED:
                    go_undecided()
                else:
                    emit("[verify] all decided -- b to review, j <N> to jump, Enter/q to finish.")
                continue

            row = rows[idx]
            emit(
                format_card(
                    row, idx + 1, total, decided_count(rows), show_crosscheck=show_crosscheck
                )
            )
            cmd = parse_command(read(f"{PROMPT_HINT}\nverify> "))

            if cmd.kind == QUIT:
                raise _Quit
            if cmd.kind == HELP:
                emit(help_text())
                continue
            if cmd.kind == NEXT:
                go_forward()
                continue
            if cmd.kind == PREV:
                idx = max(idx - 1, 0)
                continue
            if cmd.kind == JUMP:
                jump_to(cmd.value if isinstance(cmd.value, int) else 0)
                continue
            if cmd.kind == UNDECIDED:
                go_undecided()
                continue
            if cmd.kind == CHECK:
                if cmd.field == "chk_planted" and not _is_synthetic_row(row):
                    emit("[verify] planted check is N/A for a real (non-synthetic) item.")
                    continue
                row[cmd.field] = PASS if cmd.value else FAIL
                save()
                continue
            if cmd.kind == ACCEPT_CMD:
                set_decision(row, ACCEPT)
                save()
                go_forward()
                continue
            if cmd.kind == REJECT_CMD:
                set_decision(row, REJECT)
                save()
                go_forward()
                continue
            if cmd.kind == CLEAR:
                for col in HUMAN_COLS:
                    row[col] = ""
                row["human_status"] = STATUS_PENDING
                save()
                continue
            if cmd.kind == NOTE:
                text = read("note (empty to clear): ").strip()
                row["human_note"] = text
                save()
                continue
            if cmd.raw.startswith(_ESC):
                emit("[verify] arrow keys garbled -- use n (next) / b (prev).")
            else:
                emit(f"[verify] not a command: {cmd.raw!r} (? for help).")
    except (_Quit, EOFError):
        pass
    except KeyboardInterrupt:
        emit("")

    save()
    for line in summary_lines(rows, path):
        emit(line)
    return decided_count(rows)
