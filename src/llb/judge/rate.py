"""Interactive human rater for the judge-calibration worksheet (judge calibration gate).

A terminal session that walks a pre-filled calibration worksheet item by item and writes
the HUMAN columns (`human_answer`, `human_rating`, `human_note`, `human_status`) in place.
Interactive I/O lives here, OUT of the pure-stats `calibration.py`; the two share the
worksheet schema + atomic load/save (`load_worksheet` / `write_worksheet_rows`).

Design notes that matter for trust:
- The judge's `judge_rating` is HIDDEN by default. The manual requires rating INDEPENDENTLY;
  seeing the judge first anchors the human and contaminates the ground truth. `--show-judge`
  reveals it for post-hoc review only.
- The CSV IS the session state: every edit rewrites the whole file atomically, so resume and
  crash-safety are free (no separate journal). Calibration sets are small by design.
- Spearman (in `calibration.py`) is rank-based, so the 1-5 human scale and the judge's [0,1]
  scale are compatible -- only the ORDER the answers are put in matters.

The session loop is driven by an injected input iterator + output sink, so it is fully
unit-testable without a terminal, model, endpoint, or GPU (it operates only on the CSV).
"""

import csv
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from llb.judge.calibration import HUMAN_COLS, load_worksheet, write_worksheet_rows

# Integer Likert scale for the human rating. Named so the anchors, the help text, and the
# parser agree on one range; Spearman is rank-based so 1-5 (human) vs [0,1] (judge) is fine.
RATING_MIN = 1
RATING_MAX = 5
RATING_ANCHORS: dict[int, str] = {
    1: "wrong / unfaithful",
    2: "mostly wrong",
    3: "partially correct",
    4: "mostly correct",
    5: "fully correct + faithful",
}

# human_status values (a deliberate skip is distinguishable from not-yet-reached; resume
# still keys on an empty human_rating, so this is a refinement, not a requirement).
STATUS_PENDING = "pending"
STATUS_RATED = "rated"

# Command kinds returned by `parse_command` (the loop maps these to actions).
RATE = "rate"
ANSWER = "answer"
NOTE = "note"
NEXT = "next"
PREV = "prev"
JUMP = "jump"
UNRATED = "unrated"
CLEAR = "clear"
HELP = "help"
QUIT = "quit"
UNKNOWN = "unknown"

# Arrow-key escape sequences. A single arrow submitted as a line maps to navigation as a
# convenience; full char-by-char arrow handling would need raw terminal mode, which the
# line-based (and unit-testable) loop deliberately avoids. ESC is shown as `^[` by terminals.
_ESC = "\x1b"
_ARROWS = {f"{_ESC}[A": PREV, f"{_ESC}[D": PREV, f"{_ESC}[B": NEXT, f"{_ESC}[C": NEXT}

# Compact, always-visible action + scale legend printed at the prompt (full anchors via `?`).
PROMPT_HINT = "[1-5]=rate (1=wrong..5=correct)  a=answer  n=next  p=prev  j<N>=jump  ?=help  q=quit"


@dataclass(frozen=True)
class Command:
    """A parsed prompt command. `value` carries the rating int (RATE) or jump target (JUMP);
    `raw` carries the offending text for UNKNOWN so the loop can echo a helpful error."""

    kind: str
    value: int | None = None
    raw: str = ""


_SIMPLE_COMMANDS: dict[str, str] = {
    "n": NEXT,
    "p": PREV,
    "b": PREV,
    "u": UNRATED,
    "c": CLEAR,
    "q": QUIT,
    "quit": QUIT,
    "?": HELP,
    "h": HELP,
    "help": HELP,
    "a": ANSWER,
    "answer": ANSWER,
    "note": NOTE,
}


def _simple_command(text: str) -> Command | None:
    kind = _SIMPLE_COMMANDS.get(text.lower())
    return Command(kind) if kind is not None else None


def _jump_command(text: str) -> Command | None:
    if not text.lower().startswith("j"):
        return None
    target = text[1:].strip()
    if target.isdigit():
        return Command(JUMP, value=int(target))
    return Command(UNKNOWN, raw=text)


def _rating_command(text: str) -> Command | None:
    if not text.isdigit():
        return None
    value = int(text)
    if RATING_MIN <= value <= RATING_MAX:
        return Command(RATE, value=value)
    return Command(UNKNOWN, raw=text)


def parse_command(raw: str) -> Command:
    """Parse one prompt line into a `Command` (pure; no I/O, no state).

    Empty input / `n` = next; `p`/`b` = previous; a bare integer in range = set that rating;
    `a` = author the answer; `note` = edit the note; `j <N>` / `jN` = jump to item N;
    `u` = next unrated; `c` = clear the rating; `?`/`h` = help; `q` = save + quit.
    """
    s = raw.strip()
    if s == "":
        return Command(NEXT)
    if s in _ARROWS:  # a lone arrow key -> navigate (up/left = prev, down/right = next)
        return Command(_ARROWS[s])
    for parser in (_simple_command, _jump_command, _rating_command):
        command = parser(s)
        if command is not None:
            return command
    return Command(UNKNOWN, raw=s)


def first_unrated_index(rows: Sequence[dict[str, str]]) -> int:
    """Index of the first row with an empty `human_rating` (resume point). 0 if all rated."""
    for i, row in enumerate(rows):
        if not (row.get("human_rating") or "").strip():
            return i
    return 0


def rated_count(rows: Sequence[dict[str, str]]) -> int:
    """How many rows carry a non-empty `human_rating`."""
    return sum(1 for row in rows if (row.get("human_rating") or "").strip())


def rating_histogram(rows: Sequence[dict[str, str]]) -> dict[int, int]:
    """Count of rows at each rating `RATING_MIN..RATING_MAX` (ignores blank/out-of-range)."""
    hist = {n: 0 for n in range(RATING_MIN, RATING_MAX + 1)}
    for row in rows:
        value = (row.get("human_rating") or "").strip()
        if value.isdigit() and int(value) in hist:
            hist[int(value)] += 1
    return hist


def summary_lines(rows: Sequence[dict[str, str]], path: Path) -> list[str]:
    """The end-of-session report: progress, the rating spread, and the next command.

    The spread matters for calibration -- a good worksheet spans the full 1-5 range and
    deliberately includes fluent-but-wrong answers, so it is surfaced explicitly here.
    """
    total = len(rows)
    rated = rated_count(rows)
    answered = sum(1 for row in rows if (row.get("human_answer") or "").strip())
    hist = rating_histogram(rows)
    spread = "  ".join(f"{n}:{hist[n]}" for n in range(RATING_MIN, RATING_MAX + 1))
    lines = [
        f"[calibration] saved {path}",
        f"[calibration] progress : {rated}/{total} rated, {total - rated} remaining, "
        f"{answered} with your own answer",
        f"[calibration] ratings  : {spread}  (1=wrong .. 5=fully correct)",
    ]
    if rated < total:
        lines.append(
            "[calibration] resume   : re-run `make calibration-rate` "
            "(continues at the first unrated item)"
        )
    lines.append(f"[calibration] score    : make calibration-score RATINGS={path}")
    return lines


def _advanced_index(idx: int, total: int, rows: Sequence[dict[str, str]]) -> int:
    """Where to go after rating/skipping the item at `idx`.

    Next item normally; at the last item it lands on the completion screen (`total`) once
    everything is rated, else it wraps back to the first remaining unrated item so gaps left
    by out-of-order rating still get filled (never stuck re-showing the last card).
    """
    if idx < total - 1:
        return idx + 1
    if rated_count(rows) == total:
        return total
    return first_unrated_index(rows)


def completion_panel(rows: Sequence[dict[str, str]], total: int) -> str:
    """The 'all items rated' review screen shown once you advance past the last item."""
    answered = sum(1 for row in rows if (row.get("human_answer") or "").strip())
    hist = rating_histogram(rows)
    spread = "  ".join(f"{n}:{hist[n]}" for n in range(RATING_MIN, RATING_MAX + 1))
    return "\n".join(
        [
            f"===== all {total} items rated ({answered} with your own answer) =====",
            f"  rating spread: {spread}   (1=wrong .. 5=fully correct)",
            "  review/change: p = last item, j <N> = jump to item N, u = next unrated",
            "  finish: press Enter or q to save + quit (then run make calibration-score)",
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

    The rater owns the human columns; every other column (`model_answer`, `judge_rating`,
    `provenance`, ...) belongs to `calibration-run`. Re-reading the file on each save and
    overlaying only the human columns means an intervening `calibration-run` that filled
    `judge_rating` is never clobbered by the rater's load-time snapshot -- the failure mode
    that silently dropped a whole judge column. Falls back to a full write if the file is
    missing or unreadable.
    """
    try:
        disk_rows, disk_fields = load_worksheet(path)
    except (OSError, csv.Error):
        disk_rows = []
        disk_fields = list(fieldnames)
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


def format_card(
    row: dict[str, str],
    position: int,
    total: int,
    rated: int,
    *,
    show_judge: bool = False,
) -> str:
    """Render the per-item card (ASCII labels; the rated content is the item's own text).

    Shows progress, ids/provenance, question, reference, the candidate's `model_answer`, and
    the human's current answer/rating. `judge_rating` is shown ONLY when `show_judge` is set.
    """
    remaining = total - rated
    lines = [
        f"item {position}/{total} (rated {rated}, remaining {remaining})",
        f"  item_id    : {row.get('item_id', '')}",
        f"  provenance : {_field(row, 'provenance', '(unknown)')}",
        f"  question   : {row.get('question', '')}",
        f"  reference  : {row.get('reference_answer', '')}",
        f"  model      : {_field(row, 'model_answer', '(no answer)')}",
        f"  your answer: {_field(row, 'human_answer', '(none)')}",
        f"  your rating: {_field(row, 'human_rating', '(unrated)')}",
    ]
    note = (row.get("human_note") or "").strip()
    if note:
        lines.append(f"  your note  : {note}")
    if show_judge:
        lines.append(f"  judge      : {_field(row, 'judge_rating', '(none)')}")
    return "\n".join(lines)


def help_text() -> str:
    """The command + rating-anchor reference shown by `?`."""
    anchors = "; ".join(f"{n} = {RATING_ANCHORS[n]}" for n in range(RATING_MIN, RATING_MAX + 1))
    return "\n".join(
        [
            "commands:",
            f"  {RATING_MIN}-{RATING_MAX}  set rating + advance     a     author/edit your answer",
            "  n/Enter  next (no change)         note  edit a note",
            "  p/b      previous                 c     clear this rating",
            "  j <N>    jump to item N           u     jump to next unrated",
            "  ?/h      this help                q     save + quit",
            "  arrow keys: up/left = previous, down/right = next",
            f"rating anchors: {anchors}",
            "rate INDEPENDENTLY -- author your own answer first; do not anchor to the judge.",
        ]
    )


def _default_output(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _stdin_reader(prompt: str) -> str:
    return input(prompt)


class _Quit(Exception):
    """Internal: end the session (q, EOF, or exhausted injected input)."""


def _emit_help(emit: Callable[[str], None]) -> None:
    emit(
        "judge calibration -- rate each model answer against the reference, "
        f"{RATING_MIN} (wrong) to {RATING_MAX} (fully correct)."
    )
    emit(help_text())


def _get_idx(start: int | None, total: int, rows: Sequence[dict[str, str]]) -> int:
    if start is not None:
        idx = max(0, min(start - 1, total - 1))
    elif rated_count(rows) == total:
        idx = total  # everything already rated -> open on the review/finish screen
    else:
        idx = first_unrated_index(rows)
    return idx


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


def _set_rating(row: dict[str, str], value: int) -> None:
    row["human_rating"] = str(value)
    row["human_status"] = STATUS_RATED


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
        "clear ALL human ratings/answers and start fresh? type 'yes' to confirm: ", it, emit
    )
    if ans.strip().lower() != "yes":
        emit("[calibration] clear aborted; nothing changed.")
        return False
    clear_human_columns(rows)
    _save(path, rows, fieldnames)
    emit("[calibration] cleared all human columns.")
    return True


def _clear_rating(row: dict[str, str]) -> None:
    row["human_rating"] = ""
    row["human_status"] = STATUS_PENDING


def _go_forward(
    idx: int, total: int, rows: Sequence[dict[str, str]], emit: Callable[[str], None]
) -> int:
    new = _advanced_index(idx, total, rows)
    if idx == total - 1 and new < total:  # wrapped back to fill an unrated gap
        remaining = total - rated_count(rows)
        emit(f"[calibration] {remaining} item(s) still unrated -- jumping there.")
    return new


def _jump_to(idx: int, target: int, total: int, emit: Callable[[str], None]) -> int:
    if 1 <= target <= total:
        idx = target - 1
    else:
        emit(f"[calibration] item out of range 1..{total}: {target}")
    return idx


def _go_unrated(idx: int, rows: Sequence[dict[str, str]], emit: Callable[[str], None]) -> int:
    next_idx = first_unrated_index(rows)
    if (rows[next_idx].get("human_rating") or "").strip():
        emit("[calibration] all items are rated.")
    else:
        idx = next_idx
    return idx


def _handle_idx_eq_total(
    idx: int,
    total: int,
    rows: Sequence[dict[str, str]],
    emit: Callable[[str], None],
    it: Iterator[str] | None,
) -> tuple[int, bool]:
    is_last_item = idx >= total
    if is_last_item:  # past the last item -> the review / finish screen
        emit(completion_panel(rows, total))
        cmd = parse_command(_read("review (p / j <N> / u) or finish (Enter / q) > ", it, emit))
        if cmd.kind in (QUIT, NEXT):
            raise _Quit
        if cmd.kind == HELP:
            emit(help_text())
        elif cmd.kind == PREV:
            idx = total - 1
        elif cmd.kind == JUMP:
            idx = _jump_to(idx, cmd.value if cmd.value is not None else 0, total, emit)
        elif cmd.kind == UNRATED:
            idx = _go_unrated(idx, rows, emit)
        else:
            emit("[calibration] all rated -- p to review, j <N> to jump, Enter/q to finish.")
    return idx, is_last_item


def _handle_row_action(
    rows: Sequence[dict[str, str]],
    idx: int,
    total: int,
    emit: Callable[[str], None],
    it: Iterator[str] | None,
    path: Path,
    fieldnames: Sequence[str],
    show_judge: bool = False,
) -> int:
    row = rows[idx]
    emit(format_card(row, idx + 1, total, rated_count(rows), show_judge=show_judge))
    cmd = parse_command(_read(f"{PROMPT_HINT}\nrating> ", it, emit))
    kind = cmd.kind

    if kind == QUIT:
        raise _Quit
    elif kind == HELP:
        emit(help_text())
    elif kind == NEXT:
        idx = _go_forward(idx, total, rows, emit)
    elif kind == PREV:
        idx = max(idx - 1, 0)
    elif kind == JUMP:
        idx = _jump_to(idx, cmd.value if cmd.value is not None else 0, total, emit)
    elif kind == UNRATED:
        idx = _go_unrated(idx, rows, emit)
    elif kind == RATE:
        _set_rating(row, cmd.value or RATING_MIN)
        _save(path, rows, fieldnames)
        idx = _go_forward(idx, total, rows, emit)
    elif kind == CLEAR:
        _clear_rating(row)
        _save(path, rows, fieldnames)
    elif kind == ANSWER:
        text = _read("your answer (empty to clear): ", it, emit).strip()
        row["human_answer"] = text
        _save(path, rows, fieldnames)
    elif kind == NOTE:
        text = _read("note (empty to clear): ", it, emit).strip()
        row["human_note"] = text
        _save(path, rows, fieldnames)
    elif cmd.raw.startswith(_ESC):
        emit("[calibration] arrow keys garbled -- use n (next) / p (prev).")
    else:
        emit(f"[calibration] not a command: {cmd.raw!r} (? for help; 1-5 to rate).")
    return idx


def run_session(
    worksheet_path: Path | str,
    *,
    inputs: Iterable[str] | None = None,
    output: Callable[[str], None] | None = None,
    start: int | None = None,
    show_judge: bool = False,
    clear: bool = False,
) -> int:
    """Drive the interactive rating session over `worksheet_path`; return the rated count.

    `inputs` (an iterable of lines) and `output` (a line sink) are injected for testing; in a
    real terminal they default to `input()` / stdout. Every edit writes the whole CSV through
    atomically, so a crash, EOF, or Ctrl-C never loses work. With no `start`, resume at the
    first unrated item. `clear` wipes all human columns first (confirmation-gated).
    """
    path = Path(worksheet_path)
    emit = output or _default_output
    it: Iterator[str] | None = iter(inputs) if inputs is not None else None

    rows, fieldnames = load_worksheet(path)
    if not rows:
        emit(f"[calibration] worksheet has no rows: {path}")
        return 0

    if not _maybe_clear_human_columns(clear, rows, path, fieldnames, it, emit):
        return rated_count(rows)

    total = len(rows)
    idx = _get_idx(start, total, rows)

    _emit_help(emit)

    try:
        while True:
            idx, is_last_item = _handle_idx_eq_total(idx, total, rows, emit, it)
            if is_last_item:
                continue

            idx = _handle_row_action(rows, idx, total, emit, it, path, fieldnames, show_judge)

    except (_Quit, EOFError):
        pass
    except KeyboardInterrupt:
        emit("")  # break the prompt line cleanly

    _save(path, rows, fieldnames)
    for line in summary_lines(rows, path):
        emit(line)
    return rated_count(rows)
