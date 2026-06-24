"""Interactive human rater for the judge-calibration worksheet (M3.8).

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


@dataclass(frozen=True)
class Command:
    """A parsed prompt command. `value` carries the rating int (RATE) or jump target (JUMP);
    `raw` carries the offending text for UNKNOWN so the loop can echo a helpful error."""

    kind: str
    value: int | None = None
    raw: str = ""


def parse_command(raw: str) -> Command:
    """Parse one prompt line into a `Command` (pure; no I/O, no state).

    Empty input / `n` = next; `p`/`b` = previous; a bare integer in range = set that rating;
    `a` = author the answer; `note` = edit the note; `j <N>` / `jN` = jump to item N;
    `u` = next unrated; `c` = clear the rating; `?`/`h` = help; `q` = save + quit.
    """
    s = raw.strip()
    if s == "":
        return Command(NEXT)
    low = s.lower()
    if low in ("n",):
        return Command(NEXT)
    if low in ("p", "b"):
        return Command(PREV)
    if low in ("u",):
        return Command(UNRATED)
    if low in ("c",):
        return Command(CLEAR)
    if low in ("q", "quit"):
        return Command(QUIT)
    if low in ("?", "h", "help"):
        return Command(HELP)
    if low in ("a", "answer"):
        return Command(ANSWER)
    if low in ("note",):
        return Command(NOTE)
    if low[0] == "j":  # jump: "j 5" or "j5"
        rest = s[1:].strip()
        if rest.isdigit():
            return Command(JUMP, value=int(rest))
        return Command(UNKNOWN, raw=s)
    if s.isdigit():
        value = int(s)
        if RATING_MIN <= value <= RATING_MAX:
            return Command(RATE, value=value)
        return Command(UNKNOWN, raw=s)
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


def clear_human_columns(rows: Sequence[dict[str, str]]) -> None:
    """Wipe every human column in place (the `--clear` start-fresh path)."""
    for row in rows:
        for col in HUMAN_COLS:
            row[col] = ""


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
        emit(f"[calibration] worksheet has no rows: {path}")
        return 0

    def save() -> None:
        write_worksheet_rows(path, rows, fieldnames)

    if clear:
        ans = read("clear ALL human ratings/answers and start fresh? type 'yes' to confirm: ")
        if ans.strip().lower() != "yes":
            emit("[calibration] clear aborted; nothing changed.")
            return rated_count(rows)
        clear_human_columns(rows)
        save()
        emit("[calibration] cleared all human columns.")

    total = len(rows)
    if start is not None:
        idx = max(0, min(start - 1, total - 1))
    else:
        idx = first_unrated_index(rows)

    emit("judge calibration -- interactive rater. Type ? for help, q to save + quit.")

    def set_rating(row: dict[str, str], value: int) -> None:
        row["human_rating"] = str(value)
        row["human_status"] = STATUS_RATED

    def clear_rating(row: dict[str, str]) -> None:
        row["human_rating"] = ""
        row["human_status"] = STATUS_PENDING

    try:
        while True:
            row = rows[idx]
            emit(format_card(row, idx + 1, total, rated_count(rows), show_judge=show_judge))
            cmd = parse_command(read("> "))

            if cmd.kind == QUIT:
                raise _Quit
            if cmd.kind == HELP:
                emit(help_text())
                continue
            if cmd.kind == NEXT:
                idx = min(idx + 1, total - 1)
                continue
            if cmd.kind == PREV:
                idx = max(idx - 1, 0)
                continue
            if cmd.kind == JUMP:
                target = cmd.value if cmd.value is not None else 0
                if 1 <= target <= total:
                    idx = target - 1
                else:
                    emit(f"[calibration] item out of range 1..{total}: {target}")
                continue
            if cmd.kind == UNRATED:
                nxt = first_unrated_index(rows)
                if (rows[nxt].get("human_rating") or "").strip():
                    emit("[calibration] all items are rated.")
                else:
                    idx = nxt
                continue
            if cmd.kind == RATE:
                set_rating(row, cmd.value or RATING_MIN)
                save()
                idx = min(idx + 1, total - 1)
                continue
            if cmd.kind == CLEAR:
                clear_rating(row)
                save()
                continue
            if cmd.kind == ANSWER:
                text = read("your answer (empty to clear): ").strip()
                row["human_answer"] = text
                save()
                continue
            if cmd.kind == NOTE:
                text = read("note (empty to clear): ").strip()
                row["human_note"] = text
                save()
                continue
            emit(f"[calibration] unknown command: {cmd.raw!r} (? for help)")
    except (_Quit, EOFError):
        pass
    except KeyboardInterrupt:
        emit("")  # break the prompt line cleanly

    save()
    n_rated = rated_count(rows)
    emit(
        f"[calibration] saved {path} -- {n_rated}/{total} rated. "
        "Resume by re-running calibration-rate; score with: "
        f"make calibration-score RATINGS={path}"
    )
    return n_rated
