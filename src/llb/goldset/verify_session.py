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
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from llb.core.fsutil import atomic_write_text
from llb.goldset.verify import (
    ACCEPT,
    CHECK_COLS,
    CORPUS_DIRNAME,
    FAIL,
    HUMAN_COLS,
    KIND_CHAINS,
    PASS,
    REJECT,
    REJECT_CODES,
    STATUS_DECIDED,
    STATUS_PENDING,
    _worksheet_bundle_hint,
    confidence_order,
    corpus_window,
    ground_answer,
    infer_reject_code,
    load_worksheet,
    write_worksheet_rows,
)

SESSION_STATS_FILENAME = "verify_session_stats.json"
_EDIT_CONFIRM_CTX_CHARS = 80  # corpus window rendered to confirm a re-grounded edit
_CHAIN_SEPARATOR = "+" * 64
_CHAIN_DEFAULT_WIDTH = 120
_CHAIN_MIN_VALUE_WIDTH = 24
_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_QUESTION = "\033[1;36m"
_ANSI_ANSWER = "\033[1;32m"
_ANSI_SOURCE = "\033[33m"
_ANSI_DEPENDENCY = "\033[35m"

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
EDIT = "edit"
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
# Command keys mirror the external-RAG review session (`llb.scoring.external_rag_session`) where
# the two tools do the same thing: o=note and w=edit-the-answer are shared aliases, and the
# navigation row (Enter/n, b, u, j<N>, ?, q) is identical. The decision keys necessarily differ:
# here a/r/p mark CHECKS (answerable/reference/planted), so accept/reject are y/x.
_SIMPLE_COMMANDS = {
    "": NEXT,
    "n": NEXT,
    "b": PREV,
    "u": UNDECIDED,
    "c": CLEAR,
    "y": ACCEPT_CMD,
    "x": REJECT_CMD,
    "e": EDIT,
    "w": EDIT,
    "q": QUIT,
    "quit": QUIT,
    "?": HELP,
    "h": HELP,
    "help": HELP,
    "o": NOTE,
    "note": NOTE,
}

PROMPT_HINT = (
    "decide: y=accept, x=reject (code inferred), x <code>=coded reject; "
    "checks: g/a/r/p=PASS, G/A/R/P=FAIL\n"
    "edit/nav: e/w=edit answer, o=note, c=clear, Enter/n=next, b=prev, u=undecided, "
    "j<N>=jump, ?=help, q=quit"
)


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


def _parse_reject_code_command(s: str) -> Command | None:
    """`x <code>` -- reject with an explicit coded reason (bare `x` stays a simple command)."""
    if not s.lower().startswith("x "):
        return None
    return Command(REJECT_CMD, field=s[2:].strip().lower())


def parse_command(raw: str) -> Command:
    """Parse one prompt line into a `Command` (pure; no I/O, no state).

    Empty / `n` = next; `b`/up/left arrow = previous; `g/a/r/p` mark a check PASS, the uppercase
    forms FAIL; `y` = accept, `x` = reject (code inferred from failed checks), `x <code>` = reject
    with an explicit coded reason; `e` = edit the reference answer (re-grounded immediately);
    `note` = edit a note; `j <N>`/`jN` = jump; `u` = next undecided; `c` = clear this item;
    `?`/`h` = help; `q` = save + quit.
    """
    s = raw.strip()
    if s in _ARROWS:
        return Command(_ARROWS[s])
    parsers = (
        _parse_simple_command,
        _parse_check_command,
        _parse_reject_code_command,
        _parse_jump_command,
    )
    for parser in parsers:
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


def summary_lines(
    rows: Sequence[dict[str, str]], path: Path, stats: "SessionStats | None" = None
) -> list[str]:
    """The end-of-session report: progress, accept/reject split, pace, and the next command."""
    total = len(rows)
    decided = decided_count(rows)
    accepted, rejected = decision_tally(rows)
    lines = [
        f"[verify] saved {path}",
        f"[verify] progress : {decided}/{total} decided, {total - decided} remaining "
        f"(accept {accepted}, reject {rejected})",
    ]
    if stats is not None and stats.decisions:
        lines.append(
            f"[verify] pace     : {stats.decisions} decided this session in "
            f"{stats.elapsed_seconds() / 60.0:.1f} min -- {stats.items_per_hour():.1f} items/h "
            f"(recorded in {SESSION_STATS_FILENAME})"
        )
    if decided < total:
        lines.append(
            "[verify] resume   : re-run `make verify-review` (continues at the first undecided item)"
        )
    lines.append(f"[verify] accept   : make verify-accept VERIFY_WS={path} BUNDLE=<draft bundle>")
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
            f"===== all {total} items decided (accept={accepted}, reject={rejected}) =====",
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


def _is_chain_row(row: dict[str, str]) -> bool:
    return (row.get("item_kind") or "").strip() == KIND_CHAINS


def _indent(text: str, prefix: str = "    ") -> str:
    if not text:
        return prefix.rstrip()
    return "\n".join(prefix + line for line in text.splitlines())


def _one_line(value: object) -> str:
    return " ".join(str(value or "").split())


def _truncate(value: object, limit: int, *, blank: str = "(none)") -> str:
    text = _one_line(value) or blank
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 3)].rstrip() + "..."


def _context_excerpt(value: object, limit: int) -> str:
    text = _one_line(value) or "(missing)"
    cited_start = text.find(">>>")
    cited_end = text.find("<<<", cited_start + 3)
    if cited_start < 0 or cited_end < 0:
        return _truncate(text, limit, blank="(missing)")
    cited_end += 3
    cited = text[cited_start:cited_end]
    if len(cited) >= limit:
        return _truncate(cited, limit, blank="(missing)")
    context_budget = max(0, limit - len(cited) - 6)
    before = context_budget // 2
    after = context_budget - before
    start = max(0, cited_start - before)
    end = min(len(text), cited_end + after)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end] + suffix


def _color(text: str, code: str, enabled: bool) -> str:
    return f"{code}{text}{_ANSI_RESET}" if enabled else text


def _chain_field(
    label: str,
    value: object,
    *,
    width: int,
    color: str = "",
    use_color: bool = False,
    blank: str = "(none)",
) -> str:
    available = max(_CHAIN_MIN_VALUE_WIDTH, width - len(label) - 1)
    line = f"{label} {_truncate(value, available, blank=blank)}"
    return _color(line, color, use_color and bool(color))


def _format_chain_steps(
    row: dict[str, str], *, color: bool = False, width: int = _CHAIN_DEFAULT_WIDTH
) -> list[str]:
    raw = (row.get("chain_steps") or "").strip()
    if not raw:
        return []
    try:
        steps = json.loads(raw)
    except json.JSONDecodeError:
        return ["== chain_steps: (invalid JSON)"]
    if not isinstance(steps, list):
        return ["== chain_steps: (invalid JSON)"]
    valid_steps = [step for step in steps if isinstance(step, dict)]
    lines: list[str] = []
    for index, step in enumerate(valid_steps, start=1):
        order = str(step.get("order", ""))
        page = str(step.get("page_citation") or "(none)")
        doc = _truncate(step.get("span_doc_id"), max(_CHAIN_MIN_VALUE_WIDTH, width // 2))
        header = f"STEP {order or index}/{len(valid_steps)} | doc={doc} | page={page}"
        lines.append(_color(header, _ANSI_BOLD, color))
        lines.append(
            _chain_field(
                "Q:", step.get("question"), width=width, color=_ANSI_QUESTION, use_color=color
            )
        )
        lines.append(
            _chain_field(
                "A:",
                step.get("reference_answer"),
                width=width,
                color=_ANSI_ANSWER,
                use_color=color,
            )
        )
        lines.append(
            _chain_field(
                "SOURCE:",
                step.get("span_text"),
                width=width,
                color=_ANSI_SOURCE,
                use_color=color,
                blank="(missing)",
            )
        )
        dependency = str(step.get("dependency_note") or "").strip()
        if dependency:
            lines.append(
                _chain_field(
                    "DEPENDENCY:",
                    dependency,
                    width=width,
                    color=_ANSI_DEPENDENCY,
                    use_color=color,
                )
            )
        context = str(step.get("context") or "").strip()
        if context:
            available = max(_CHAIN_MIN_VALUE_WIDTH, width - len("CONTEXT:") - 1)
            lines.append(f"CONTEXT: {_context_excerpt(context, available)}")
    return lines


def _chain_checks(row: dict[str, str]) -> str:
    checks = [
        f"{col.removeprefix('chk_')}={_field(row, col, '(unchecked)')}"
        for col in CHECK_COLS
        if col != "chk_planted"
    ]
    return "CHECKS: " + " | ".join(checks)


def format_card(
    row: dict[str, str],
    position: int,
    total: int,
    decided: int,
    *,
    show_crosscheck: bool = False,
    color: bool = False,
    width: int = _CHAIN_DEFAULT_WIDTH,
) -> str:
    """Render a review card; chain rows use a dense one-line-per-field terminal layout."""
    remaining = total - decided
    rank = (row.get("retrieval_rank") or "").strip() or "(none)"
    page = (row.get("page_citation") or "").strip() or "(none)"
    lines = [
        "===== goldset verification review =====",
        f"item {position}/{total} (decided {decided}, remaining {remaining})",
        f"== id: {row.get('item_id', '')}",
        f"== meta: stratum={row.get('stratum', '')} synthetic={row.get('synthetic', '')} "
        f"retrieval_rank={rank}",
    ]
    reviewer = (row.get("reviewer_id") or "").strip()
    if reviewer:
        lines.append(f"== reviewer: {reviewer}")
    priors = (row.get("prior_decisions") or "").strip()
    if priors:
        lines.append(f"== prior_decisions: {priors} -- decide independently, then compare")
    is_chain = _is_chain_row(row)
    if is_chain:
        decision = _field(row, "decision", "(undecided)")
        lines = [
            _CHAIN_SEPARATOR,
            f"CHAIN {position}/{total} | id={row.get('item_id', '')} | decided={decided} "
            f"remaining={remaining} | decision={decision}",
            "REVIEW: compare A with SOURCE; confirm Q is answered and later steps add context.",
        ]
        lines.extend(_format_chain_steps(row, color=color, width=width))
        lines.append(_chain_checks(row))
    else:
        lines += [
            "",
            f"== question: {row.get('question', '')}",
            f"== reference_answer: {row.get('reference_answer', '')}",
            f"== span: doc={row.get('span_doc_id', '')} page={page}",
            "== context (cited span between >>> <<<)",
            _indent(row.get("context", "") or "(missing)"),
            "== checks:",
        ]
    if not is_chain:
        synthetic = _is_synthetic_row(row)
        for col in CHECK_COLS:
            if col == "chk_planted" and not synthetic:
                continue
            lines.append(f"    {col:<15}: {_field(row, col, '(unchecked)')}  -- {CHECK_LABEL[col]}")
        lines.append(f"== decision: {_field(row, 'decision', '(undecided)')}")
    edited = (row.get("edited_answer") or "").strip()
    if edited:
        lines.append(f"== edited_answer: {edited}")
    code = (row.get("reject_code") or "").strip()
    if code:
        lines.append(f"== reject_code: {code}")
    note = (row.get("human_note") or "").strip()
    if note:
        lines.append(f"== human_note: {note}")
    if show_crosscheck:
        cc = "  ".join(
            f"{key}={_field(row, f'cc_{key}', '?')}"
            for key in ("grounded", "non_circular", "supported", "answerable")
        )
        lines.append(f"== crosscheck: {cc}  note={_field(row, 'cc_note', '')}")
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
            "  y        accept (within tolerance)",
            "  x        reject (code inferred from failed checks)",
            f"  x <code> reject with an explicit code: {', '.join(REJECT_CODES)}",
            "edits:",
            "  e / w    edit the reference answer (accept-with-edit; re-grounded immediately)",
            "  o / note edit a note",
            "  c        clear this item's marks",
            "navigation:",
            "  n/Enter  next                           b        previous",
            "  j <N>    jump to item N                 u        next undecided",
            "  ?/h      this help                      q        save + quit",
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


# --- session throughput stats (the items-per-hour evidence) --------------------------------


@dataclass
class SessionStats:
    """Wall-clock throughput of ONE review sitting.

    Decisions per elapsed hour is the measured reviewer-throughput number the human evidence
    records; the clock is injected so tests never sleep.
    """

    clock: Callable[[], float]
    started: float = 0.0
    decisions: int = 0

    def __post_init__(self) -> None:
        self.started = self.clock()

    def on_decision(self) -> None:
        self.decisions += 1

    def elapsed_seconds(self) -> float:
        return max(self.clock() - self.started, 0.0)

    def items_per_hour(self) -> float:
        elapsed = self.elapsed_seconds()
        if not self.decisions or elapsed <= 0:
            return 0.0
        return self.decisions * 3600.0 / elapsed


def throughput_line(stats: SessionStats, rows: Sequence[dict[str, str]]) -> str:
    """One-line session pace: decided count, items/hour, and the ETA for the remaining rows."""
    remaining = len(rows) - decided_count(rows)
    rate = stats.items_per_hour()
    minutes = stats.elapsed_seconds() / 60.0
    line = f"[stats] session: {stats.decisions} decided in {minutes:.1f} min"
    if rate > 0:
        line += f" -- {rate:.1f} items/h"
        if remaining:
            line += f"; ~{remaining * 60.0 / rate:.0f} min for {remaining} remaining"
    return line


def append_session_stats(worksheet_path: Path, record: dict[str, object]) -> Path:
    """Append one session record to `verify_session_stats.json` beside the worksheet.

    The durable trace of measured reviewer throughput (what the current docs cite), so a
    finished 40-item pass does not live only in scrollback.
    """
    path = Path(worksheet_path).with_name(SESSION_STATS_FILENAME)
    payload: dict[str, object] = {"sessions": []}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("sessions"), list):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            pass
    sessions = payload["sessions"]
    assert isinstance(sessions, list)
    sessions.append(record)
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def _session_record(stats: SessionStats, rows: Sequence[dict[str, str]]) -> dict[str, object]:
    return {
        "decided_this_session": stats.decisions,
        "elapsed_seconds": round(stats.elapsed_seconds(), 1),
        "items_per_hour": round(stats.items_per_hour(), 1),
        "total_decided": decided_count(rows),
        "total_rows": len(rows),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


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


def _doc_text_for_row(corpus_root: Path | None, row: dict[str, str]) -> str | None:
    """The span doc's full text, or None when the bundle corpus is not reachable."""
    if corpus_root is None:
        return None
    doc_id = (row.get("span_doc_id") or "").strip()
    if not doc_id:
        return None
    path = Path(corpus_root) / doc_id
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _edit_still_grounds(corpus_root: Path | None, row: dict[str, str]) -> bool:
    """Whether the row's `edited_answer` (if any) is still a verbatim span of its doc.

    Returns True when there is no edit or the corpus is unreachable (the ledger emission
    re-checks authoritatively and raises); False only on a positively failed re-ground.
    """
    edited = (row.get("edited_answer") or "").strip()
    if not edited:
        return True
    text = _doc_text_for_row(corpus_root, row)
    if text is None:
        return True
    return ground_answer(text, edited) is not None


def _reject_code_for(cmd: Command, row: dict[str, str], emit: Callable[[str], None]) -> str | None:
    """Resolve the coded rejection reason: explicit `x <code>` (validated) or inferred."""
    explicit = cmd.field.strip().lower()
    if not explicit:
        code = infer_reject_code(row)
        emit(f"[verify] reject code: {code} (inferred; `x <code>` to override)")
        return code
    if explicit not in REJECT_CODES:
        emit(f"[verify] unknown reject code {explicit!r}; use one of: {', '.join(REJECT_CODES)}")
        return None
    return explicit


def _handle_decision_action(
    cmd: Command,
    row: dict[str, str],
    idx: int,
    total: int,
    ctx: "SessionContext",
) -> tuple[int, bool]:
    if cmd.kind == ACCEPT_CMD:
        if not _edit_still_grounds(ctx.corpus_root, row):
            ctx.emit(
                "[verify] BLOCKED: the edited answer no longer matches a verbatim span of "
                f"{row.get('span_doc_id', '')} -- re-ground it with `e` before accepting."
            )
            return idx, True
        _set_decision(row, ACCEPT)
        row["reject_code"] = ""
    elif cmd.kind == REJECT_CMD:
        code = _reject_code_for(cmd, row, ctx.emit)
        if code is None:
            return idx, True
        _set_decision(row, REJECT)
        row["reject_code"] = code
    else:
        return idx, False

    _save(ctx.path, ctx.rows, ctx.fieldnames)
    ctx.stats.on_decision()
    ctx.emit(throughput_line(ctx.stats, ctx.rows))
    return _go_forward(idx, total, ctx.rows, ctx.emit), True


def _handle_answer_edit(row: dict[str, str], ctx: "SessionContext") -> None:
    """Accept-with-edit: capture an edited reference answer and re-ground it IMMEDIATELY.

    The edit is stored only when the new answer exists verbatim in the span's corpus doc; an
    un-groundable edit is refused on the spot (and `emit_accepted_ledger` re-checks at accept
    time, so a hand-edited CSV cannot certify either).
    """
    if _is_chain_row(row):
        ctx.emit(
            "[verify] chain answer edits are not supported; use o=note and reject the chain "
            "if any step needs a different span."
        )
        return
    text = _read("edited reference answer (empty to clear): ", ctx.it, ctx.emit).strip()
    if not text:
        row["edited_answer"] = ""
        _save(ctx.path, ctx.rows, ctx.fieldnames)
        ctx.emit("[verify] edit cleared; the original reference answer stands.")
        return
    doc_text = _doc_text_for_row(ctx.corpus_root, row)
    if doc_text is None:
        ctx.emit(
            "[verify] BLOCKED: cannot re-ground the edit -- bundle corpus not reachable "
            "(is sample_manifest.json beside the worksheet?). Edit not saved."
        )
        return
    hint = max(doc_text.find((row.get("span_text") or "").strip()), 0)
    offsets = ground_answer(doc_text, text, hint_start=hint)
    if offsets is None:
        ctx.emit(
            f"[verify] BLOCKED: edited answer is not a verbatim span of "
            f"{row.get('span_doc_id', '')} -- not saved. Re-word it to exact corpus text."
        )
        return
    row["edited_answer"] = text
    _save(ctx.path, ctx.rows, ctx.fieldnames)
    ctx.emit(
        "[verify] edit re-grounded: "
        + corpus_window(doc_text, offsets[0], offsets[1], ctx=_EDIT_CONFIRM_CTX_CHARS)
    )


def _handle_edit_action(
    cmd: Command,
    row: dict[str, str],
    ctx: "SessionContext",
) -> bool:
    if cmd.kind == CHECK:
        if _set_check(row, cmd, ctx.emit):
            _save(ctx.path, ctx.rows, ctx.fieldnames)
    elif cmd.kind == CLEAR:
        _clear_row(row)
        _save(ctx.path, ctx.rows, ctx.fieldnames)
    elif cmd.kind == EDIT:
        _handle_answer_edit(row, ctx)
    elif cmd.kind == NOTE:
        text = _read("note (empty to clear): ", ctx.it, ctx.emit).strip()
        row["human_note"] = text
        _save(ctx.path, ctx.rows, ctx.fieldnames)
    else:
        return False
    return True


def _emit_unknown_command(cmd: Command, emit: Callable[[str], None]) -> None:
    if cmd.raw.startswith(_ESC):
        emit("[verify] arrow keys garbled -- use n (next) / b (prev).")
    else:
        emit(f"[verify] not a command: {cmd.raw!r} (? for help).")


@dataclass
class SessionContext:
    """Everything one review sitting shares across handlers (I/O, worksheet, corpus, stats)."""

    path: Path
    fieldnames: Sequence[str]
    rows: list[dict[str, str]]
    emit: Callable[[str], None]
    it: Iterator[str] | None
    corpus_root: Path | None
    stats: SessionStats
    show_crosscheck: bool = False
    color: bool = False
    terminal_width: int = _CHAIN_DEFAULT_WIDTH


def _handle_row_action(ctx: SessionContext, idx: int, total: int) -> int:
    rows = ctx.rows
    row = rows[idx]
    ctx.emit(
        format_card(
            row,
            idx + 1,
            total,
            decided_count(rows),
            show_crosscheck=ctx.show_crosscheck,
            color=ctx.color,
            width=ctx.terminal_width,
        )
    )
    cmd = parse_command(_read(f"{PROMPT_HINT}\nverify> ", ctx.it, ctx.emit))

    if cmd.kind == QUIT:
        raise _Quit
    idx, handled = _handle_navigation_action(cmd, idx, total, rows, ctx.emit)
    if handled:
        return idx

    idx, handled = _handle_decision_action(cmd, row, idx, total, ctx)
    if handled:
        return idx

    if _handle_edit_action(cmd, row, ctx):
        return idx

    _emit_unknown_command(cmd, ctx.emit)
    return idx


def _resolve_corpus_root(worksheet_path: Path) -> Path | None:
    """The bundle `corpus/` dir named by the sibling `sample_manifest.json`, if reachable."""
    bundle = _worksheet_bundle_hint(worksheet_path)
    if bundle is None:
        return None
    corpus = Path(bundle) / CORPUS_DIRNAME
    return corpus if corpus.is_dir() else None


def _session_view(rows: list[dict[str, str]], order: str) -> list[dict[str, str]]:
    """The review-queue view of `rows`: worksheet order, or least-confident first.

    Reordering the view (the SAME row dicts) is enough: saves merge human columns back into the
    on-disk worksheet by item id, so the CSV row order never changes.
    """
    if order == "confidence":
        return [rows[i] for i in confidence_order(rows)]
    return list(rows)


def run_session(
    worksheet_path: Path | str,
    *,
    inputs: Iterable[str] | None = None,
    output: Callable[[str], None] | None = None,
    start: int | None = None,
    show_crosscheck: bool = False,
    clear: bool = False,
    order: str = "worksheet",
    corpus_root: Path | str | None = None,
    clock: Callable[[], float] | None = None,
) -> int:
    """Drive the interactive verification over `worksheet_path`; return the decided count.

    `inputs` / `output` are injected for testing; in a real terminal they default to `input()` /
    stdout. Every edit writes the whole CSV atomically, so a crash, EOF, or Ctrl-C never loses
    work. With no `start`, resume at the first undecided item. `clear` wipes all human columns
    first (confirmation-gated). `order="confidence"` reviews least-confident items first without
    reordering the CSV. `corpus_root` (default: resolved from the sibling `sample_manifest.json`)
    enables accept-with-edit re-grounding. `clock` is injected for stats tests; the session
    appends its measured items-per-hour to `verify_session_stats.json` beside the worksheet.
    """
    path = Path(worksheet_path)
    emit = output or _default_output
    it: Iterator[str] | None = iter(inputs) if inputs is not None else None
    interactive_terminal = output is None and sys.stdout.isatty()
    use_color = interactive_terminal and "NO_COLOR" not in os.environ
    terminal_width = (
        shutil.get_terminal_size(fallback=(_CHAIN_DEFAULT_WIDTH, 24)).columns
        if interactive_terminal
        else _CHAIN_DEFAULT_WIDTH
    )

    disk_rows, fieldnames = load_worksheet(path)
    if not disk_rows:
        emit(f"[verify] worksheet has no rows: {path}")
        return 0

    if not _maybe_clear_human_columns(clear, disk_rows, path, fieldnames, it, emit):
        return decided_count(disk_rows)

    rows = _session_view(disk_rows, order)
    if corpus_root is None:
        resolved_corpus = _resolve_corpus_root(path)
    else:
        resolved_corpus = Path(corpus_root)
    stats = SessionStats(clock=clock or time.monotonic)
    ctx = SessionContext(
        path=path,
        fieldnames=fieldnames,
        rows=rows,
        emit=emit,
        it=it,
        corpus_root=resolved_corpus,
        stats=stats,
        show_crosscheck=show_crosscheck,
        color=use_color,
        terminal_width=terminal_width,
    )

    total = len(rows)
    idx = _get_idx(start, total, rows)

    _emit_intro(emit)

    try:
        while True:
            idx, is_completion = _handle_completion_screen(idx, total, rows, emit, it)
            if is_completion:
                continue

            idx = _handle_row_action(ctx, idx, total)
    except (_Quit, EOFError):
        pass
    except KeyboardInterrupt:
        emit("")

    _save(path, rows, fieldnames)
    if stats.decisions:
        append_session_stats(path, _session_record(stats, rows))
    for line in summary_lines(rows, path, stats=stats):
        emit(line)
    return decided_count(rows)
