"""The decision + edit command handlers and the shared `SessionContext`.

Accept/reject advance the queue and persist atomically; accept-with-edit re-grounds the edited
reference answer against the corpus IMMEDIATELY and refuses an un-groundable edit on the spot (the
ledger emission re-checks authoritatively at accept time).
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from llb.goldset.verify import (
    ACCEPT,
    REJECT,
    REJECT_CODES,
    corpus_window,
    ground_answer,
    infer_reject_code,
)
from llb.goldset.verify_card import (
    ACCEPT_CMD,
    CHECK,
    CLEAR,
    EDIT,
    NOTE,
    REJECT_CMD,
    _CHAIN_DEFAULT_WIDTH,
    Command,
    _is_chain_row,
)
from llb.goldset.verify_session.commands import (
    _clear_row,
    _go_forward,
    _read,
    _save,
    _set_check,
    _set_decision,
)
from llb.goldset.verify_session.report import SessionStats, throughput_line

_EDIT_CONFIRM_CTX_CHARS = 80  # corpus window rendered to confirm a re-grounded edit
_ESC = "\x1b"


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
