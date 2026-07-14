"""The interactive verification session loop: render each sampled card, dispatch the typed command,
and drive the review queue until quit / EOF / interrupt.

`run_session` is the entry point; it is driven by an injected input iterator + output sink, so the
whole loop is unit-testable without a terminal, model, endpoint, or GPU (it operates only on the
CSV worksheet).
"""

import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from llb.goldset.verify_base import CORPUS_DIRNAME, load_worksheet
from llb.goldset.verify_refcheck import _worksheet_bundle_hint
from llb.goldset.verify_sampling.confidence import confidence_order
from llb.goldset.verify_card import _CHAIN_DEFAULT_WIDTH, format_card
from llb.goldset.verify_commands import (
    PROMPT_HINT,
    QUIT,
    parse_command,
)
from llb.goldset.verify_session.commands import (
    _Quit,
    _default_output,
    _emit_intro,
    _get_idx,
    _handle_completion_screen,
    _handle_navigation_action,
    _maybe_clear_human_columns,
    _read,
    _save,
)
from llb.goldset.verify_session.decision import (
    SessionContext,
    _handle_decision_action,
    _handle_edit_action,
    _emit_unknown_command,
)
from llb.goldset.verify_session.report import (
    SessionStats,
    append_session_stats,
    decided_count,
    _session_record,
    summary_lines,
)


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
    idx, handled = _handle_navigation_action(
        cmd, idx, total, rows, ctx.emit, row.get("review_profile", "")
    )
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
    use_color, terminal_width = _terminal_presentation(interactive=output is None)

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

    _emit_intro(emit, rows[0].get("review_profile", ""))
    _review_until_done(ctx, idx, total, rows, emit, it)
    _save(path, rows, fieldnames)
    if stats.decisions:
        append_session_stats(path, _session_record(stats, rows))
    for line in summary_lines(rows, path, stats=stats):
        emit(line)
    return decided_count(rows)


def _terminal_presentation(*, interactive: bool) -> tuple[bool, int]:
    """`(use_color, terminal_width)` for a real terminal; plain defaults when injected."""
    interactive_terminal = interactive and sys.stdout.isatty()
    use_color = interactive_terminal and "NO_COLOR" not in os.environ
    terminal_width = (
        shutil.get_terminal_size(fallback=(_CHAIN_DEFAULT_WIDTH, 24)).columns
        if interactive_terminal
        else _CHAIN_DEFAULT_WIDTH
    )
    return use_color, terminal_width


def _review_until_done(
    ctx: "SessionContext",
    idx: int,
    total: int,
    rows: list[dict[str, str]],
    emit: Callable[[str], None],
    it: Iterator[str] | None,
) -> None:
    """Advance through the review queue until quit / EOF / interrupt (never raises)."""
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
