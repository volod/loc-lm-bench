"""Turn a workbench action into a reviewer edit on the correction overlay."""

from pathlib import Path

from llb.core.contracts.data_prep.book_resolve import CorrectionEdit, ReviewerDecisions
from llb.core.contracts.data_prep.book_review import ReviewQueue, ReviewQueueEntry
from llb.prep.book.resolve.constants import REVIEWER_KIND, REVIEWER_SCHEMA_ID, SCHEMA_VERSION
from llb.prep.book.resolve.io import decisions_path, read_decisions, write_decisions
from llb.prep.book.review.constants import (
    ACCEPT_CANDIDATE,
    ACCEPT_RAW,
    CLEAR_DECISION,
    ENTER_TEXT,
    MARK_ILLEGIBLE,
)
from llb.prep.book.review.io import read_queue, run_dir_of, write_queue
from llb.prep.book.review.markers import render_marker

_ACTIONS = frozenset((ACCEPT_RAW, ACCEPT_CANDIDATE, ENTER_TEXT, MARK_ILLEGIBLE, CLEAR_DECISION))


def apply_review_decision(queue_file: Path, index: int, action: str) -> ReviewQueue:
    """Record one decision and the overlay edit it becomes. The raw ledger is not touched."""
    queue = read_queue(queue_file)
    if index < 0 or index >= len(queue.entries):
        raise ValueError(f"review entry {index} is outside the queue")
    name, typed = _split(action)
    entry = queue.entries[index]
    if name not in _ACTIONS:
        raise ValueError(f"unsupported book review action: {action}")
    if name != CLEAR_DECISION and not entry.block_present:
        raise ValueError("the block is gone; the old decision is only a record")
    run_dir = run_dir_of(queue_file)
    decisions = _load(decisions_path(run_dir))
    updated_entry, updated = _mutate(entry, decisions, name, typed)
    entries = [updated_entry if item.entry_id == entry.entry_id else item for item in queue.entries]
    new_queue = queue.model_copy(update={"entries": entries})
    write_decisions(decisions_path(run_dir), updated)
    write_queue(queue_file, new_queue)
    return new_queue


def reviewer_edit_id(entry_id: str) -> str:
    """The stable id of the overlay edit one queue entry writes."""
    return f"reviewer:{entry_id}"


def _mutate(
    entry: ReviewQueueEntry, decisions: ReviewerDecisions, action: str, typed: str
) -> tuple[ReviewQueueEntry, ReviewerDecisions]:
    drop = {reviewer_edit_id(entry.entry_id), *(prior.edit_id for prior in entry.priors)}
    kept = [edit for edit in decisions.edits if edit.edit_id not in drop]
    verdict = ""
    if action != CLEAR_DECISION:
        kept.append(_edit(entry, action, typed))
        verdict = action
    kept.sort(key=lambda edit: (edit.page_id, edit.block_id, edit.start, edit.edit_id))
    updated = decisions.model_copy(update={"edits": kept})
    return entry.model_copy(update={"verdict": verdict}), updated


def _edit(entry: ReviewQueueEntry, action: str, typed: str) -> CorrectionEdit:
    return CorrectionEdit(
        edit_id=reviewer_edit_id(entry.entry_id),
        page_id=entry.page_id,
        block_id=entry.block_id,
        anchor=entry.anchor,
        author="reviewer",
        start=entry.start,
        end=entry.end,
        raw_text=entry.raw_text,
        replacement=_replacement(entry, action, typed),
        evidence=action,
        source=f"review:{entry.entry_id}",
    )


def _replacement(entry: ReviewQueueEntry, action: str, typed: str) -> str:
    if action == ACCEPT_RAW:
        return entry.raw_text
    if action == ACCEPT_CANDIDATE:
        if entry.candidate is None:
            raise ValueError("no candidate reading for this span")
        return entry.candidate
    if action == ENTER_TEXT:
        if not typed.strip() or "\n" in typed or "\r" in typed:
            raise ValueError("enter a single line of replacement text")
        return typed
    return render_marker(
        "unreadable", entry.pdf_link, entry.block_id, entry.start, entry.end, entry.crop
    )


def _split(action: str) -> tuple[str, str]:
    prefix = f"{ENTER_TEXT}:"
    if action.startswith(prefix):
        return ENTER_TEXT, action[len(prefix) :]
    return action, ""


def _load(path: Path) -> ReviewerDecisions:
    if not path.is_file():
        return ReviewerDecisions(
            schema_id=REVIEWER_SCHEMA_ID,
            schema_version=SCHEMA_VERSION,
            kind=REVIEWER_KIND,
            edits=[],
        )
    return read_decisions(path)
