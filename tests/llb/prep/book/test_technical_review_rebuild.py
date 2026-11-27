"""A reviewer's edits to a formula, then a ledger rebuild that leaves them stale.

The edits settle every open span, so the formula is re-read from the corrected text and
published as math. A changed table-stage engine rebuilds the ledger: the block keeps its id but
not its anchor, so every edit is orphaned and requeued with its decision, and its replacement is
in neither rendering.
"""

from pathlib import Path

from llb.core.contracts.data_prep.book_blocks import Block
from llb.core.contracts.data_prep.book_resolve import CorrectionEdit, ReviewerDecisions
from llb.prep.book.config import read_config, write_config
from llb.prep.book.ledger.io import ledger_path, read_ledger
from llb.prep.book.resolve.anchor import block_anchor
from llb.prep.book.resolve.io import decisions_path, write_decisions
from llb.prep.book.review.build import build_review_queue
from llb.prep.book.review.io import queue_path, read_queue
from llb.prep.book.runner import resume_run
from llb.prep.book.validate.check import validate_book
from tests.llb.prep.book.technical_fixture import OPEN_FORMULA, planted_pdf, recover, registry
from tests.llb.prep.book.test_technical_content_audit import overlay, publish

_REVIEWED = r"v_1 - v_k = \Delta v \tag{2-4}"


def _formula(run: Path) -> Block:
    ledger = read_ledger(ledger_path(run, "p0001"))
    return next(block for block in ledger.blocks if block.readings[0].text == OPEN_FORMULA)


def _edit(block: Block, start: int, end: int, replacement: str, token: str) -> CorrectionEdit:
    return CorrectionEdit(
        edit_id=f"reviewer:{block.block_id}-s{start:04d}-{end:04d}",
        page_id="p0001",
        block_id=block.block_id,
        anchor=block_anchor(block),
        author="reviewer",
        start=start,
        end=end,
        raw_text=block.readings[0].text[start:end],
        replacement=replacement,
        evidence=token,
        source="reviewer",
    )


def _decide(run: Path, block: Block) -> None:
    edits = [
        _edit(block, 2, 3, "1", "accept-raw"),
        _edit(block, 4, 5, "-", "accept-raw"),
        _edit(block, 8, 9, "k", "enter-text"),
    ]
    write_decisions(
        decisions_path(run),
        ReviewerDecisions(
            schema_id="llb.book-ocr-reviewer-decisions",
            schema_version="1.0.0",
            kind="book-ocr-reviewer-decisions",
            edits=edits,
        ),
    )


def _rebuild(run: Path) -> None:
    config = read_config(run)
    stages = [
        spec.model_copy(update={"params": {"engine": "gamma"}}) if spec.name == "table" else spec
        for spec in config.stages
    ]
    write_config(run, config.model_copy(update={"stages": stages}))
    resume_run(run, registry=registry("gamma"))


def test_a_stale_reviewer_edit_is_requeued_after_the_ledger_is_rebuilt(tmp_path: Path) -> None:
    run = recover(planted_pdf(tmp_path / "library" / "planted.pdf"), tmp_path / "runs")
    build_review_queue(run)
    first = _formula(run)
    _decide(run, first)
    resumed = resume_run(run, registry=registry())
    assert resumed.outcomes.get("ran", 0) == 1
    reviewed = publish(run, tmp_path / "reviewed")
    assert _REVIEWED in reviewed.vault_text and _REVIEWED in reviewed.corpus_text
    assert not [row for row in reviewed.queue if row[0] == first.block_id]
    applied = [edit for edit in overlay(run, "p0001").applied if edit.author == "reviewer"]
    assert sorted(edit.evidence for edit in applied) == ["accept-raw", "accept-raw", "enter-text"]

    _rebuild(run)
    rebuilt_block = _formula(run)
    assert rebuilt_block.block_id == first.block_id
    assert block_anchor(rebuilt_block) != block_anchor(first)
    assert {reading.engine for reading in rebuilt_block.readings} == {"alpha", "gamma"}
    assert rebuilt_block.readings[0] == first.readings[0]
    orphaned = overlay(run, "p0001").orphaned
    assert sorted((item.edit.start, item.reason, item.requeued) for item in orphaned) == [
        (2, "block_changed", True),
        (4, "block_changed", True),
        (8, "block_changed", True),
    ]
    rebuilt = publish(run, tmp_path / "rebuilt")
    for text in (rebuilt.vault_text, rebuilt.corpus_text):
        assert "v_k" not in text and _REVIEWED not in text
    # The rebuilt readings agree on the digit and the sign; the index letter is still open, and
    # that entry carries every refused decision for the reviewer to see again.
    entries = [row for row in read_queue(queue_path(run)).entries if row.block_id == first.block_id]
    assert [(row.start, row.end, row.reason, row.verdict) for row in entries] == [
        (8, 9, "unresolved", "")
    ]
    priors = {(prior.replacement, prior.decision, prior.reason) for prior in entries[0].priors}
    assert ("k", "enter-text", "block_changed") in priors
    assert validate_book(rebuilt.vault, run).findings == ()
