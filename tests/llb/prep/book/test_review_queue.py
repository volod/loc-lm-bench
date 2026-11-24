"""Markers, crops, the quality manifest, and the review-queue adapter."""

from pathlib import Path

from llb.core.contracts.data_prep.book_resolve import CorrectionEdit
from llb.prep.book.ledger.assemble import assemble
from llb.prep.book.preprocess.images import read_grey
from llb.prep.book.resolve.anchor import block_anchor
from llb.prep.book.resolve.apply import resolve_page
from llb.prep.book.resolve.constants import REVIEW_DIRNAME
from llb.prep.book.resolve.io import decisions_path, read_decisions
from llb.prep.book.review.build import build_review_queue
from llb.prep.book.review.decide import apply_review_decision
from llb.prep.book.review.io import queue_path, read_manifest, read_queue
from llb.prep.book.review.markers import parse_marker, render_marker
from llb.review.adapters.book_queue import BookQueueAdapter
from llb.review.core import ReviewNavigator
from llb.review.registry import open_review
from tests.llb.prep.book._review_factory import HEIGHT, WIDTH
from tests.llb.prep.book._review_factory import body as _body
from tests.llb.prep.book._review_factory import choice, ledger_bytes, read_page_ledger, reresolve
from tests.llb.prep.book._review_factory import store, write_run


def test_marker_round_trip() -> None:
    text = render_marker(
        "unreadable",
        "source/book.pdf#page=3",
        "p0003-b0001",
        2,
        5,
        "review/p0003-b0001-s0002-0005.png",
    )
    parsed = parse_marker(text)
    assert parsed is not None
    assert parsed.kind == "unreadable"
    assert parsed.pdf_link == "source/book.pdf#page=3"
    assert (parsed.block_id, parsed.start, parsed.end) == ("p0003-b0001", 2, 5)
    assert parsed.crop == "review/p0003-b0001-s0002-0005.png"
    assert parse_marker("ordinary prose") is None


def test_unreadable_span_is_marked_cropped_and_queued(tmp_path: Path) -> None:
    run = write_run(tmp_path, _body(""))
    before = ledger_bytes(run)
    queue, manifest = build_review_queue(run)
    entry = queue.entries[0]
    assert entry.kind == "unreadable"
    assert entry.rendered.strip()
    assert "[!unreadable]" in entry.rendered
    assert parse_marker(entry.marker) is not None
    assert entry.crop is not None
    crop = run / entry.crop
    assert crop.is_file()
    assert read_grey(crop).shape == (56, 116)
    assert manifest.pages[0].rows[0].entry_id == entry.entry_id
    assert manifest.counts == {"unreadable": 1}
    assert ledger_bytes(run) == before


def test_missing_image_still_marks_the_span(tmp_path: Path) -> None:
    run = write_run(tmp_path, _body(""), image=False)
    queue, manifest = build_review_queue(run)
    entry = queue.entries[0]
    assert entry.crop is None
    assert "![[" not in entry.marker
    assert entry.rendered.strip()
    assert manifest.pages[0].rows


def test_formula_disagreement_uses_its_own_marker(tmp_path: Path) -> None:
    run = write_run(
        tmp_path, _body("v=1", kind="formula"), _body("v=l", kind="formula", engine="b")
    )
    queue, _manifest = build_review_queue(run)
    assert [entry.kind for entry in queue.entries] == ["formula-table"]
    assert queue.entries[0].candidate == "l" or "l" in (queue.entries[0].candidate or "")


def test_low_confidence_is_queued_and_a_sure_reading_is_not(tmp_path: Path) -> None:
    run = write_run(
        tmp_path,
        _body("sure", confidence=0.99, y0=0, y1=30),
        _body("weak", confidence=0.89, y0=80, y1=120),
        _body("", kind="running-header", y0=200, y1=220),
    )
    queue, manifest = build_review_queue(run)
    assert [entry.kind for entry in queue.entries] == ["low-confidence"]
    assert "weak" in queue.entries[0].rendered
    assert "[!low-confidence]" in queue.entries[0].rendered
    assert manifest.pages[0].rows[0].reason == "low-confidence"


def test_decision_changes_resolved_text_and_leaves_the_ledger(tmp_path: Path) -> None:
    run = write_run(tmp_path, _body("The value is 0.05."), _body("The value is O.05.", engine="b"))
    before = ledger_bytes(run)
    build_review_queue(run)
    apply_review_decision(queue_path(run), 0, "accept-raw")
    assert ledger_bytes(run) == before
    resolved = reresolve(run)
    assert resolved.blocks[0].text == "The value is 0.05."
    again = reresolve(run)
    assert again.blocks[0].text == resolved.blocks[0].text
    assert ledger_bytes(run) == before
    rebuilt, _manifest = build_review_queue(run)
    assert rebuilt.entries == []


def test_candidate_entered_text_and_illegible(tmp_path: Path) -> None:
    run = write_run(tmp_path, _body("The value is 0.05."), _body("The value is O.05.", engine="b"))
    build_review_queue(run)
    apply_review_decision(queue_path(run), 0, "accept-candidate")
    assert reresolve(run).blocks[0].text == "The value is O.05."
    apply_review_decision(queue_path(run), 0, "enter-text:9")
    assert "9" in reresolve(run).blocks[0].text
    apply_review_decision(queue_path(run), 0, "mark-illegible")
    text = reresolve(run).blocks[0].text
    assert "[!unreadable]" in text
    assert text.strip()


def test_reopening_the_queue_resumes_at_the_pending_entry(tmp_path: Path) -> None:
    run = write_run(
        tmp_path,
        _body("", y0=0, y1=30),
        _body("weak", confidence=0.89, y0=80, y1=120),
    )
    build_review_queue(run)
    adapter = BookQueueAdapter(queue_path(run))
    adapter.apply(0, "accept-raw")
    reopened = BookQueueAdapter(queue_path(run))
    assert reopened.first_pending() == 1
    assert ReviewNavigator(reopened).index == 1


def test_orphaned_decision_returns_with_its_old_text(tmp_path: Path) -> None:
    run = write_run(tmp_path, _body("alpha"))
    build_review_queue(run)
    assert read_queue(queue_path(run)).entries == []
    ledger = read_page_ledger(run)
    block = ledger.blocks[0]
    edit = CorrectionEdit(
        edit_id="reviewer:old",
        page_id=ledger.page_id,
        block_id=block.block_id,
        anchor=block_anchor(block),
        author="reviewer",
        start=0,
        end=len("alpha"),
        raw_text="alpha",
        replacement="beta",
        evidence="enter-text",
        source="review:old",
    )
    changed = assemble("p0001", 1, WIDTH, HEIGHT, (_body("alphabet"),))
    overlay, resolved = resolve_page(changed, choice(), [edit])
    assert overlay.orphaned and overlay.orphaned[0].reason == "block_changed"
    store(run, changed, resolved, overlay)
    queue, _manifest = build_review_queue(run)
    entry = queue.entries[0]
    assert entry.verdict == ""
    assert entry.priors[0].replacement == "beta"
    record = BookQueueAdapter(queue_path(run)).record(0)
    assert "beta" in record.sections[2].text
    assert "alphabet" in entry.rendered


def test_adapter_matches_the_decision_writer(tmp_path: Path) -> None:
    legacy = write_run(tmp_path / "legacy", _body(""))
    adapted = write_run(tmp_path / "adapted", _body(""))
    build_review_queue(legacy)
    build_review_queue(adapted)
    apply_review_decision(queue_path(legacy), 0, "accept-raw")
    BookQueueAdapter(queue_path(adapted)).apply(0, "accept-raw")
    assert queue_path(legacy).read_bytes() == queue_path(adapted).read_bytes()
    assert decisions_path(legacy).read_bytes() == decisions_path(adapted).read_bytes()
    assert open_review(queue_path(adapted)).kind == "book-review-queue"
    assert open_review(adapted).kind == "book-review-queue"
    assert read_decisions(decisions_path(adapted)).edits[0].author == "reviewer"


def test_rebuild_is_byte_identical_and_drops_a_stale_crop(tmp_path: Path) -> None:
    run = write_run(tmp_path, _body(""))
    build_review_queue(run)
    queue_bytes = queue_path(run).read_bytes()
    crop = next((run / REVIEW_DIRNAME).glob("*.png"))
    crop_bytes = crop.read_bytes()
    stale = run / REVIEW_DIRNAME / "p0001-b0099-s0000-0001.png"
    stale.write_bytes(crop_bytes)
    build_review_queue(run)
    assert queue_path(run).read_bytes() == queue_bytes
    assert crop.read_bytes() == crop_bytes
    assert not stale.exists()


def test_review_queue_command(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from llb.cli.app import app

    run = write_run(tmp_path, _body(""))
    result = CliRunner().invoke(app, ["book-ocr", "review-queue", str(run)])
    assert result.exit_code == 0, result.output
    assert "unreadable: 1" in result.output
    assert read_manifest(run / "metadata" / "quality_manifest.json").counts["unreadable"] == 1
    refused = CliRunner().invoke(app, ["book-ocr", "review-queue", str(tmp_path / "missing")])
    assert refused.exit_code == 2
