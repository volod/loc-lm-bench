"""The resolution rule and the correction overlay, over synthetic ledgers."""

import json
from pathlib import Path

import pytest
from jsonschema.validators import validator_for
from pydantic import ValidationError

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.core.contracts.data_prep.book_blocks import PageBlockLedger
from llb.core.contracts.data_prep.book_resolve import AuthorKind, CorrectionEdit, ReviewerDecisions
from llb.core.paths import PROJECT_ROOT
from llb.prep.book.cache import digest_json
from llb.prep.book.engines.protocol import EngineRead, Proposed
from llb.prep.book.ledger.assemble import assemble
from llb.prep.book.resolve.anchor import block_anchor
from llb.prep.book.resolve.apply import resolve_page
from llb.prep.book.resolve.config import page_choice
from llb.prep.book.resolve.io import reviewer_digest, write_decisions
from llb.prep.book.resolve.primary import PrimaryChoice

_SCHEMA = (
    PROJECT_ROOT / "schemas" / "artifacts" / "llb.book-ocr-resolved-page" / "1.0.0.schema.json"
)
_WIDTH = 200
_HEIGHT = 400
_TEXT = "The value is 0.05."


def _choice(engine: str = "a", page_class: str = "") -> PrimaryChoice:
    return PrimaryChoice("ocr", engine, "original", None, page_class)


def _engine(
    engine: str, text: str, y0: float = 0.0, y1: float = 40.0, order: int = 0
) -> EngineRead:
    block = Proposed("body", order, 0.0, y0, 100.0, y1, text, None)
    return EngineRead(engine, "m", "original", "ocr", (block,))


def _ledger(*reads: EngineRead, page: str = "p0001") -> PageBlockLedger:
    return assemble(page, 1, _WIDTH, _HEIGHT, reads)


def _edit(
    block, author: AuthorKind, replacement: str, start: int | None = None, end: int | None = None
) -> CorrectionEdit:
    start = 0 if start is None else start
    end = len(block.readings[0].text) if end is None else end
    raw = block.readings[0].text[start:end]
    # The raw slice has to be the primary reading's slice. Callers pass the primary block.
    return CorrectionEdit(
        edit_id=f"{author}-{block.block_id}-{start}",
        page_id=block.block_id.split("-b")[0],
        block_id=block.block_id,
        anchor=block_anchor(block),
        author=author,
        start=start,
        end=end,
        raw_text=raw,
        replacement=replacement,
        evidence=f"{author} confirmed the crop",
        source=author,
    )


def _resolved(ledger: PageBlockLedger, *edits: CorrectionEdit, engine: str = "a"):
    before = ledger.model_dump(mode="json")
    overlay, page = resolve_page(ledger, _choice(engine), list(edits))
    assert ledger.model_dump(mode="json") == before
    return overlay, page


def test_a_digit_span_stays_unresolved_on_the_primary_reading():
    ledger = _ledger(
        _engine("a", _TEXT),
        _engine("b", _TEXT),
        _engine("c", "The value is O.05."),
    )
    _overlay, page = _resolved(ledger)
    block = page.blocks[0]
    unresolved = [span for span in block.spans if span.provenance == "unresolved"]
    assert len(unresolved) == 1
    assert unresolved[0].text == "0" and unresolved[0].classes == ["digit"]
    assert unresolved[0].adopted is False
    assert block.text == "The value is .05."
    assert "O" not in block.text
    assert block.engine == "a" and block.fallback is False
    assert any(span.provenance == "agreement" for span in block.spans)


def test_agreement_across_every_reading_resolves_and_a_majority_does_not():
    agreed = _ledger(_engine("a", _TEXT), _engine("b", _TEXT), _engine("c", _TEXT))
    overlay, page = _resolved(agreed)
    assert page.blocks[0].text == _TEXT
    assert {span.provenance for span in page.blocks[0].spans} == {"agreement"}
    assert overlay.applied and {edit.author for edit in overlay.applied} == {"agreement"}
    assert overlay.orphaned == []
    split = _ledger(_engine("a", "cat"), _engine("b", "cot"))
    _overlay, letters = _resolved(split)
    middle = [span for span in letters.blocks[0].spans if span.text == "a"]
    assert middle and middle[0].provenance == "primary"
    assert letters.blocks[0].text == "cat"
    assert not any(span.provenance == "unresolved" for span in letters.blocks[0].spans)


def test_a_reviewer_edit_overrides_a_model_edit_on_the_same_span():
    ledger = _ledger(_engine("a", _TEXT))
    block = ledger.blocks[0]
    model = _edit(block, "model", "9", 13, 14)
    reviewer = _edit(block, "reviewer", "8", 13, 14)
    overlay, page = _resolved(ledger, model, reviewer)
    assert page.blocks[0].text == "The value is 8.05."
    applied = {edit.author for edit in overlay.applied}
    assert "reviewer" in applied and "model" not in applied
    assert overlay.orphaned == []
    alone, _page = _resolved(ledger, model)
    assert alone.applied[0].author == "model"


def test_an_overlay_edit_covers_an_unresolved_span():
    ledger = _ledger(_engine("a", _TEXT), _engine("b", "The value is O.05."))
    block = ledger.blocks[0]
    reviewer = _edit(block, "reviewer", "0", 13, 14)
    _overlay, page = _resolved(ledger, reviewer)
    assert page.blocks[0].text == _TEXT
    assert not any(span.provenance == "unresolved" for span in page.blocks[0].spans)


def test_a_second_engine_orphans_changed_blocks_and_keeps_the_rest():
    header = Proposed("running-header", 0, 0.0, 0.0, 180.0, 20.0, "HEADER", None)
    body = Proposed("body", 1, 0.0, 40.0, 180.0, 80.0, _TEXT, None)
    changed = Proposed("body", 1, 0.0, 40.0, 180.0, 80.0, "The value is 9.05.", None)
    first = _ledger(EngineRead("alpha", "m", "original", "ocr", (header, body)))
    edits = (
        _edit(first.blocks[0], "reviewer", "HEADER-OK"),
        _edit(first.blocks[1], "reviewer", "BODY-OK"),
    )
    kept, _page = _resolved(first, *edits, engine="alpha")
    assert {edit.replacement for edit in kept.applied if edit.author == "reviewer"} == {
        "HEADER-OK",
        "BODY-OK",
    }
    second = _ledger(
        EngineRead("alpha", "m", "original", "ocr", (header,)),
        EngineRead("beta", "m", "original", "ocr", (changed,)),
    )
    overlay, page = _resolved(second, *edits, engine="alpha")
    texts = {block.block_id: block.text for block in page.blocks}
    assert "HEADER-OK" in texts["p0001-b0000"]
    assert "BODY-OK" not in texts["p0001-b0001"]
    orphan = overlay.orphaned
    assert len(orphan) == 1 and orphan[0].requeued is True
    assert orphan[0].reason == "block_changed" and orphan[0].edit.block_id == "p0001-b0001"
    assert all(item.edit.block_id != "p0001-b0000" for item in orphan)


def test_an_edit_does_not_follow_a_block_whose_id_moved():
    header = Proposed("running-header", 0, 0.0, 30.0, 180.0, 50.0, "HEADER", None)
    first = _ledger(EngineRead("alpha", "m", "original", "ocr", (header,)))
    edit = _edit(first.blocks[0], "reviewer", "HEADER-OK")
    inserted = Proposed("heading", 0, 0.0, 0.0, 180.0, 15.0, "NEW", None)
    second = _ledger(EngineRead("alpha", "m", "original", "ocr", (inserted, header)))
    assert second.blocks[1].block_id == "p0001-b0001"
    assert block_anchor(second.blocks[1]) == edit.anchor
    overlay, page = _resolved(second, edit, engine="alpha")
    assert "HEADER-OK" not in "".join(block.text for block in page.blocks)
    assert overlay.orphaned[0].reason == "block_changed" and overlay.orphaned[0].requeued is True
    assert all(item.edit_id != edit.edit_id for item in overlay.applied)


def test_a_missing_block_is_requeued_and_a_raw_mismatch_is_not_applied():
    ledger = _ledger(_engine("a", _TEXT))
    missing = _edit(ledger.blocks[0], "reviewer", "GONE")
    missing = missing.model_copy(update={"block_id": "p0001-b9999"})
    mismatched = _edit(ledger.blocks[0], "model", "9", 13, 14).model_copy(
        update={"raw_text": "O", "edit_id": "bad-raw"}
    )
    overlay, page = _resolved(ledger, missing, mismatched)
    assert {item.reason for item in overlay.orphaned} == {"block_missing", "raw_mismatch"}
    assert all(item.requeued is True for item in overlay.orphaned)
    assert "9" not in page.blocks[0].text and "GONE" not in page.blocks[0].text


def test_page_class_selects_the_primary_reading():
    ledger = _ledger(_engine("alpha", "ALPHA"), _engine("beta", "BETA"))
    alpha, _beta = resolve_page(ledger, _choice("alpha", "normal"), [])
    beta, _other = resolve_page(ledger, _choice("beta", "dense_formulas"), [])
    del alpha
    assert _beta.blocks[0].text == "ALPHA" and _beta.blocks[0].fallback is False
    assert _other.blocks[0].text == "BETA"
    fallback, _page = resolve_page(ledger, _choice("missing"), [])
    del fallback
    assert _page.blocks[0].fallback is True


def test_the_class_file_chooses_the_reading_in_the_cache_key(tmp_path: Path):
    from llb.prep.book.classes.records import write_classes
    from tests.llb.prep.book.test_page_classes import _page

    params = {
        "primary": {"stage": "ocr", "engine": "alpha"},
        "by_page_class": {"dense_formulas": {"stage": "formula", "engine": "beta"}},
    }
    write_classes(tmp_path / "classes.json", _page(1, ["dense_formulas"]))
    chosen = page_choice(params, {"classes": tmp_path})
    assert chosen["engine"] == "beta" and chosen["page_class"] == "dense_formulas"
    plain = page_choice(params, {})
    assert plain["engine"] == "alpha" and plain["page_class"] == ""


def test_reviewer_digest_ignores_other_pages(tmp_path: Path):
    assert reviewer_digest(tmp_path, "p0001") == reviewer_digest(tmp_path, "p0002")
    empty = reviewer_digest(tmp_path, "p0002")
    ledger = _ledger(_engine("a", "HEADER"))
    edit = _edit(ledger.blocks[0], "reviewer", "OK")
    write_decisions(
        tmp_path / "review" / "decisions.json",
        ReviewerDecisions(
            schema_id="llb.book-ocr-reviewer-decisions",
            schema_version="1.0.0",
            kind="book-ocr-reviewer-decisions",
            edits=[edit],
        ),
    )
    assert reviewer_digest(tmp_path, "p0001") != empty
    assert reviewer_digest(tmp_path, "p0002") == empty
    assert reviewer_digest(tmp_path, "p0002") == digest_json([])


def test_resolved_page_round_trips_its_schema():
    ledger = _ledger(_engine("a", _TEXT), _engine("b", _TEXT))
    _overlay, page = _resolved(ledger)
    payload = page.model_dump(mode="json")
    assert DEFAULT_REGISTRY.read_current(payload).model_dump(mode="json") == payload
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    validator_for(schema)(schema).validate(payload)
    broken = json.loads(page.model_dump_json())
    broken["blocks"][0]["spans"] = []
    broken["blocks"][0]["text"] = "nope"
    with pytest.raises(ValidationError):
        type(page).model_validate(broken)
