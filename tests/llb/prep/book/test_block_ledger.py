"""The block ledger contract: kinds, ids, validation, and disagreement spans."""

import json
from pathlib import Path

import pytest
from jsonschema.validators import validator_for
from pydantic import ValidationError

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.core.contracts.data_prep.book_blocks import (
    BLOCK_KINDS,
    Block,
    PageBlockLedger,
    RawReading,
    Region,
)
from llb.core.paths import PROJECT_ROOT
from llb.prep.book.engines.protocol import EngineRead, Proposed
from llb.prep.book.ledger.assemble import assemble
from llb.prep.book.ledger.reconcile import reconcile

_SCHEMA = PROJECT_ROOT / "schemas" / "artifacts" / "llb.book-ocr-block-ledger" / "1.0.0.schema.json"
_WIDTH = 200
_HEIGHT = 400


def _reading(engine: str, text: str, variant: str = "original") -> RawReading:
    return RawReading(engine=engine, variant=variant, text=text, model="m")


def _read(engine: str, text: str, variant: str = "original", order: int = 0) -> EngineRead:
    block = Proposed("body", order, 0.0, 0.0, float(_WIDTH), 40.0, text, None)
    return EngineRead(engine, "m", variant, "ocr", (block,))


def _ledger(*reads: EngineRead) -> PageBlockLedger:
    return assemble("p0001", 1, _WIDTH, _HEIGHT, reads)


def _texts(span) -> set[str]:
    return {item.text for item in span.slices}


def test_every_kind_round_trips_and_a_block_needs_a_page_and_a_region():
    region = Region(x0=0.0, y0=0.0, x1=10.0, y1=10.0, width=_WIDTH, height=_HEIGHT)
    blocks = [
        Block(
            block_id=f"p0001-b{index:04d}",
            pdf_page=1,
            ordinal=index,
            kind=kind,
            region=region,
            reading_order=index,
            readings=[_reading("fake", kind)],
        )
        for index, kind in enumerate(BLOCK_KINDS)
    ]
    ledger = PageBlockLedger(
        schema_id="llb.book-ocr-block-ledger",
        schema_version="1.0.0",
        kind="book-ocr-block-ledger",
        page_id="p0001",
        pdf_page=1,
        blocks=blocks,
    )
    again = PageBlockLedger.model_validate_json(ledger.model_dump_json())
    assert [block.kind for block in again.blocks] == list(BLOCK_KINDS)
    payload = ledger.model_dump(mode="json")
    assert DEFAULT_REGISTRY.read_current(payload).model_dump(mode="json") == payload
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    validator_for(schema)(schema).validate(payload)
    broken = json.loads(ledger.model_dump_json())
    del broken["blocks"][0]["pdf_page"]
    del broken["blocks"][0]["region"]
    with pytest.raises(ValidationError):
        PageBlockLedger.model_validate(broken)
    with pytest.raises(Exception):
        validator_for(schema)(schema).validate(broken)
    assert "text" not in Block.model_fields


def test_digit_sign_decimal_and_symbol_disagreements_are_spans_not_a_choice():
    digit = reconcile([_reading("a", "0.05"), _reading("b", "O.05")])
    assert len(digit) == 1 and digit[0].classes == ["digit"] and _texts(digit[0]) == {"0", "O"}
    sign = reconcile([_reading("a", "-"), _reading("b", "\u2014")])
    assert sign[0].classes == ["sign"] and _texts(sign[0]) == {"-", "\u2014"}
    decimal = reconcile([_reading("a", "1.5"), _reading("b", "1,5")])
    assert decimal[0].classes == ["decimal"] and _texts(decimal[0]) == {".", ","}
    symbol = reconcile([_reading("a", "\u03b1"), _reading("b", "a")])
    assert symbol[0].classes == ["symbol"] and _texts(symbol[0]) == {"\u03b1", "a"}
    assert reconcile([_reading("a", "cat"), _reading("b", "cot")]) == []


def test_a_majority_does_not_settle_a_digit(monkeypatch):
    monkeypatch.setattr("time.time", lambda: 123.0)
    ledger = _ledger(
        _read("a", "The value is 0.05."),
        _read("b", "The value is 0.05."),
        _read("c", "The value is O.05."),
    )
    spans = ledger.blocks[0].uncertainty
    assert spans and any(_texts(span) >= {"0", "O"} for span in spans)
    assert {item.text for item in ledger.blocks[0].readings} == {
        "The value is 0.05.",
        "The value is O.05.",
    }


def test_block_ids_follow_geometry_not_the_order_engines_finished(monkeypatch):
    monkeypatch.setattr("time.time", lambda: 1.0)
    top = Proposed("heading", 0, 0.0, 0.0, 100.0, 20.0, "Title", None)
    bottom = Proposed("body", 1, 0.0, 40.0, 100.0, 80.0, "Body", None)
    forward = EngineRead("a", "m", "original", "ocr", (top, bottom))
    backward = EngineRead("a", "m", "original", "ocr", (bottom, top))
    first = assemble("p0007", 7, _WIDTH, _HEIGHT, [forward])
    second = assemble("p0007", 7, _WIDTH, _HEIGHT, [backward])
    assert [block.block_id for block in first.blocks] == ["p0007-b0000", "p0007-b0001"]
    assert [block.model_dump() for block in first.blocks] == [
        block.model_dump() for block in second.blocks
    ]
    joined = assemble(
        "p0001",
        1,
        _WIDTH,
        _HEIGHT,
        [
            EngineRead(
                "a",
                "m",
                "original",
                "ocr",
                (
                    Proposed("body", 1, 0.0, 0.0, 100.0, 40.0, "second", None),
                    Proposed("body", 0, 0.0, 0.0, 100.0, 40.0, "first", None),
                ),
            )
        ],
    )
    assert joined.blocks[0].readings[0].text == "first\nsecond"


def test_same_engine_neighbours_and_a_box_inside_another_stay_separate():
    upper = Proposed("body", 0, 0.0, 0.0, 100.0, 20.0, "one", None)
    lower = Proposed("body", 1, 0.0, 10.0, 100.0, 30.0, "two", None)
    banner = Proposed("body", 0, 0.0, 0.0, 200.0, 40.0, "TITLE", None)
    inner = Proposed("body", 1, 10.0, 10.0, 30.0, 30.0, "98", None)
    neighbours = assemble(
        "p0001", 1, _WIDTH, _HEIGHT, [EngineRead("a", "m", "original", "ocr", (upper, lower))]
    )
    contained = assemble(
        "p0001", 1, _WIDTH, _HEIGHT, [EngineRead("a", "m", "original", "ocr", (banner, inner))]
    )
    assert [block.readings[0].text for block in neighbours.blocks] == ["one", "two"]
    assert [block.readings[0].text for block in contained.blocks] == ["TITLE", "98"]


def test_a_line_inside_another_engines_block_keeps_both_readings():
    line = Proposed("body", 0, 10.0, 10.0, 80.0, 30.0, "0.05", None)
    region = Proposed("body", 0, 0.0, 0.0, 100.0, 40.0, "the value is 0.05", None)
    ledger = assemble(
        "p0001",
        1,
        _WIDTH,
        _HEIGHT,
        [
            EngineRead("rapidocr", "m", "original", "ocr", (line,)),
            EngineRead("docling", "m", "original", "layout", (region,)),
        ],
    )
    assert len(ledger.blocks) == 1
    readings = {reading.engine: reading.text for reading in ledger.blocks[0].readings}
    assert readings == {"rapidocr": "0.05", "docling": "the value is 0.05"}


def test_schema_file_is_the_generated_contract():
    assert _SCHEMA.is_file()
    assert Path(_SCHEMA).name == "1.0.0.schema.json"
