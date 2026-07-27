"""Focused tests split from ``test_pdf_repeats.py``."""

import json
from pathlib import Path

import pytest
from _pdf_repeats_helpers import (
    FIXTURE,
    FIXTURE_ANCHOR_CHARS,
    FIXTURE_CHARS,
    FIXTURE_CROSS_GROUPS,
    FIXTURE_DROP_CHARS,
    FIXTURE_GROUPS,
    FIXTURE_INTRA_GROUPS,
    PROCEDURE,
    REPEATED_DOC,
    _corpus_docs,
    _item,
    fixture_text,
)

from llb.goldset.schema import load_goldset
from llb.prep.pdf.repeat_corpus import (
    REPEAT_REPORT_NAME,
    strip_corpus_repeats,
)
from llb.prep.pdf.repeats import (
    REPEAT_ANCHOR,
    REPEAT_DROP,
    REPEAT_KEEP,
)
from llb.rag.chunking.corpus import chunk_corpus
from llb.rag.duplicates import duplicate_stats


def test_corpus_census_counts_only_repeats_inside_one_document() -> None:
    report = strip_corpus_repeats(FIXTURE, mode=REPEAT_KEEP)

    per_doc = {document["doc_id"]: document for document in report["documents"]}
    assert per_doc[REPEATED_DOC]["census"]["groups"] == FIXTURE_GROUPS
    # the support block is shared with the second document, but appears there ONCE
    assert per_doc["dovidka-oblik.md"]["census"]["groups"] == 0
    assert report["chars_before"] == report["chars_after"] == FIXTURE_CHARS


def test_chunk_census_splits_intra_from_cross_document_groups() -> None:
    stats = duplicate_stats(chunk_corpus(FIXTURE, "sentence", 200, 30))

    assert stats["intra_document_groups"] == FIXTURE_INTRA_GROUPS
    assert stats["cross_document_groups"] == FIXTURE_CROSS_GROUPS
    assert stats["groups"] == FIXTURE_INTRA_GROUPS + FIXTURE_CROSS_GROUPS


@pytest.mark.parametrize(
    ("mode", "chars"),
    [(REPEAT_DROP, FIXTURE_DROP_CHARS), (REPEAT_ANCHOR, FIXTURE_ANCHOR_CHARS)],
)
def test_strip_corpus_repeats_rewrites_into_a_new_root(
    tmp_path: Path, mode: str, chars: int
) -> None:
    out = tmp_path / mode

    report = strip_corpus_repeats(FIXTURE, out, mode=mode)

    assert report["chars_before"] == FIXTURE_CHARS
    assert report["chars_after"] == chars
    assert sum(len((out / doc).read_text(encoding="utf-8")) for doc in _corpus_docs(out)) == chars
    assert fixture_text() == (FIXTURE / REPEATED_DOC).read_text(encoding="utf-8")  # never in place
    assert json.loads((out / REPEAT_REPORT_NAME).read_text(encoding="utf-8"))["mode"] == mode


def test_strip_corpus_repeats_remaps_a_goldset_onto_the_stripped_corpus(tmp_path: Path) -> None:
    text = fixture_text()
    second = text.index(PROCEDURE, text.index(PROCEDURE) + 1)
    unique = text.index("Списання майна виконується комісією")
    goldset = tmp_path / "goldset.jsonl"
    goldset.write_text(
        "\n".join(
            item.model_dump_json()
            for item in (
                _item("repeat", text, second, len(PROCEDURE)),
                _item("unique", text, unique, len("Списання майна виконується комісією")),
            )
        ),
        encoding="utf-8",
    )
    out = tmp_path / "drop"

    report = strip_corpus_repeats(
        FIXTURE, out, mode=REPEAT_DROP, goldset=goldset, goldset_out=out / "goldset.jsonl"
    )

    # the "repeat" item's evidence sat on a dropped copy -> re-homed onto the survivor; "unique"
    # is untouched.
    assert report["goldset"] == {
        "items": 2,
        "remapped": 2,
        "dropped": [],
        "rehomed": ["repeat"],
    }
    stripped = (out / REPEATED_DOC).read_text(encoding="utf-8")
    for item in load_goldset(out / "goldset.jsonl"):
        span = item.source_spans[0]
        assert stripped[span.char_start : span.char_end] == span.text


def test_recover_straddle_keeps_a_boundary_crossing_item_as_two_spans(tmp_path: Path) -> None:
    text = fixture_text()
    second = text.index(PROCEDURE, text.index(PROCEDURE) + 1)
    start, end = second - 40, second + len(PROCEDURE)  # kept tail + dropped procedure head
    goldset = tmp_path / "goldset.jsonl"
    goldset.write_text(
        _item("straddle", text, start, end - start).model_dump_json(), encoding="utf-8"
    )

    dropped = strip_corpus_repeats(
        FIXTURE, tmp_path / "a", mode=REPEAT_DROP, goldset=goldset, goldset_out=tmp_path / "a.jsonl"
    )
    recovered = strip_corpus_repeats(
        FIXTURE,
        tmp_path / "b",
        mode=REPEAT_DROP,
        goldset=goldset,
        goldset_out=tmp_path / "b.jsonl",
        recover_straddle=True,
    )

    # without recovery the straddler is dropped; with it, the item survives as two re-anchored spans
    assert dropped["goldset"] == {"items": 1, "remapped": 0, "dropped": ["straddle"], "rehomed": []}
    assert recovered["goldset"] == {
        "items": 1,
        "remapped": 1,
        "dropped": [],
        "rehomed": ["straddle"],
    }
    stripped = (tmp_path / "b" / REPEATED_DOC).read_text(encoding="utf-8")
    (item,) = load_goldset(tmp_path / "b.jsonl")
    assert len(item.source_spans) == 2
    for span in item.source_spans:
        assert stripped[span.char_start : span.char_end] == span.text
    assert "".join(span.text for span in item.source_spans) == text[start:end]
