"""Draft-time ambiguous-evidence guard: occurrence counting, worksheet column, and review card."""

import json

from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset
from llb.goldset.span_occurrences import (
    OCCURRENCES_SIDECAR,
    SPAN_OCCURRENCES_COL,
    count_span_occurrences,
    load_occurrences_sidecar,
    span_occurrence_counts,
    write_occurrences_sidecar,
)
from llb.goldset.verify_base import WORKSHEET_COLS, load_worksheet
from llb.goldset.verify_card import format_card
from llb.goldset.verify_sampling.worksheet import build_sample_worksheet
from tests.llb.goldset._verify_helpers import DOC, TEXT, _bundle, _item

# A second document that repeats the "1871" span verbatim, so that span is ambiguous by
# construction: its text now exists in two places in the corpus.
DOC2 = "squad/doc2.txt"
TEXT2 = "У 1871 році сталася інша подія, не пов'язана з першою."


def _dup_item(item_id, *, answer, doc, split="calibration"):
    start = (TEXT if doc == DOC else TEXT2).find(answer)
    return GoldItem(
        id=item_id,
        question=f"Коли {item_id}?",
        reference_answer=answer,
        source_doc_id=doc,
        source_spans=[
            SourceSpan(doc_id=doc, char_start=start, char_end=start + len(answer), text=answer)
        ],
        provenance="frontier-drafted",
        split=split,
    )


def _repeated_block_bundle(tmp_path, items):
    """A bundle whose corpus repeats the '1871' block across two documents."""
    dump_goldset(items, tmp_path / "goldset.jsonl")
    (tmp_path / "corpus" / DOC).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "corpus" / DOC).write_text(TEXT + "\n", encoding="utf-8")
    (tmp_path / "corpus" / DOC2).write_text(TEXT2 + "\n", encoding="utf-8")
    return tmp_path


def test_count_span_occurrences_sums_across_documents():
    assert count_span_occurrences([TEXT, TEXT2], "1871") == 2
    assert count_span_occurrences([TEXT, TEXT2], "поетесою") == 1
    assert count_span_occurrences([TEXT, TEXT2], "") == 0


def test_span_occurrence_counts_uses_primary_span():
    items = [_item("a")]  # answer "1871", present in TEXT once
    assert span_occurrence_counts(items, {DOC: TEXT}) == {"a": 1}
    assert span_occurrence_counts(items, {DOC: TEXT, DOC2: TEXT2}) == {"a": 2}


def test_unique_span_keeps_worksheet_byte_for_byte(tmp_path):
    """A corpus with no repeated spans writes no occurrence column -- the sheet is unchanged."""
    bundle = _bundle(tmp_path, [_item("a"), _item("b")])
    out = tmp_path / "ws.csv"
    build_sample_worksheet(bundle, out, n=2, seed=1)
    rows, fields = load_worksheet(out)
    assert SPAN_OCCURRENCES_COL not in fields
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert "span_occurrences" not in header
    assert not (bundle / OCCURRENCES_SIDECAR).exists()  # nothing flagged, no sidecar


def test_repeated_span_shows_count_on_worksheet(tmp_path):
    bundle = _repeated_block_bundle(
        tmp_path,
        [_dup_item("dup", answer="1871", doc=DOC), _dup_item("uniq", answer="поетесою", doc=DOC)],
    )
    out = tmp_path / "ws.csv"
    build_sample_worksheet(bundle, out, n=2, seed=1)
    rows, fields = load_worksheet(out)
    assert SPAN_OCCURRENCES_COL in fields
    by_id = {row["item_id"]: row for row in rows}
    assert by_id["dup"][SPAN_OCCURRENCES_COL] == "2"  # ambiguous: two places in the corpus
    assert by_id["uniq"][SPAN_OCCURRENCES_COL] == ""  # unique span stays blank


def test_sidecar_round_trip_and_threshold(tmp_path):
    counts = {"dup": 3, "uniq": 1}
    written = write_occurrences_sidecar(tmp_path, counts)
    assert written == 1  # only the flagged item
    loaded = load_occurrences_sidecar(tmp_path)
    assert loaded == {"dup": 3}
    # An all-unique count set writes no file.
    (tmp_path / OCCURRENCES_SIDECAR).unlink()
    assert write_occurrences_sidecar(tmp_path, {"uniq": 1}) == 0
    assert not (tmp_path / OCCURRENCES_SIDECAR).exists()


def test_worksheet_prefers_draft_time_sidecar(tmp_path):
    """A committed sidecar count is surfaced verbatim, without a corpus rescan."""
    bundle = _bundle(tmp_path, [_item("a")])  # corpus has "1871" once -> a scan would say 1
    (bundle / OCCURRENCES_SIDECAR).write_text(
        json.dumps({"id": "a", SPAN_OCCURRENCES_COL: 4}) + "\n", encoding="utf-8"
    )
    out = tmp_path / "ws.csv"
    build_sample_worksheet(bundle, out, n=1, seed=1)
    rows, fields = load_worksheet(out)
    assert SPAN_OCCURRENCES_COL in fields
    assert rows[0][SPAN_OCCURRENCES_COL] == "4"


def test_review_card_flags_ambiguous_evidence():
    flagged = {col: "" for col in [*WORKSHEET_COLS, SPAN_OCCURRENCES_COL]}
    flagged.update(
        {"item_id": "dup", "question": "q", "span_doc_id": DOC, SPAN_OCCURRENCES_COL: "2"}
    )
    card = format_card(flagged, 1, 1, 0)
    assert "appears in 2 places" in card
    unique = {**flagged, SPAN_OCCURRENCES_COL: "1"}
    assert "appears in" not in format_card(unique, 1, 1, 0)
