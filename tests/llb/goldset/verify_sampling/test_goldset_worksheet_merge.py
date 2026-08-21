"""Tests for goldset worksheet merge."""

import json
from llb.goldset.verify_base import load_worksheet, write_worksheet_rows
from llb.goldset.verify_sampling.confidence import confidence_order, row_confidence
from llb.goldset.verify_sampling.context import (
    load_retrieval_ranks,
)
from llb.goldset.verify_sampling.worksheet import (
    build_sample_worksheet,
    merge_sample_worksheet,
)
from llb.goldset.verify_card import format_card
from tests.llb.goldset._verify_helpers import (
    TEXT,
    _bundle,
    _item,
    _ws_row,
)


def test_worksheet_carries_retrieval_rank_and_page_citation(tmp_path):
    bundle = _bundle(tmp_path, [_item("a")])
    (bundle / "needle_items.jsonl").write_text(
        json.dumps({"id": "a", "retrieval_rank": 2}) + "\n", encoding="utf-8"
    )
    sidecar = bundle / "corpus" / "squad" / "doc1.citations.json"
    sidecar.write_text(
        json.dumps(
            {
                "source": "orig/doc1.pdf",
                "pages": [{"page": 3, "char_start": 0, "char_end": len(TEXT)}],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "ws.csv"
    build_sample_worksheet(bundle, out, n=1)
    rows, _ = load_worksheet(out)
    assert rows[0]["retrieval_rank"] == "2"
    assert rows[0]["page_citation"] == "doc1.pdf p.3"
    # The card renders both so the reviewer sees them without leaving the terminal.
    card = format_card(rows[0], 1, 1, 0)
    assert "doc1.pdf p.3" in card and "retrieval_rank=2" in card


def test_load_retrieval_ranks_reads_both_sidecars(tmp_path):
    (tmp_path / "needle_items.jsonl").write_text(
        json.dumps({"id": "a", "retrieval_rank": 1}) + "\n", encoding="utf-8"
    )
    (tmp_path / "item_provenance.jsonl").write_text(
        json.dumps({"id": "b", "retrieval_rank": 4})
        + "\n"
        + json.dumps({"id": "c", "retrieval_rank": None})
        + "\n",
        encoding="utf-8",
    )
    ranks = load_retrieval_ranks(tmp_path)
    assert ranks == {"a": 1, "b": 4}  # a null rank (retrieval miss) is simply absent


def test_confidence_order_puts_least_confident_first():
    good = _ws_row("good", cc_grounded="true", cc_supported="true", retrieval_rank="1")
    bad = _ws_row("bad", cc_grounded="false")
    mid = _ws_row("mid")
    assert row_confidence(good) > row_confidence(mid) > row_confidence(bad)
    assert confidence_order([good, bad, mid]) == [1, 2, 0]


def test_merge_adds_only_new_rows_and_preserves_decided_bytes(tmp_path):
    bundle = _bundle(tmp_path, [_item(f"i{k}") for k in range(6)])
    out = tmp_path / "verify_sample.csv"
    build_sample_worksheet(bundle, out, n=2, seed=1)
    rows, fields = load_worksheet(out)
    rows[0]["decision"] = "accept"
    rows[0]["human_note"] = "ok, з комою"  # the comma forces CSV quoting on this row
    write_worksheet_rows(out, rows, fields)
    before_lines = out.read_bytes().splitlines(keepends=True)
    decided_id = rows[0]["item_id"]

    added, total = merge_sample_worksheet(bundle, out, n=5, seed=1)
    assert added == 3 and total == 5  # same-seed draw is a superset; only new ids appended

    after_rows, _ = load_worksheet(out)
    ids = [r["item_id"] for r in after_rows]
    assert len(set(ids)) == len(ids)  # a decided row is never re-drawn
    assert ids[:2] == [r["item_id"] for r in rows]  # existing rows keep their order
    after_lines = out.read_bytes().splitlines(keepends=True)
    assert after_lines[: len(before_lines)] == before_lines  # decided rows byte-for-byte
    assert next(r for r in after_rows if r["item_id"] == decided_id)["decision"] == "accept"
    manifest = json.loads((out.with_name("sample_manifest.json")).read_text(encoding="utf-8"))
    assert manifest["merged_added"] == 3 and manifest["sample_size"] == 5


def test_merge_is_idempotent(tmp_path):
    bundle = _bundle(tmp_path, [_item(f"i{k}") for k in range(6)])
    out = tmp_path / "verify_sample.csv"
    build_sample_worksheet(bundle, out, n=2, seed=1)
    merge_sample_worksheet(bundle, out, n=5, seed=1)
    snapshot = out.read_bytes()
    added, total = merge_sample_worksheet(bundle, out, n=5, seed=1)
    assert added == 0 and total == 5
    assert out.read_bytes() == snapshot


def test_merge_falls_back_to_fresh_build(tmp_path):
    bundle = _bundle(tmp_path, [_item("a"), _item("b")])
    out = tmp_path / "ws.csv"
    added, total = merge_sample_worksheet(bundle, out, n=2, seed=1)
    assert added == 2 and total == 2
    assert out.is_file()
