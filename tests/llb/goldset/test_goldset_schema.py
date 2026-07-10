import pytest

from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset, load_goldset


def _item(**overrides):
    base = {
        "id": "x1",
        "lang": "uk",
        "question": "q",
        "reference_answer": "a",
        "source_doc_id": "d.txt",
        "source_spans": [{"doc_id": "d.txt", "char_start": 0, "char_end": 3, "text": "abc"}],
        "provenance": "human-authored",
        "verified": True,
        "split": "final",
    }
    base.update(overrides)
    return base


def test_valid_item():
    assert GoldItem.model_validate(_item()).split == "final"


def test_bad_provenance_rejected():
    with pytest.raises(Exception):
        GoldItem.model_validate(_item(provenance="nope"))


def test_bad_split_rejected():
    with pytest.raises(Exception):
        GoldItem.model_validate(_item(split="train"))


def test_span_length_mismatch_rejected():
    with pytest.raises(Exception):
        SourceSpan(doc_id="d", char_start=0, char_end=2, text="abc")


def test_roundtrip(tmp_path):
    path = tmp_path / "g.jsonl"
    dump_goldset([GoldItem.model_validate(_item())], path)
    loaded = load_goldset(path)
    assert len(loaded) == 1 and loaded[0].id == "x1"
