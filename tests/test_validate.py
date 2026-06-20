from llb.goldset.schema import GoldItem
from llb.goldset.validate import validate_items


def _corpus(tmp_path, text="abcdef"):
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "d.txt").write_text(text, encoding="utf-8")
    return corpus


def _item(span_text="abc", start=0, end=3, item_id="x1"):
    return GoldItem.model_validate(
        {
            "id": item_id,
            "lang": "uk",
            "question": "q",
            "reference_answer": "a",
            "source_doc_id": "d.txt",
            "source_spans": [
                {"doc_id": "d.txt", "char_start": start, "char_end": end, "text": span_text}
            ],
            "provenance": "human-authored",
            "verified": True,
            "split": "final",
        }
    )


def test_pass(tmp_path):
    assert validate_items([_item()], _corpus(tmp_path))["errors"] == []


def test_span_mismatch(tmp_path):
    # text[1:4] == "bcd" != "abc"
    rep = validate_items([_item(span_text="abc", start=1, end=4)], _corpus(tmp_path))
    assert any("mismatch" in e for e in rep["errors"])


def test_missing_doc(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rep = validate_items([_item()], corpus)
    assert any("missing corpus doc" in e for e in rep["errors"])


def test_duplicate_id(tmp_path):
    rep = validate_items([_item(item_id="dup"), _item(item_id="dup")], _corpus(tmp_path))
    assert any("duplicate id" in e for e in rep["errors"])
