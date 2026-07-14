import json

from llb.core.paths import PROJECT_ROOT
from llb.goldset.schema import GoldItem, dump_goldset, load_goldset
from llb.goldset.validate import validate_items
from llb.prep.ingest_squad import load_squad_json, main, squad_to_gold, write_corpus
from llb.prep.verified_ledger import DEFAULT_VERIFIED_GOLDSET

VERIFIED_CORPUS = DEFAULT_VERIFIED_GOLDSET.parent / "corpus"


def _reviewed_squad_record() -> tuple[dict[str, object], GoldItem]:
    reviewed = load_goldset(DEFAULT_VERIFIED_GOLDSET)[0]
    context = (
        (VERIFIED_CORPUS / reviewed.source_doc_id).read_text(encoding="utf-8").removesuffix("\n")
    )
    span = reviewed.source_spans[0]
    return (
        {
            "id": reviewed.id,
            "context": context,
            "question": reviewed.question,
            "answers": {"text": [span.text], "answer_start": [span.char_start]},
        },
        reviewed,
    )


def test_ingest_fixture(tmp_path):
    records = load_squad_json(PROJECT_ROOT / "samples" / "data-prep" / "squad_uk_fixture.json")
    docs, items, skipped = squad_to_gold(records)
    assert len(items) == 4 and skipped == 0

    corpus = tmp_path / "corpus"
    write_corpus(docs, corpus)
    assert validate_items(items, corpus)["errors"] == []  # spans resolve to labels
    assert all(it.provenance == "public-reused" and it.verified is False for it in items)


def test_ingest_can_mark_a_pinned_published_fixture_verified():
    records = load_squad_json(PROJECT_ROOT / "samples" / "data-prep" / "squad_uk_fixture.json")
    _docs, items, _skipped = squad_to_gold(records, verified=True)
    assert items and all(item.provenance == "public-reused" and item.verified for item in items)


def test_main_adopts_matching_human_verified_item_and_corpus(tmp_path):
    record, reviewed = _reviewed_squad_record()
    source = tmp_path / "source.json"
    source.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")
    out_dir = tmp_path / "out"

    assert main(["--squad-json", str(source), "--out-dir", str(out_dir)]) == 0

    generated = load_goldset(out_dir / "goldset" / "squad_uk.jsonl")
    assert generated == [reviewed]
    doc_id = generated[0].source_doc_id
    assert (out_dir / "corpus" / doc_id).read_bytes() == (VERIFIED_CORPUS / doc_id).read_bytes()


def test_main_can_disable_default_verification_ledger(tmp_path):
    record, reviewed = _reviewed_squad_record()
    source = tmp_path / "source.json"
    source.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")
    out_dir = tmp_path / "out"

    assert (
        main(
            [
                "--squad-json",
                str(source),
                "--out-dir",
                str(out_dir),
                "--no-verification-ledger",
            ]
        )
        == 0
    )

    generated = load_goldset(out_dir / "goldset" / "squad_uk.jsonl")
    assert generated[0].id == reviewed.id
    assert generated[0].verified is False


def test_main_accepts_custom_reviewed_draft_ledger(tmp_path):
    record, reviewed = _reviewed_squad_record()
    custom_item = reviewed.model_copy(
        update={"id": "custom-draft-0", "provenance": "human-verified"}
    )
    record["id"] = custom_item.id
    source = tmp_path / "source.json"
    source.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")

    ledger_root = tmp_path / "reviewed-m3.5"
    dump_goldset([custom_item], ledger_root / "reviewed.jsonl")
    ledger_doc = ledger_root / "corpus" / custom_item.source_doc_id
    ledger_doc.parent.mkdir(parents=True)
    ledger_doc.write_bytes((VERIFIED_CORPUS / custom_item.source_doc_id).read_bytes())
    out_dir = tmp_path / "out"

    assert (
        main(
            [
                "--squad-json",
                str(source),
                "--out-dir",
                str(out_dir),
                "--verified-goldset",
                str(ledger_root / "reviewed.jsonl"),
            ]
        )
        == 0
    )

    assert load_goldset(out_dir / "goldset" / "squad_uk.jsonl") == [custom_item]
