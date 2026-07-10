import json
import sys
from types import SimpleNamespace

import llb.prep.ingest_squad as ingest_squad_module
from llb.core.paths import PROJECT_ROOT
from llb.goldset.schema import GoldItem, dump_goldset, load_goldset
from llb.goldset.validate import validate_items
from llb.prep.ingest_squad import load_hf, load_squad_json, main, squad_to_gold, write_corpus
from llb.prep.ua_squad_source import DATASET_ID, DATASET_REVISION, DATASET_SPLIT
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


def test_load_hf_forwards_revision_and_selects_nested_contexts(monkeypatch):
    captured = {}
    rows = [
        {
            "version": "v1.0",
            "data": {
                "title": "fixture",
                "paragraphs": [
                    {
                        "context": "alpha beta",
                        "qas": [
                            {
                                "id": "a",
                                "question": "q1",
                                "answers": [{"text": "alpha", "answer_start": 0}],
                            },
                            {
                                "id": "duplicate-context",
                                "question": "q2",
                                "answers": [{"text": "beta", "answer_start": 6}],
                            },
                        ],
                    },
                    {
                        "context": "gamma delta",
                        "qas": [
                            {
                                "id": "b",
                                "question": "q3",
                                "answers": [{"text": "delta", "answer_start": 6}],
                            }
                        ],
                    },
                ],
            },
        }
    ]

    def fake_load_dataset(dataset_id, **kwargs):
        captured["dataset_id"] = dataset_id
        captured.update(kwargs)
        return rows

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    records = load_hf(
        "source/dataset",
        "validation",
        token="test-token",
        limit=2,
        revision="pinned-revision",
        context_diverse=True,
    )

    assert [record["id"] for record in records] == ["a", "b"]
    assert captured == {
        "dataset_id": "source/dataset",
        "split": "validation",
        "token": "test-token",
        "streaming": True,
        "revision": "pinned-revision",
    }


def test_pinned_development_profile_generates_verified_items(monkeypatch, tmp_path):
    record, reviewed = _reviewed_squad_record()
    captured = {}

    def fake_load_hf(
        dataset_id, split, token=None, limit=None, revision=None, context_diverse=False
    ):
        captured.update(
            dataset_id=dataset_id,
            split=split,
            limit=limit,
            revision=revision,
            context_diverse=context_diverse,
        )
        return [record]

    monkeypatch.setattr(ingest_squad_module, "load_hf", fake_load_hf)
    out_dir = tmp_path / "out"

    assert (
        main(
            [
                "--pinned-development-source",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )

    assert captured == {
        "dataset_id": DATASET_ID,
        "split": DATASET_SPLIT,
        "limit": 250,
        "revision": DATASET_REVISION,
        "context_diverse": True,
    }
    assert load_goldset(out_dir / "goldset" / "squad_uk.jsonl") == [reviewed]


def test_skips_unanswerable():
    records = [
        {"id": "x", "context": "abc", "question": "q", "answers": {"text": [], "answer_start": []}}
    ]
    docs, items, skipped = squad_to_gold(records)
    assert items == [] and skipped == 1


def test_nested_squad_normalized():
    nested = {
        "data": [
            {
                "paragraphs": [
                    {
                        "context": "Київ — столиця.",
                        "qas": [
                            {
                                "id": "n1",
                                "question": "Що це?",
                                "answers": [{"text": "Київ", "answer_start": 0}],
                            }
                        ],
                    }
                ]
            }
        ]
    }
    from llb.prep.ingest_squad import normalize

    records = normalize(nested)
    docs, items, skipped = squad_to_gold(records)
    assert len(items) == 1 and items[0].reference_answer == "Київ"


def test_coerce_answers_dict_string():
    from llb.prep.ingest_squad import coerce_answers

    # HPLT/ua-squad serializes answers as a Python-repr string
    a = coerce_answers({"answers": "{'answer_start': [5], 'text': ['abc']}"})
    assert a == {"text": ["abc"], "answer_start": [5]}


def test_coerce_answers_flat_columns():
    from llb.prep.ingest_squad import coerce_answers

    a = coerce_answers({"answer_text": "abc", "answer_start": "5.0"})
    assert a["text"] == ["abc"] and a["answer_start"] == [5]
