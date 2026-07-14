"""Tests for ingest squad sources."""

import sys
from types import SimpleNamespace
import llb.prep.ingest_squad as ingest_squad_module
from llb.goldset.schema import load_goldset
from llb.prep.ingest_squad import load_hf, main, squad_to_gold
from llb.prep.ua_squad_source import DATASET_ID, DATASET_REVISION, DATASET_SPLIT
from test_ingest_squad import _reviewed_squad_record


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
    from llb.prep.squad_records import normalize

    records = normalize(nested)
    docs, items, skipped = squad_to_gold(records)
    assert len(items) == 1 and items[0].reference_answer == "Київ"


def test_coerce_answers_dict_string():
    from llb.prep.squad_records import coerce_answers

    # HPLT/ua-squad serializes answers as a Python-repr string
    a = coerce_answers({"answers": "{'answer_start': [5], 'text': ['abc']}"})
    assert a == {"text": ["abc"], "answer_start": [5]}


def test_coerce_answers_flat_columns():
    from llb.prep.squad_records import coerce_answers

    a = coerce_answers({"answer_text": "abc", "answer_start": "5.0"})
    assert a["text"] == ["abc"] and a["answer_start"] == [5]
