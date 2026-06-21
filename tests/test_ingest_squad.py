from pathlib import Path

from llb.goldset.validate import validate_items
from llb.prep.ingest_squad import load_squad_json, squad_to_gold, write_corpus

REPO = Path(__file__).resolve().parents[1]


def test_ingest_fixture(tmp_path):
    records = load_squad_json(REPO / "samples" / "squad_uk_fixture.json")
    docs, items, skipped = squad_to_gold(records)
    assert len(items) == 4 and skipped == 0

    corpus = tmp_path / "corpus"
    write_corpus(docs, corpus)
    assert validate_items(items, corpus)["errors"] == []  # spans resolve to labels
    assert all(it.provenance == "public-reused" and it.verified is False for it in items)


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
