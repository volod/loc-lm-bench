"""Frontier prep utilities (M3.5): pure parsing/grounding/guards, fake LLM completions."""

import json

import pytest

from llb.prep.frontier import (
    build_drafted_items,
    parse_json_block,
    prepare_goldset,
    prepare_synthetic_corpus,
)

DOC = "Київ є столицею України. Дніпро тече через місто."


def test_parse_json_block_plain_fenced_and_prose():
    assert parse_json_block('[{"a": 1}]') == [{"a": 1}]
    assert parse_json_block('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert parse_json_block('Ось результат:\n[{"a": 1}]\nдякую') == [{"a": 1}]


def test_build_drafted_items_grounds_spans_and_drops_ungrounded():
    drafts = [
        {"question": "Що таке Київ?", "reference_answer": "столиця", "answer_span": "столицею"},
        {"question": "вигадка", "reference_answer": "x", "answer_span": "Лондон"},  # not in doc
    ]
    items = build_drafted_items("doc1", DOC, drafts, "final")
    assert len(items) == 1  # ungrounded dropped
    item = items[0]
    span = item.source_spans[0]
    assert DOC[span.char_start : span.char_end] == "столицею"  # offsets are exact
    assert item.verified is False and item.provenance == "frontier-drafted"


def test_prepare_goldset_drafts_unverified_items_with_splits(tmp_path):
    (tmp_path / "doc1.md").write_text(DOC, encoding="utf-8")
    payload = json.dumps(
        [{"question": "Що таке Київ?", "reference_answer": "столиця", "answer_span": "столицею"}]
    )
    out = tmp_path / "drafted.jsonl"
    items = prepare_goldset(
        tmp_path, model="x", complete=lambda _p: payload, out_path=out, n_per_doc=1
    )
    assert len(items) == 1 and items[0].verified is False
    assert out.exists()
    assert items[0].split in ("calibration", "tuning", "final")


def test_prepare_synthetic_corpus_rejects_planter_equals_judge():
    with pytest.raises(ValueError, match="planter != judge"):
        prepare_synthetic_corpus(
            ["тема"], planter_model="gpt", judge_model="gpt", complete=lambda _p: "{}"
        )


def test_prepare_synthetic_corpus_writes_docs_labels_and_provenance(tmp_path):
    payload = json.dumps(
        {
            "document": DOC,
            "labels": [
                {"question": "Столиця?", "reference_answer": "Київ", "answer_span": "столицею"}
            ],
        }
    )
    docs, items = prepare_synthetic_corpus(
        ["українські міста"],
        planter_model="planter",
        judge_model="judge",
        complete=lambda _p: payload,
        out_dir=tmp_path,
    )
    assert len(docs) == 1 and len(items) == 1
    assert (tmp_path / "synth-000.md").exists()
    assert (tmp_path / "planted_labels.jsonl").exists()
    prov = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert prov["planter_model"] == "planter" and prov["judge_model"] == "judge"
