"""Frontier prep utilities (M3.5): pure parsing/grounding/guards, fake LLM completions."""

import json

import pytest

from llb.prep.frontier import (
    ProvenanceLog,
    build_drafted_items,
    ground_span,
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


def test_prepare_goldset_uses_corpus_relative_doc_ids(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "same.md").write_text(DOC, encoding="utf-8")
    (second / "same.md").write_text(DOC, encoding="utf-8")
    payload = json.dumps(
        [{"question": "Що таке Київ?", "reference_answer": "Київ", "answer_span": "Київ"}]
    )
    items = prepare_goldset(tmp_path, model="x", complete=lambda _p: payload, n_per_doc=1)
    assert {item.source_doc_id for item in items} == {"first/same.md", "second/same.md"}
    assert len({item.id for item in items}) == 2


@pytest.mark.parametrize("payload", ['{"question": "not-an-array"}', '["not-an-object"]'])
def test_prepare_goldset_skips_malformed_json_shapes(tmp_path, payload):
    (tmp_path / "doc.md").write_text(DOC, encoding="utf-8")
    assert prepare_goldset(tmp_path, model="x", complete=lambda _p: payload) == []


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
    assert (tmp_path / "corpus" / "synth-000.md").exists()  # ready for build-index --corpus-root
    assert (tmp_path / "planted_labels.jsonl").exists()
    prov = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert prov["planter_model"] == "planter" and prov["judge_model"] == "judge"
    assert prov["synthetic"] is True and prov["corpus_root"] == "corpus"  # explicitly tagged
    assert "total_cost_usd" in prov["cost"]


# --- M3.5 additions: fuzzy-but-exact grounding + per-call cost provenance ------------------


def test_ground_span_exact_then_normalized_then_none():
    # exact substring
    assert ground_span(DOC, "столицею") == (DOC.find("столицею"), "столицею")
    # whitespace/case-normalized still grounds, mapped back to the EXACT doc substring
    doc = "Київ   є  СТОЛИЦЕЮ України."
    start, exact = ground_span(doc, "є столицею") or (None, None)
    assert exact == doc[start : start + len(exact)] and exact.lower().split() == ["є", "столицею"]
    # ungrounded
    assert ground_span(DOC, "Лондон") is None


def test_build_drafted_items_grounds_via_normalization(tmp_path):
    doc = "Київ   є  СТОЛИЦЕЮ України."
    drafts = [{"question": "Столиця?", "reference_answer": "Київ", "answer_span": "столицею"}]
    items = build_drafted_items("d", doc, drafts, "final")
    assert len(items) == 1
    span = items[0].source_spans[0]
    assert doc[span.char_start : span.char_end] == span.text  # exact offsets preserved


def test_provenance_log_summary_aggregates_cost():
    log = ProvenanceLog()
    log.record("gpt-x", 100, 50, 0.002)
    log.record("gpt-x", 80, 40, 0.001)
    summary = log.summary()
    assert summary["calls"] == 2 and summary["models"] == ["gpt-x"]
    assert summary["total_prompt_tokens"] == 180
    assert summary["total_cost_usd"] == pytest.approx(0.003)


def test_prepare_goldset_writes_provenance_sidecar(tmp_path):
    (tmp_path / "doc1.md").write_text(DOC, encoding="utf-8")
    payload = json.dumps([{"question": "Q?", "reference_answer": "Київ", "answer_span": "Київ"}])
    out = tmp_path / "drafted.jsonl"
    prepare_goldset(tmp_path, model="x", complete=lambda _p: payload, out_path=out, n_per_doc=1)
    prov = json.loads((tmp_path / "drafted.provenance.json").read_text(encoding="utf-8"))
    assert prov["synthetic"] is False and prov["model"] == "x" and "cost" in prov


def test_prepare_synthetic_corpus_skips_non_object_payload():
    docs, items = prepare_synthetic_corpus(
        ["тема"],
        planter_model="planter",
        judge_model="judge",
        complete=lambda _p: "[]",
    )
    assert docs == {} and items == []
