"""category expansion chat-period -- chat-log-shaped synthetic planter + real chat-corpus ingestion."""

import json

from llb.bench.text_analysis import run_text_analysis
from llb.prep import chat_corpus as cc
from llb.scoring import text_analysis as ta


def test_render_chat_log_speaker_text_lines():
    messages = [
        {"from": "Олена", "text": "Бюджет зріс на 15%."},
        {"speaker": "Іван", "content": "Це через нові інвестиції."},
        {"role": "Олена", "text": ""},  # empty body dropped
    ]
    doc = cc.render_chat_log(messages)
    assert "Олена: Бюджет зріс на 15%." in doc
    assert "Іван: Це через нові інвестиції." in doc
    assert doc.count("\n") == 1  # the empty message produced no line


def test_render_chat_log_telegram_text_runs():
    messages = [{"from": "A", "text": ["частина ", {"type": "bold", "text": "важлива"}]}]
    assert cc.render_chat_log(messages) == "A: частина важлива"


def test_load_chat_conversations_array(tmp_path):
    path = tmp_path / "chats.json"
    path.write_text(
        json.dumps(
            [{"id": "c1", "messages": [{"from": "A", "text": "привіт"}]}], ensure_ascii=False
        ),
        encoding="utf-8",
    )
    convos = cc.load_chat_conversations(path)
    assert len(convos) == 1 and convos[0][0] == "c1"


def test_load_chat_conversations_telegram_full_export(tmp_path):
    path = tmp_path / "tg.json"
    path.write_text(
        json.dumps(
            {"chats": {"list": [{"name": "grp", "messages": [{"from": "A", "text": "hi"}]}]}}
        ),
        encoding="utf-8",
    )
    convos = cc.load_chat_conversations(path)
    assert len(convos) == 1 and convos[0][0] == "grp"


def test_load_chat_conversations_single_export(tmp_path):
    path = tmp_path / "one.json"
    path.write_text(json.dumps({"messages": [{"from": "A", "text": "hi"}]}), encoding="utf-8")
    convos = cc.load_chat_conversations(path)
    assert len(convos) == 1 and len(convos[0][1]) == 1


def test_ingest_chat_corpus_drafts_grounded_labels(tmp_path):
    convos = [
        (
            "c1",
            [
                {"from": "Олена", "text": "Бюджет міста зріс на 15 відсотків."},
                {"from": "Іван", "text": "Через нові інвестиції в енергетику."},
            ],
        )
    ]
    # a LOCAL drafter completion (no egress) returning a grounded key_fact label
    labels_json = json.dumps(
        [
            {
                "kind": ta.KEY_FACT,
                "value": "Бюджет міста зріс на 15 відсотків",
                "evidence": "Бюджет міста зріс на 15 відсотків.",
            }
        ],
        ensure_ascii=False,
    )
    docs, records = cc.ingest_chat_corpus(
        convos, complete=lambda _p: labels_json, kinds=(ta.KEY_FACT,), out_dir=tmp_path
    )
    assert len(docs) == 1 and records and records[0]["kind"] == ta.KEY_FACT
    # the bundle is tagged as a REAL corpus with no egress
    prov = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert prov["synthetic"] is False and prov["egress"] == "none"
    # the grounded label carries offsets pointing into the rendered chat doc
    assert records[0].get("char_start") is not None


def test_ingested_bundle_scores_through_text_analysis_real_path(tmp_path):
    convos = [("c1", [{"from": "A", "text": "Київ є столицею України."}])]
    labels_json = json.dumps(
        [{"kind": ta.KEY_FACT, "value": "Київ", "evidence": "Київ є столицею України."}],
        ensure_ascii=False,
    )
    cc.ingest_chat_corpus(
        convos, complete=lambda _p: labels_json, kinds=(ta.KEY_FACT,), out_dir=tmp_path
    )
    # the existing runner consumes the bundle via the REAL path (synthetic=false)
    run = run_text_analysis(
        tmp_path,
        model="m",
        backend="ollama",
        complete=lambda _p: json.dumps({"key_fact": ["Київ"]}, ensure_ascii=False),
        similarity=lambda a, b: 1.0 if a == b else 0.0,
        synthetic=False,
        persist=False,
    )
    assert run.result.objective_score == 1.0


def test_prepare_synthetic_chat_uses_chat_prompt(tmp_path):
    payload = json.dumps(
        {
            "document": "Олена: Бюджет зріс.\nІван: Так, на 15%.",
            "labels": [{"kind": ta.KEY_FACT, "value": "Бюджет зріс", "evidence": "Бюджет зріс."}],
        },
        ensure_ascii=False,
    )
    docs, records = cc.prepare_synthetic_chat_corpus(
        ["міський бюджет"],
        planter_model="planter",
        judge_model="judge",
        kinds=(ta.KEY_FACT,),
        complete=lambda _p: payload,
        out_dir=tmp_path,
    )
    assert docs and records
    prov = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert prov["kind"] == "synthetic-chat" and prov["synthetic"] is True


def test_chat_doc_prompt_is_chat_shaped():
    prompt = cc.chat_doc_prompt("тема", 2, (ta.KEY_FACT,))
    assert "ЧАТ-ЛОГ" in prompt and "репліка" in prompt
