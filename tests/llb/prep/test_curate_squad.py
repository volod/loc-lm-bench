"""Tests for curate squad."""

import json
import pytest
from llb.prep.curation import dispatcher as curation
from llb.prep.curation.input import load_json_documents
from curation_helpers import DOC, FakeEmbedder, _squad_file, corpus as corpus


def test_load_json_documents_handles_fences_and_jsonl(tmp_path):
    fenced = tmp_path / "reply.md"
    fenced.write_text(
        'Ось перша партія:\n```json\n{"a": 1}\n```\nПродовжую:\n```\n[{"b": 2}]\n```\n',
        encoding="utf-8",
    )
    assert load_json_documents(fenced) == [{"a": 1}, [{"b": 2}]]

    jsonl = tmp_path / "chains.jsonl"
    jsonl.write_text('{"chain_id": "c1"}\n{"chain_id": "c2"}\n', encoding="utf-8")
    assert [v["chain_id"] for v in load_json_documents(jsonl)] == ["c1", "c2"]

    with pytest.raises(ValueError, match="empty artifact"):
        empty = tmp_path / "empty.json"
        empty.write_text("", encoding="utf-8")
        load_json_documents(empty)


def test_squad_merge_repair_filter_dedup(tmp_path, corpus):
    q_full = "Хто призначається відповідальною особою за облік матеріальних цінностей?"
    a = _squad_file(
        tmp_path,
        "claude.json",
        [
            {"id": "ext-claude-0001", "q": q_full, "a": "начальник служби"},
            # paraphrased answer (wrong whitespace) -> repaired via normalized grounding
            {
                "id": "ext-claude-0002",
                "q": "Скільки робочих днів триває передача цінностей?",
                "a": "п'яти  робочих  днів",
            },
            # answer not in context at all -> invalid
            {
                "id": "ext-claude-0003",
                "q": "Що складається після передачі справ?",
                "a": "сім примірників",
            },
            # circular: question contains the answer -> flabby
            {
                "id": "ext-claude-0004",
                "q": "Чи акт приймання складається у трьох примірниках?",
                "a": "у трьох примірниках",
            },
            # structure-referencing question -> flabby
            {
                "id": "ext-claude-0005",
                "q": "Що сказано у цьому документі про акт приймання?",
                "a": "Акт приймання",
            },
        ],
    )
    b = _squad_file(
        tmp_path,
        "gemini.json",
        [
            # exact duplicate question of claude 0001 -> exact-dup drop
            {"id": "ext-gemini-0001", "q": q_full, "a": "начальник служби"},
            # near-duplicate (one word changed) -> semantic-dup drop
            {
                "id": "ext-gemini-0002",
                "q": q_full.replace("Хто", "Яка особа"),
                "a": "начальник служби",
            },
            # unique keeper
            {
                "id": "ext-gemini-0003",
                "q": "У скількох примірниках складається акт приймання?",
                "a": "у трьох примірниках",
            },
        ],
    )

    payload, report = curation.curate(
        "squad",
        [a, b],
        corpus_root=corpus,
        embedder=FakeEmbedder(),
        dedup_threshold=0.8,
    )

    kept_ids = [
        qa["id"] for art in payload["data"] for para in art["paragraphs"] for qa in para["qas"]
    ]
    assert kept_ids == ["ext-claude-0001", "ext-claude-0002", "ext-gemini-0003"]
    counts = report.to_dict()["counts"]
    assert counts["invalid"] == 1 and counts["flabby"] == 2
    assert counts["exact_duplicates"] == 1 and counts["near_duplicates"] == 1
    # the whitespace-broken answer was repaired to the exact corpus text
    repaired_answers = [
        qa["answers"][0]["text"]
        for art in payload["data"]
        for para in art["paragraphs"]
        for qa in para["qas"]
        if qa["id"] == "ext-claude-0002"
    ]
    assert repaired_answers == ["п'яти робочих днів"]
    assert any(r["repair"].startswith("answer re-snapped") for r in report.repaired)


def test_squad_context_grounding_fixes_title_and_rejects_unknown(tmp_path, corpus):
    ctx = DOC.split("Передача")[0].strip()  # the first two sentences, verbatim
    good = _squad_file(
        tmp_path,
        "svc.json",
        [
            {
                "id": "x-1",
                "q": "Хто призначається особою, відповідальною за майно?",
                "a": "начальник служби",
            }
        ],
        context=ctx,
        title="wrong-name.md",  # title corrected by grounding search
    )
    bad = _squad_file(
        tmp_path,
        "svc2.json",
        [{"id": "x-2", "q": "Яке питання ставиться до вигаданого тексту?", "a": "вигаданого"}],
        context="Цього тексту немає у жодному документі корпусу, він вигаданий повністю і навмисно досить довгий.",
    )
    payload, report = curation.curate("squad", [good, bad], corpus_root=corpus, embedder=None)
    assert payload["data"][0]["title"] == "doc-a.md"
    assert [r["reason"] for r in report.invalid] == ["context not found in corpus"]
    assert any("title corrected" in r["repair"] for r in report.repaired)


def test_squad_id_collision_and_prior_bundle_dedup(tmp_path, corpus):
    q1 = "Хто призначається відповідальною особою за облік цінностей?"
    q2 = "Скільки примірників акта приймання складається за правилами?"
    a = _squad_file(tmp_path, "s1.json", [{"id": "dup-id", "q": q1, "a": "начальник служби"}])
    b = _squad_file(tmp_path, "s2.json", [{"id": "dup-id", "q": q2, "a": "у трьох примірниках"}])

    # prior bundle whose goldset already covers q2 (verbatim) -> dropped by prior dedup
    bundle = tmp_path / "prior-bundle"
    bundle.mkdir()
    prior_item = {
        "id": "prior-1",
        "lang": "uk",
        "question": q2,
        "reference_answer": "у трьох примірниках",
        "source_doc_id": "doc-a.md",
        "source_spans": [
            {
                "doc_id": "doc-a.md",
                "char_start": DOC.find("у трьох примірниках"),
                "char_end": DOC.find("у трьох примірниках") + len("у трьох примірниках"),
                "text": "у трьох примірниках",
            }
        ],
        "provenance": "ontology-drafted",
        "verified": False,
        "split": "final",
    }
    (bundle / "goldset.jsonl").write_text(
        json.dumps(prior_item, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    prior = curation.load_prior_bundle_questions([bundle])
    assert prior == [q2]

    payload, report = curation.curate(
        "squad",
        [a, b],
        corpus_root=corpus,
        embedder=FakeEmbedder(),
        dedup_threshold=0.99,
        prior_questions=prior,
    )
    kept = [qa for art in payload["data"] for para in art["paragraphs"] for qa in para["qas"]]
    assert [qa["id"] for qa in kept] == ["dup-id"]
    assert report.near_duplicates[0]["duplicate_of"] == "prior-bundle"
