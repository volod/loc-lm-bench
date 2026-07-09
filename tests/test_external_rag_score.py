import csv
import json

from llb.scoring.external_rag import (
    HUMAN_DECISION_FIELD,
    HUMAN_NOTES_FIELD,
    HUMAN_SCORE_FIELD,
    classify_external_answer,
    clean_answer_for_scoring,
    load_jsonl,
    score_external_rag_file,
)
from llb.scoring.external_rag_session import PROMPT_HINT, format_card, run_external_rag_session


def test_clean_answer_strips_source_footer_before_scoring():
    answer = "Київ\n\nДжерело: doc - /knowledge/articles/1"

    assert clean_answer_for_scoring(answer) == "Київ"
    assert clean_answer_for_scoring("Київ\nDzherelo: doc") == "Київ"


def test_score_external_rag_writes_csv_and_report(tmp_path):
    answers = tmp_path / "answered.jsonl"
    answers.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "Що є столицею України?",
                "reference_answer": "Київ",
                "split": "final",
                "verified": False,
                "source_doc_id": "doc.txt",
                "source_spans": [
                    {"doc_id": "doc.txt", "char_start": 0, "char_end": 4, "text": "Київ"}
                ],
                "llm_answer": "Київ\n\nДжерело: столиця - /knowledge/articles/a",
                "llm_model": "external-model",
                "llm_provider": "local",
                "llm_route": "rag",
                "llm_sources": [
                    {
                        "article_id": "a",
                        "article_title": "столиця",
                        "score": 0.9,
                        "url": "/knowledge/articles/a",
                    }
                ],
                "llm_error": None,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = score_external_rag_file(answers)

    rows = list(csv.DictReader(result.paths.csv.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["objective_score"] == "1.0"
    assert rows[0]["scored_answer"] == "Київ"
    assert rows[0]["llm_answer"].startswith("Київ")
    assert "\n" not in rows[0]["llm_answer"]
    assert rows[0]["source_1_article_id"] == "a"
    assert "human_score_0_1" in rows[0]
    report = result.paths.report.read_text(encoding="utf-8")
    assert "External RAG score report" in report
    assert "objective mean" in report


def test_external_rag_review_saves_jsonl_and_finalizes_only_when_complete(tmp_path):
    answers = tmp_path / "answered.jsonl"
    records = [
        {
            "id": "q1",
            "question": "Що є столицею України?",
            "reference_answer": "Київ",
            "split": "final",
            "source_spans": [{"doc_id": "doc.txt", "text": "Київ"}],
            "llm_answer": "Київ",
            "llm_sources": [{"article_title": "столиця", "score": 0.9}],
        },
        {
            "id": "q2",
            "question": "Яка валюта України?",
            "reference_answer": "гривня",
            "split": "final",
            "source_spans": [{"doc_id": "doc.txt", "text": "гривня"}],
            "llm_answer": "карбованець",
            "llm_sources": [{"article_title": "валюта", "score": 0.5}],
        },
    ]
    answers.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    messages: list[str] = []
    partial = run_external_rag_session(answers, inputs=["a", "q"], output=messages.append)

    assert partial.complete is False
    assert partial.reviewed == 1
    saved = load_jsonl(answers)
    assert saved[0][HUMAN_SCORE_FIELD] == "1"
    assert saved[0][HUMAN_DECISION_FIELD] == "accept"
    assert saved[1][HUMAN_SCORE_FIELD] == ""
    assert not answers.with_suffix(".csv").exists()
    assert not answers.with_name("answered.report.md").exists()

    final = run_external_rag_session(
        answers,
        inputs=["o", "wrong currency", "r"],
        output=messages.append,
    )

    assert final.complete is True
    assert final.score_result is not None
    saved = load_jsonl(answers)
    assert saved[1][HUMAN_NOTES_FIELD] == "wrong currency"
    assert saved[1][HUMAN_SCORE_FIELD] == "0"
    assert saved[1][HUMAN_DECISION_FIELD] == "reject"
    csv_rows = list(
        csv.DictReader(final.score_result.paths.csv.read_text(encoding="utf-8").splitlines())
    )
    assert {row[HUMAN_DECISION_FIELD] for row in csv_rows} == {"accept", "reject"}
    report = final.score_result.paths.report.read_text(encoding="utf-8")
    assert "Human decisions" in report
    assert "human mean score" in report


def test_external_rag_review_card_is_compact():
    record = {
        "id": "q1",
        "question": "Яку кнопку натиснути?",
        "reference_answer": "Імпортувати ШПС",
        "split": "tuning",
        "source_doc_id": "doc.txt",
        "source_spans": [{"doc_id": "doc.txt", "char_start": 1, "char_end": 2, "text": "текст"}],
        "llm_answer": "Натиснути кнопку імпорту",
        "llm_sources": [{"article_title": "інструкція", "score": 0.9}],
    }

    card = format_card(
        record,
        1,
        2,
        0,
        answer_field=None,
        sources_field=None,
        error_field=None,
        source_limit=3,
        strip_source_footer=True,
    )

    assert card.startswith("===== external RAG human review =====")
    assert "== fields:" not in card
    assert "\n\n== question: Яку кнопку натиснути?" in card
    assert "== question: Яку кнопку натиснути?" in card
    assert "== reference_answer: Імпортувати ШПС" in card
    assert "== llm_sources\n  source 1:" in card
    assert "question:\n  " not in card
    assert PROMPT_HINT.count("\n") == 1
    assert "a=accept(1.0)" in PROMPT_HINT
    assert "p=partial(0.5)" in PROMPT_HINT
    assert "r=reject(0.0)" in PROMPT_HINT
    assert "o=note" in PROMPT_HINT
    assert "w=corrected answer" in PROMPT_HINT
    assert "Enter/n=next" in PROMPT_HINT
    assert "note  corr  n/" not in PROMPT_HINT


def test_score_external_rag_falls_back_to_predicted_answer(tmp_path):
    answers = tmp_path / "answered.jsonl"
    answers.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "Що?",
                "reference_answer": "так",
                "predicted_answer": "так",
                "sources": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = score_external_rag_file(answers)

    assert result.rows[0]["objective_score"] == 1.0
    assert result.rows[0]["answer_field"] == "predicted_answer"


def test_classify_external_answer_detects_abstention():
    assert classify_external_answer("У базі знань немає відповіді", "") == "abstained"
    assert classify_external_answer("", "Timeout") == "error"
