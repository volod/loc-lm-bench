"""Frontier-judge agreement lane: worksheet inputs, rho, cost math, and the report bundle.

Every provider call is an injected fake completer -- no network, no spend.
"""

import json
import re

import pytest

from llb.judge.calibration_worksheet import write_worksheet_rows
from llb.scoring.frontier_agreement import (
    AGREEMENT_FILENAME,
    CAP_SAFETY_FACTOR,
    REPORT_FILENAME,
    SCORES_FILENAME,
    correlate,
    cost_summary,
    load_agreement_items,
    provider_slug,
    run_frontier_agreement,
)
from llb.scoring.policy.errors import ScorerPolicyError

MODEL_A = "anthropic/fake-judge"
MODEL_B = "openai/fake-judge"
FAKE_COST_USD = 0.002


def _row(item_id: str, human: str, judge: str, answer: str = "Київ") -> dict:
    return {
        "item_id": item_id,
        "split": "calibration",
        "provenance": "public-reused",
        "question": "Столиця України?",
        "reference_answer": "Київ",
        "model_answer": answer,
        "human_answer": "",
        "human_rating": human,
        "human_note": "",
        "human_status": "rated" if human else "pending",
        "judge_rating": judge,
    }


def _worksheet(tmp_path, rows):
    path = tmp_path / "worksheet.csv"
    write_worksheet_rows(path, rows)
    return path


def _graded_rows(n: int = 6) -> list[dict]:
    """Rows whose human and local ratings both rise with the index (rank-correlated)."""
    return [
        _row(
            f"item-{index}",
            str(index + 1),
            f"{0.1 * (index + 1):.2f}",
            answer=f"відповідь #{index}",
        )
        for index in range(n)
    ]


def _ranked_completer(model: str):
    """Fake provider whose score rises with the answer's index, mirroring the ratings."""

    def complete(prompt: str) -> tuple[str, float, int, int]:
        index = int(re.search(r"відповідь #(\d+)", prompt).group(1))
        score = min(1.0, 0.1 * (index + 1))
        payload = json.dumps({"faithfulness": score, "answer_relevancy": score})
        return payload, FAKE_COST_USD, 100, 20

    del model
    return complete


def test_load_items_skips_unanswered_rows_and_reads_both_ratings(tmp_path):
    rows = [_row("a", "5", "0.9"), _row("b", "", "", answer=""), _row("c", "2", "")]
    items = load_agreement_items(_worksheet(tmp_path, rows))
    assert [item.item_id for item in items] == ["a", "c"]
    assert items[0].human_rating == 5.0
    assert items[0].local_rating == 0.9
    assert items[1].local_rating is None
    assert items[0].contexts == ["Київ"]


def test_load_items_grounds_contexts_on_the_goldset_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    doc = "Довгий вступний текст. Київ — столиця України. Далі йде інший розділ."
    (corpus / "doc.txt").write_text(doc, encoding="utf-8")
    start = doc.index("Київ")
    goldset = tmp_path / "goldset.jsonl"
    goldset.write_text(
        json.dumps(
            {
                "id": "a",
                "question": "Столиця України?",
                "reference_answer": "Київ",
                "source_doc_id": "doc.txt",
                "source_spans": [
                    {
                        "doc_id": "doc.txt",
                        "char_start": start,
                        "char_end": start + len("Київ"),
                        "text": "Київ",
                    }
                ],
                "provenance": "public-reused",
                "verified": True,
                "split": "calibration",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    items = load_agreement_items(_worksheet(tmp_path, [_row("a", "5", "0.9")]), goldset=goldset)
    assert len(items[0].contexts) == 1
    # The window carries surrounding document text, not just the bare span.
    assert "Київ" in items[0].contexts[0]
    assert len(items[0].contexts[0]) > len("Київ")


def test_correlate_ignores_scale_and_needs_two_pairs():
    human = [1.0, 2.0, 3.0, 4.0]
    judged = [0.1, 0.2, 0.3, 0.4]
    result = correlate(human, judged)
    assert result is not None
    assert result["rho"] == pytest.approx(1.0)
    assert result["trusted"] is True
    assert correlate([None, None, 3.0, None], judged) is None


def test_cost_summary_scales_the_cap_by_the_safety_factor():
    summary = cost_summary({"cost_usd": 0.2, "calls": 10, "max_usd": 1.0}, n_items=10)
    assert summary["cost_per_item_usd"] == pytest.approx(0.02)
    # 0.02 * 10 * factor, rounded up to the cent.
    assert summary["recommended_cap_usd"] == pytest.approx(0.02 * 10 * CAP_SAFETY_FACTOR)
    assert summary["priced"] is True


def test_cost_summary_declines_to_recommend_a_cap_for_an_unpriced_model():
    summary = cost_summary({"cost_usd": 0.0, "calls": 10}, n_items=10)
    assert summary["priced"] is False
    assert summary["cost_per_item_usd"] is None
    assert summary["recommended_cap_usd"] is None


def test_run_writes_agreement_report_and_per_provider_artifacts(tmp_path):
    worksheet = _worksheet(tmp_path, _graded_rows())
    out_dir = tmp_path / "out"
    payload, path = run_frontier_agreement(
        worksheet,
        [MODEL_A, MODEL_B],
        out_dir=out_dir,
        max_usd=1.0,
        complete_factory=_ranked_completer,
    )
    assert path == out_dir
    assert payload["n_items"] == 6
    assert payload["failures"] == []
    assert [p["model"] for p in payload["providers"]] == [MODEL_A, MODEL_B]

    provider = payload["providers"][0]
    # The fake tracks both reference ratings exactly, so both correlations are perfect.
    assert provider["vs_human"]["mean"]["rho"] == pytest.approx(1.0)
    assert provider["vs_local"]["mean"]["rho"] == pytest.approx(1.0)
    assert provider["recommendation"] == "trusted"
    assert provider["human_decision"] == "pending"
    assert provider["cost"]["cost_per_item_usd"] == pytest.approx(FAKE_COST_USD)

    assert json.loads((out_dir / AGREEMENT_FILENAME).read_text(encoding="utf-8"))
    report = (out_dir / REPORT_FILENAME).read_text(encoding="utf-8")
    assert "Frontier vs human rating" in report
    assert "Human decision" in report

    run_dir = out_dir / provider_slug(MODEL_A)
    assert (run_dir / "scorer" / "consent.json").is_file()
    scores = [
        json.loads(line)
        for line in (run_dir / SCORES_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert len(scores) == 6
    assert scores[0]["item_id"] == "item-0"
    assert scores[0]["mean"] == pytest.approx(0.1)


def test_budget_cap_records_a_failure_without_discarding_other_providers(tmp_path):
    worksheet = _worksheet(tmp_path, _graded_rows())
    payload, out_dir = run_frontier_agreement(
        worksheet,
        [MODEL_A, MODEL_B],
        out_dir=tmp_path / "out",
        max_calls=2,
        complete_factory=_ranked_completer,
    )
    # Both providers exhaust the same cap, so both fail -- and both keep their partial ledgers.
    assert [f["model"] for f in payload["failures"]] == [MODEL_A, MODEL_B]
    assert payload["providers"] == []
    for model in (MODEL_A, MODEL_B):
        ledger = tmp_path / "out" / provider_slug(model) / "scorer" / "ledger.jsonl"
        assert len(ledger.read_text(encoding="utf-8").splitlines()) == 2


def test_budget_abort_resumes_scored_cases_instead_of_re_spending(tmp_path):
    worksheet = _worksheet(tmp_path, _graded_rows())
    out_dir = tmp_path / "out"
    run_frontier_agreement(
        worksheet, [MODEL_A], out_dir=out_dir, max_calls=2, complete_factory=_ranked_completer
    )
    calls: list[str] = []

    def counting_factory(model: str):
        inner = _ranked_completer(model)

        def complete(prompt: str):
            calls.append(prompt)
            return inner(prompt)

        return complete

    payload, _ = run_frontier_agreement(
        worksheet, [MODEL_A], out_dir=out_dir, max_calls=6, complete_factory=counting_factory
    )
    assert payload["failures"] == []
    # Two cases were already checkpointed; only the remaining four hit the provider.
    assert len(calls) == 4


def test_run_refuses_a_worksheet_with_no_judgeable_rows(tmp_path):
    worksheet = _worksheet(tmp_path, [_row("a", "5", "0.9", answer="")])
    with pytest.raises(ScorerPolicyError, match="no judgeable rows"):
        run_frontier_agreement(
            worksheet,
            [MODEL_A],
            out_dir=tmp_path / "out",
            max_usd=1.0,
            complete_factory=_ranked_completer,
        )
