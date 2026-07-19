"""Policy-driven autonomous and human-paused verification gates."""

from pathlib import Path

import pytest

from llb.auto_rag.verification import (
    VerificationPending,
    _structurally_grounded,
    verify_bundle,
)
from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset, load_goldset
from llb.goldset.verify_base import (
    ACCEPT,
    CHECK_COLS,
    PASS,
    STATUS_DECIDED,
    load_worksheet,
    write_worksheet_rows,
)


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    corpus = bundle / "corpus"
    corpus.mkdir(parents=True)
    text = "Київ є столицею України. Дніпро є великою річкою України."
    (corpus / "facts.md").write_text(text, encoding="utf-8")
    items = []
    for index, split in enumerate(("calibration", "tuning", "final")):
        answer = "Київ" if index < 2 else "Дніпро"
        start = text.index(answer)
        items.append(
            GoldItem(
                id=f"item-{index}",
                question=f"Який факт номер {index}?",
                reference_answer=answer,
                source_doc_id="facts.md",
                source_spans=[
                    SourceSpan(
                        doc_id="facts.md",
                        char_start=start,
                        char_end=start + len(answer),
                        text=answer,
                    )
                ],
                provenance="ontology-drafted",
                split=split,
            )
        )
    dump_goldset(items, bundle / "goldset.jsonl")
    return bundle


def test_auto_gate_uses_policy_scorer_and_emits_accepted_ledger(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    calls = 0

    def scorer(records, model):
        nonlocal calls
        del model
        calls += 1
        return [{"faithfulness": 0.9, "answer_relevancy": 0.8} for _record in records]

    result = verify_bundle(
        bundle,
        tmp_path / "verify",
        policy="local",
        judge_model="judge",
        judge_base_url=None,
        threshold=0.5,
        min_accept_rate=0.5,
        egress_consent=False,
        max_usd=None,
        max_calls=None,
        scorer_ledger=tmp_path / "scorer_ledger.jsonl",
        local_scorer=scorer,
    )

    accepted = load_goldset(result["goldset"])
    assert result["n_accepted"] == 3
    assert all(item.verified for item in accepted)
    assert len((tmp_path / "scorer_ledger.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    verify_bundle(
        bundle,
        tmp_path / "verify",
        policy="local",
        judge_model="judge",
        judge_base_url=None,
        threshold=0.5,
        min_accept_rate=0.5,
        egress_consent=False,
        max_usd=None,
        max_calls=None,
        scorer_ledger=tmp_path / "scorer_ledger.jsonl",
        local_scorer=scorer,
    )
    assert calls == 1


def test_human_gate_pauses_then_resumes_from_workbench_worksheet(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    kwargs = {
        "policy": "human",
        "judge_model": "unused",
        "judge_base_url": None,
        "threshold": 0.5,
        "min_accept_rate": 0.5,
        "egress_consent": False,
        "max_usd": None,
        "max_calls": None,
        "scorer_ledger": tmp_path / "scorer_ledger.jsonl",
    }
    stage = tmp_path / "human"
    with pytest.raises(VerificationPending, match="review 3 pending rows"):
        verify_bundle(bundle, stage, **kwargs)

    worksheet = stage / "verify_sample.csv"
    rows, fields = load_worksheet(worksheet)
    for row in rows:
        row.update({field: PASS for field in CHECK_COLS if field != "chk_planted"})
        row["decision"] = ACCEPT
        row["status"] = STATUS_DECIDED
    write_worksheet_rows(worksheet, rows, fields)
    result = verify_bundle(bundle, stage, **kwargs)
    assert result["policy"] == "human"
    assert result["n_accepted"] == 3


def test_structural_gate_ignores_review_rendering_whitespace() -> None:
    assert _structurally_grounded(
        {
            "reference_answer": "Вода переходить у газоподібний стан.",
            "span_text": "Вода\nпереходить у газоподібний стан.",
            "context": "before >>>Водапереходить у газоподібний стан.<<< after",
        }
    )
