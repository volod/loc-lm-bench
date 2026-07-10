"""Rejection-feedback loop into the draft prompts (draft-feedback-rejection-reasons)."""

import json

import pytest

from llb.prep.ontology.endpoint import EndpointConfig
from llb.prep.ontology.feedback import (
    REJECT_CODE_HINTS,
    applied_feedback_block,
    feedback_hint_text,
    feedback_hints,
    load_rejection_feedback,
)
from llb.prep.ontology.pipeline import draft_goldset

DOC = "Alpha керує Beta. Beta належить Gamma. Alpha заснована 1991 року в Києві."


def _summary(**codes):
    return {
        "rejected": sum(cell.get("count", 0) for cell in codes.values()),
        "by_code": codes,
    }


# --- hint mapping (deterministic per code) ----------------------------------------------------


def test_every_closed_reject_code_has_a_hint():
    from llb.goldset.verify import REJECT_CODES

    assert set(REJECT_CODE_HINTS) == set(REJECT_CODES)


def test_hints_are_ordered_by_dominant_code_and_carry_an_example():
    summary = _summary(
        circular={"count": 5, "items": [{"item_id": "a", "note": "питання видає відповідь"}]},
        ungrounded={"count": 2, "items": [{"item_id": "b"}]},
    )
    hints = feedback_hints(summary)
    assert [h["code"] for h in hints] == ["circular", "ungrounded"]
    assert hints[0]["hint"] == REJECT_CODE_HINTS["circular"]
    assert hints[0]["example"] == "питання видає відповідь"
    assert hints[1]["example"] == ""  # no note recorded -> no example
    text = feedback_hint_text(hints)
    assert REJECT_CODE_HINTS["circular"] in text
    assert "питання видає відповідь" in text


def test_empty_summary_is_a_no_op():
    assert feedback_hints(_summary()) == []
    assert feedback_hints(_summary(circular={"count": 0, "items": []})) == []
    assert feedback_hint_text([]) == ""


def test_unknown_codes_are_skipped_and_ties_break_by_code():
    summary = _summary(
        wrong_reference={"count": 1, "items": []},
        bad_question={"count": 1, "items": []},
        mystery={"count": 9, "items": []},
    )
    hints = feedback_hints(summary)
    assert [h["code"] for h in hints] == ["bad_question", "wrong_reference"]


def test_load_rejection_feedback_rejects_non_summaries(tmp_path):
    path = tmp_path / "not_feedback.json"
    path.write_text(json.dumps({"anything": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="by_code"):
        load_rejection_feedback(path)


def test_applied_feedback_block_pins_the_source_digest(tmp_path):
    path = tmp_path / "rejection_reasons.json"
    path.write_text(json.dumps(_summary(circular={"count": 1, "items": []})), encoding="utf-8")
    block = applied_feedback_block(path, feedback_hints(load_rejection_feedback(path)))
    assert block["source"] == str(path)
    assert len(block["sha256"]) == 64
    assert block["hints"] == [{"code": "circular", "count": 1, "example": ""}]


# --- end to end: the hint reaches the draft prompt and provenance ------------------------------


def _fake_endpoint(prompts: list[str]):
    def complete(prompt: str) -> str:
        prompts.append(prompt)
        if "будує онтологію" in prompt:
            return json.dumps(
                {
                    "entities": [{"name": "Alpha", "type": "ORG", "mentions": ["Alpha"]}],
                    "facts": [
                        {
                            "subject": "Alpha",
                            "relation": "керує",
                            "object": "Beta",
                            "evidence": "Alpha керує Beta",
                        }
                    ],
                }
            )
        return json.dumps(
            {
                "question": "Що згадано поряд з Alpha?",
                "reference_answer": "Beta",
                "answer_span": "Beta",
            }
        )

    return complete


def test_draft_prompts_carry_the_feedback_and_provenance_records_it(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.md").write_text(DOC, encoding="utf-8")
    feedback = tmp_path / "rejection_reasons.json"
    feedback.write_text(
        json.dumps(
            _summary(circular={"count": 3, "items": [{"item_id": "x", "note": "циркулярне"}]}),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    prompts: list[str] = []
    out = tmp_path / "bundle"

    draft_goldset(
        corpus,
        EndpointConfig(kind="local", model="fake"),
        complete=_fake_endpoint(prompts),
        max_items=4,
        out_dir=out,
        rejection_feedback=feedback,
    )

    draft_prompts = [p for p in prompts if "укладач набору запитань" in p]
    assert draft_prompts, "the fake endpoint must have received draft prompts"
    assert all(REJECT_CODE_HINTS["circular"] in p for p in draft_prompts)
    assert all("циркулярне" in p for p in draft_prompts)

    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    applied = prov["applied_feedback"]
    assert applied["source"] == str(feedback)
    assert [h["code"] for h in applied["hints"]] == ["circular"]
    assert prov["settings"]["rejection_feedback"] == str(feedback)


def test_draft_without_feedback_is_unchanged(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.md").write_text(DOC, encoding="utf-8")
    prompts: list[str] = []

    draft_goldset(
        corpus,
        EndpointConfig(kind="local", model="fake"),
        complete=_fake_endpoint(prompts),
        max_items=4,
        out_dir=tmp_path / "bundle",
    )

    assert all("Врахуй відгук" not in p for p in prompts)
    prov = json.loads((tmp_path / "bundle" / "provenance.json").read_text(encoding="utf-8"))
    assert "applied_feedback" not in prov
    assert prov["settings"]["rejection_feedback"] is None
