"""Round-trip and detection coverage for every workbench ledger adapter."""

import json
from dataclasses import asdict

from llb.goldset.verify_base import WORKSHEET_COLS, write_worksheet_rows
from llb.goldset.verify_session.commands import _save, _set_decision
from llb.judge.calibration_worksheet import WORKSHEET_COLS as JUDGE_COLS
from llb.judge.calibration_worksheet import write_worksheet_rows as write_judge_rows
from llb.judge.rate.state import save_human_columns, set_rating
from llb.prompt_system.review import PromptCandidate, approve, save_candidates
from llb.prompt_system.template import TemplateFields
from llb.review.adapters import (
    DraftCompareAdapter,
    ExternalRagAdapter,
    GoldsetVerifyAdapter,
    JudgeCalibrationAdapter,
    KnowledgeCutoffAdapter,
    PromptSystemAdapter,
)
from llb.review.registry import open_review
from llb.scoring.external_rag.records import ensure_human_fields, write_jsonl
from llb.scoring.external_rag_common import HUMAN_DECISION_ACCEPT
from llb.scoring.external_rag_session.records import _set_decision as set_external_decision


def _gold_row(item_id: str = "item-1", **values: str) -> dict[str, str]:
    row = {field: "" for field in WORKSHEET_COLS}
    row.update(
        {
            "item_id": item_id,
            "question": "Question?",
            "reference_answer": "Answer",
            "span_text": "Evidence",
            "stratum": "facts",
            **values,
        }
    )
    return row


def test_goldset_adapter_matches_legacy_csv_writer(tmp_path) -> None:
    legacy = tmp_path / "legacy.csv"
    adapted = tmp_path / "adapted.csv"
    row = _gold_row()
    write_worksheet_rows(legacy, [row], WORKSHEET_COLS)
    write_worksheet_rows(adapted, [row], WORKSHEET_COLS)

    rows = [_gold_row()]
    _set_decision(rows[0], "accept")
    _save(legacy, rows, WORKSHEET_COLS)
    adapter = GoldsetVerifyAdapter(adapted)
    adapter.apply(0, "accept")

    assert adapted.read_bytes() == legacy.read_bytes()
    assert open_review(adapted).kind == "goldset-verify"


def test_goldset_check_actions_keep_pass_fail_semantics(tmp_path) -> None:
    path = tmp_path / "verify.csv"
    write_worksheet_rows(path, [_gold_row(synthetic="true")], WORKSHEET_COLS)
    adapter = GoldsetVerifyAdapter(path)
    adapter.apply(0, "check:chk_grounded:pass")
    adapter.apply(0, "check:chk_planted:fail")
    rows = GoldsetVerifyAdapter(path)._worksheets[0].rows
    assert rows[0]["chk_grounded"] == "pass"
    assert rows[0]["chk_planted"] == "fail"


def test_judge_adapter_matches_legacy_csv_writer(tmp_path) -> None:
    legacy = tmp_path / "legacy.csv"
    adapted = tmp_path / "adapted.csv"
    row = {field: "" for field in JUDGE_COLS}
    row.update({"item_id": "rate-1", "question": "Q", "model_answer": "A"})
    write_judge_rows(legacy, [row])
    write_judge_rows(adapted, [row])

    set_rating(row, 4)
    save_human_columns(legacy, [row], JUDGE_COLS)
    adapter = JudgeCalibrationAdapter(adapted)
    adapter.apply(0, "4")

    assert adapted.read_bytes() == legacy.read_bytes()
    assert open_review(adapted).kind == "judge-calibration"


def test_external_adapter_matches_legacy_jsonl_writer(tmp_path) -> None:
    legacy = tmp_path / "legacy.jsonl"
    adapted = tmp_path / "adapted.jsonl"
    record = {"id": "rag-1", "question": "Q", "reference_answer": "A", "llm_answer": "A"}
    legacy_rows = [dict(record)]
    adapted_rows = [dict(record)]
    ensure_human_fields(legacy_rows)
    ensure_human_fields(adapted_rows)
    write_jsonl(legacy, legacy_rows)
    write_jsonl(adapted, adapted_rows)

    set_external_decision(legacy_rows[0], HUMAN_DECISION_ACCEPT)
    write_jsonl(legacy, legacy_rows)
    adapter = ExternalRagAdapter(adapted)
    adapter.apply(0, HUMAN_DECISION_ACCEPT)

    assert adapted.read_bytes() == legacy.read_bytes()
    assert open_review(adapted).kind == "external-rag"


def _candidate() -> PromptCandidate:
    return PromptCandidate(
        prompt_system_id="prompt-1",
        fields=TemplateFields(),
        system_prompt="System",
        additional_prompt="Additional",
        dropped_context={"sections": []},
        used_tokens=12,
    )


def test_prompt_adapter_matches_legacy_json_writer(tmp_path) -> None:
    legacy = tmp_path / "legacy.json"
    adapted = tmp_path / "candidates.json"
    save_candidates([_candidate()], legacy)
    save_candidates([_candidate()], adapted)

    candidate = _candidate()
    approve(candidate)
    save_candidates([candidate], legacy)
    adapter = PromptSystemAdapter(adapted)
    adapter.apply(0, "approved")

    assert adapted.read_bytes() == legacy.read_bytes()
    assert open_review(adapted).kind == "prompt-system"


def test_cutoff_and_draft_compare_are_named_goldset_adapters(tmp_path) -> None:
    bundle = tmp_path / "cutoff"
    bundle.mkdir()
    cutoff_fields = [*WORKSHEET_COLS, "review_profile"]
    row = _gold_row(review_profile="knowledge-cutoff-translation")
    write_worksheet_rows(bundle / "translation_review.csv", [row], cutoff_fields)
    assert isinstance(open_review(bundle), KnowledgeCutoffAdapter)

    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    write_worksheet_rows(left, [_gold_row("left")], WORKSHEET_COLS)
    write_worksheet_rows(right, [_gold_row("right")], WORKSHEET_COLS)
    report = tmp_path / "comparison.json"
    report.write_text(
        json.dumps(
            {
                "lane_order": ["left", "right"],
                "lanes": {
                    "left": {"verify_sample": {"worksheet": str(left)}},
                    "right": {"verify_sample": {"worksheet": str(right)}},
                },
            }
        ),
        encoding="utf-8",
    )
    adapter = open_review(report)
    assert isinstance(adapter, DraftCompareAdapter)
    assert len(adapter) == 2
    adapter.apply(1, "reject")
    assert GoldsetVerifyAdapter(right).record(0).verdict == "reject"


def test_candidate_fields_remain_serializable() -> None:
    assert "role" in asdict(_candidate().fields)
