"""Autonomous scorer execution and decision persistence for auto-RAG verification."""

import json
import re
from pathlib import Path
from typing import Any, cast

from llb.core.contracts.judging import JudgeInputRecord
from llb.goldset.verify_acceptance import emit_accepted_ledger
from llb.goldset.verify_base import (
    ACCEPT,
    FAIL,
    PASS,
    REJECT,
    STATUS_DECIDED,
    write_worksheet_rows,
)
from llb.scoring.policy import ScorerPolicyRequest, resolve_scorer
from llb.scoring.policy.lanes import ScorerLane

GATE_ALGORITHM_REVISION = 2


def _resolve_scorer(
    *,
    lane: str,
    judge_model: str,
    judge_base_url: str | None,
    egress_consent: bool,
    max_usd: float | None,
    max_calls: int | None,
    stage_dir: Path,
    local_scorer: Any,
    frontier_complete: Any,
) -> Any:
    return resolve_scorer(
        ScorerPolicyRequest(
            lane=cast(ScorerLane, lane),
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            egress_consent=egress_consent,
            max_usd=max_usd,
            max_calls=max_calls,
            run_dir=stage_dir,
            local_scorer=local_scorer,
            frontier_complete=frontier_complete,
        )
    )


def _score_missing_rows(
    rows: list[dict[str, str]],
    scorer: Any,
    scorer_ledger: Path,
    *,
    lane: str,
    threshold: float,
    judge_model: str,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    cached = _load_scores(scorer_ledger, lane, threshold)
    missing_rows = [row for row in rows if row["item_id"] not in cached]
    records: list[JudgeInputRecord] = [
        {
            "question": row["question"],
            "answer": row["reference_answer"],
            "contexts": [row["context"]],
        }
        for row in missing_rows
    ]
    fresh_scores = scorer.scorer(records, judge_model) if records else []
    cached.update(zip((row["item_id"] for row in missing_rows), fresh_scores))
    return cached, missing_rows


def _apply_score(
    row: dict[str, str],
    score: dict[str, Any],
    *,
    lane: str,
    threshold: float,
) -> tuple[bool, dict[str, Any]]:
    faithfulness = float(score["faithfulness"])
    answer_relevancy = float(score["answer_relevancy"])
    structural = _structurally_grounded(row)
    passed = structural and min(faithfulness, answer_relevancy) >= threshold
    row["chk_grounded"] = PASS if structural and faithfulness >= threshold else FAIL
    row["chk_answerable"] = PASS if answer_relevancy >= threshold else FAIL
    row["chk_reference"] = PASS if faithfulness >= threshold else FAIL
    row["decision"] = ACCEPT if passed else REJECT
    row["status"] = STATUS_DECIDED
    row["reject_code"] = "" if passed else ("ungrounded" if not structural else "wrong_reference")
    return passed, {
        "item_id": row["item_id"],
        "algorithm_revision": GATE_ALGORITHM_REVISION,
        "policy": lane,
        "structurally_grounded": structural,
        "threshold": threshold,
        "scores": score,
        "decision": row["decision"],
    }


def _decide_rows(
    rows: list[dict[str, str]],
    scores: dict[str, Any],
    *,
    lane: str,
    threshold: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    accepted: list[str] = []
    ledger_rows: list[dict[str, Any]] = []
    for row in rows:
        passed, ledger_row = _apply_score(
            row, scores[row["item_id"]], lane=lane, threshold=threshold
        )
        if passed:
            accepted.append(row["item_id"])
        ledger_rows.append(ledger_row)
    return accepted, ledger_rows


def _check_accept_rate(accepted: list[str], total: int, minimum: float) -> float:
    rate = len(accepted) / total
    if rate < minimum:
        raise ValueError(
            f"autonomous verification accepted {len(accepted)}/{total} ({rate:.1%}), "
            f"below the {minimum:.1%} gate"
        )
    return rate


def run_autonomous_gate(
    rows: list[dict[str, str]],
    fields: list[str],
    worksheet: Path,
    bundle: Path,
    accepted_dir: Path,
    items: list[Any],
    *,
    lane: str,
    judge_model: str,
    judge_base_url: str | None,
    threshold: float,
    min_accept_rate: float,
    egress_consent: bool,
    max_usd: float | None,
    max_calls: int | None,
    scorer_ledger: Path,
    stage_dir: Path,
    local_scorer: Any,
    frontier_complete: Any,
) -> dict[str, Any]:
    """Score undecided rows, persist their decisions, and emit the accepted ledger."""
    scorer = _resolve_scorer(
        lane=lane,
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        egress_consent=egress_consent,
        max_usd=max_usd,
        max_calls=max_calls,
        stage_dir=stage_dir,
        local_scorer=local_scorer,
        frontier_complete=frontier_complete,
    )
    scores, missing_rows = _score_missing_rows(
        rows,
        scorer,
        scorer_ledger,
        lane=lane,
        threshold=threshold,
        judge_model=judge_model,
    )
    accepted, ledger_rows = _decide_rows(rows, scores, lane=lane, threshold=threshold)
    write_worksheet_rows(worksheet, rows, fields)
    fresh_ids = {row["item_id"] for row in missing_rows}
    _append_ledger(scorer_ledger, [row for row in ledger_rows if row["item_id"] in fresh_ids])
    rate = _check_accept_rate(accepted, len(rows), min_accept_rate)
    _require_eval_splits(items, set(accepted))
    emit_accepted_ledger(bundle, accepted, accepted_dir)
    return {
        "policy": lane,
        "worksheet": str(worksheet),
        "accepted_dir": str(accepted_dir),
        "goldset": str(accepted_dir / "goldset.jsonl"),
        "n_total": len(rows),
        "n_accepted": len(accepted),
        "accept_rate": rate,
        "scorer": scorer.metadata or {},
    }


def _structurally_grounded(row: dict[str, str]) -> bool:
    answer = row.get("reference_answer", "").strip()
    span = row.get("span_text", "").strip()
    context = row.get("context", "")
    return bool(answer and span and _normalize_text(span) in _normalize_text(context))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value).replace(">>>", "").replace("<<<", "")


def _require_eval_splits(items: list[Any], accepted: set[str]) -> None:
    splits = {item.split for item in items if item.id in accepted}
    missing = sorted({"tuning", "final"} - splits)
    if missing:
        raise ValueError(
            "verification left no accepted rows for required split(s): " + ", ".join(missing)
        )


def _append_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_scores(path: Path, policy: str, threshold: float) -> dict[str, Any]:
    if not path.is_file():
        return {}
    scores: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            row.get("algorithm_revision") == GATE_ALGORITHM_REVISION
            and row.get("policy") == policy
            and row.get("threshold") == threshold
            and isinstance(row.get("scores"), dict)
        ):
            scores[str(row["item_id"])] = row["scores"]
    return scores
