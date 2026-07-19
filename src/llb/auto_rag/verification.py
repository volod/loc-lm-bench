"""Autonomous and human-paused goldset verification gates."""

import json
import re
from pathlib import Path
from typing import Any, cast

from llb.core.contracts.judging import JudgeInputRecord
from llb.goldset.schema import load_goldset
from llb.goldset.verify_acceptance import emit_accepted_ledger, run_accept
from llb.goldset.verify_base import (
    ACCEPT,
    FAIL,
    PASS,
    REJECT,
    STATUS_DECIDED,
    load_worksheet,
    write_worksheet_rows,
)
from llb.goldset.verify_sampling.worksheet import build_sample_worksheet
from llb.scoring.policy import ScorerPolicyRequest, resolve_scorer
from llb.scoring.policy.lanes import ScorerLane


class VerificationPending(RuntimeError):
    """Raised after the human worksheet is ready but still has pending rows."""


GATE_ALGORITHM_REVISION = 2


def verify_bundle(
    bundle: Path,
    stage_dir: Path,
    *,
    policy: str,
    judge_model: str,
    judge_base_url: str | None,
    threshold: float,
    min_accept_rate: float,
    egress_consent: bool,
    max_usd: float | None,
    max_calls: int | None,
    scorer_ledger: Path,
    local_scorer: Any = None,
    frontier_complete: Any = None,
) -> dict[str, Any]:
    """Verify every drafted row, emitting a self-contained accepted ledger."""
    items = load_goldset(bundle / "goldset.jsonl")
    if not items:
        raise ValueError("ontology drafting produced no goldset items")
    worksheet = stage_dir / "verify_sample.csv"
    if not worksheet.is_file():
        build_sample_worksheet(bundle, worksheet, n=len(items))
    rows, fields = load_worksheet(worksheet)
    accepted_dir = stage_dir / "accepted"
    if policy == "human":
        return _human_gate(rows, fields, worksheet, bundle, accepted_dir)
    lane = "frontier" if policy == "frontier" else "local"
    scorer = resolve_scorer(
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
    accepted: list[str] = []
    ledger_rows: list[dict[str, Any]] = []
    for row in rows:
        score = cached[row["item_id"]]
        structural = _structurally_grounded(row)
        passed = (
            structural
            and min(float(score["faithfulness"]), float(score["answer_relevancy"])) >= threshold
        )
        row["chk_grounded"] = (
            PASS if structural and float(score["faithfulness"]) >= threshold else FAIL
        )
        answer_pass = float(score["answer_relevancy"]) >= threshold
        row["chk_answerable"] = PASS if answer_pass else FAIL
        row["chk_reference"] = PASS if float(score["faithfulness"]) >= threshold else FAIL
        row["decision"] = ACCEPT if passed else REJECT
        row["status"] = STATUS_DECIDED
        row["reject_code"] = (
            "" if passed else ("ungrounded" if not structural else "wrong_reference")
        )
        if passed:
            accepted.append(row["item_id"])
        ledger_rows.append(
            {
                "item_id": row["item_id"],
                "algorithm_revision": GATE_ALGORITHM_REVISION,
                "policy": lane,
                "structurally_grounded": structural,
                "threshold": threshold,
                "scores": score,
                "decision": row["decision"],
            }
        )
    write_worksheet_rows(worksheet, rows, fields)
    fresh_ids = {row["item_id"] for row in missing_rows}
    _append_ledger(scorer_ledger, [row for row in ledger_rows if row["item_id"] in fresh_ids])
    rate = len(accepted) / len(rows)
    if rate < min_accept_rate:
        raise ValueError(
            f"autonomous verification accepted {len(accepted)}/{len(rows)} ({rate:.1%}), "
            f"below the {min_accept_rate:.1%} gate"
        )
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


def _human_gate(
    rows: list[dict[str, str]],
    fields: list[str],
    worksheet: Path,
    bundle: Path,
    accepted_dir: Path,
) -> dict[str, Any]:
    del fields
    pending = [row for row in rows if row.get("decision") not in (ACCEPT, REJECT)]
    if pending:
        raise VerificationPending(
            f"review {len(pending)} pending rows in {worksheet} with `llb review {worksheet}`, "
            "then resume the same auto-RAG run"
        )
    if run_accept(worksheet, bundle, accepted_dir) != 0:
        raise ValueError(f"human verification tolerance gate failed: {worksheet}")
    accepted = [row for row in rows if row.get("decision") == ACCEPT]
    items = load_goldset(bundle / "goldset.jsonl")
    _require_eval_splits(items, {row["item_id"] for row in accepted})
    return {
        "policy": "human",
        "worksheet": str(worksheet),
        "accepted_dir": str(accepted_dir),
        "goldset": str(accepted_dir / "goldset.jsonl"),
        "n_total": len(rows),
        "n_accepted": len(accepted),
        "accept_rate": len(accepted) / len(rows),
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
