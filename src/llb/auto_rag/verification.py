"""Autonomous and human-paused goldset verification gates."""

from pathlib import Path
from typing import Any

from llb.auto_rag.verification_auto import (
    _require_eval_splits,
    _structurally_grounded as _auto_structurally_grounded,
    run_autonomous_gate,
)
from llb.goldset.schema import load_goldset
from llb.goldset.verify_acceptance import run_accept
from llb.goldset.verify_base import (
    ACCEPT,
    REJECT,
    load_worksheet,
)
from llb.goldset.verify_sampling.worksheet import build_sample_worksheet


class VerificationPending(RuntimeError):
    """Raised after the human worksheet is ready but still has pending rows."""


def _structurally_grounded(row: dict[str, str]) -> bool:
    """Compatibility seam for the structural check owned by autonomous verification."""
    return _auto_structurally_grounded(row)


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
    return run_autonomous_gate(
        rows,
        fields,
        worksheet,
        bundle,
        accepted_dir,
        items,
        lane=lane,
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        threshold=threshold,
        min_accept_rate=min_accept_rate,
        egress_consent=egress_consent,
        max_usd=max_usd,
        max_calls=max_calls,
        scorer_ledger=scorer_ledger,
        stage_dir=stage_dir,
        local_scorer=local_scorer,
        frontier_complete=frontier_complete,
    )


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
