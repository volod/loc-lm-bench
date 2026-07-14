"""Focused verify acceptance io implementation."""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import cast
from llb.core.fsutil import atomic_write_text
from llb.goldset.verify_base import (
    POLICY_GLOBAL,
    POLICY_WEIGHTED,
    REJECTION_REASONS_FILENAME,
    load_worksheet,
)
from llb.goldset.verify_acceptance_report import (
    rejection_reasons_summary,
)

_LOG = logging.getLogger(__name__)


def _log_report(report: dict[str, object]) -> None:
    _LOG.info(
        "[verify] policy=%s decided=%s accepted=%s rejected=%s reject_rate=%.3f tolerance=%s -> %s",
        report.get("policy", POLICY_GLOBAL),
        report["decided"],
        report["accepted"],
        report["rejected"],
        report["reject_rate"],
        report["tolerance"],
        "PASS" if report["passed"] else "FAIL",
    )
    if report.get("policy") == POLICY_WEIGHTED:
        _LOG.info(
            "[verify] confidence-weighted reject rate: %.3f",
            float(cast(float, report["weighted_reject_rate"])),
        )
    if report["undecided"]:
        _LOG.info("[verify] %s sampled item(s) still undecided", report["undecided"])
    if report["undecided_with_failures"]:
        _LOG.warning(
            "[verify] %s undecided item(s) have a failed check -- decide them before accepting",
            report["undecided_with_failures"],
        )
    per_stratum = report["per_stratum"]
    assert isinstance(per_stratum, dict)
    for key, cell in sorted(per_stratum.items()):
        if not cell["passed"]:
            _LOG.warning(
                "[verify] stratum FAIL (%.3f > tolerance): %s [%d rejected / %d decided]",
                cell["reject_rate"],
                key,
                int(cell["rejected"]),
                int(cell["decided"]),
            )


def write_rejection_reasons(rows: Sequence[dict[str, str]], out_dir: Path) -> Path | None:
    """Export the coded-rejection summary beside the accepted ledger; None when nothing rejected."""
    summary = rejection_reasons_summary(rows)
    if not cast(int, summary["rejected"]):
        return None
    out_path = Path(out_dir) / REJECTION_REASONS_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, json.dumps(summary, ensure_ascii=False, indent=2))
    return out_path


def _accept_rows(worksheet: Path) -> list[dict[str, str]]:
    """The row set acceptance scores: multi-reviewer consensus when the sibling manifest
    records reviewer worksheets (see `verify_multi.py`), else the single worksheet as-is."""
    from llb.goldset.verify_multi.consensus import resolve_multi_reviewer_rows

    consensus = resolve_multi_reviewer_rows(worksheet)
    if consensus is not None:
        _LOG.info(
            "[verify] multi-reviewer bundle: scoring the consensus of the recorded "
            "worksheets (+ adjudication.csv when present)"
        )
        return consensus
    rows, _ = load_worksheet(worksheet)
    return rows
