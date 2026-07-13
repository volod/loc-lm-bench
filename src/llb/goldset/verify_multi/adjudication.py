"""Disagreement worksheet construction and adjudication command orchestration."""

import logging
from collections.abc import Sequence
from pathlib import Path

from llb.goldset.verify_base import (
    ACCEPT,
    HUMAN_COLS,
    REJECT,
    REVIEWER_COL,
    load_worksheet,
    worksheet_fieldnames,
    write_worksheet_rows,
)
from llb.goldset.verify_multi.agreement import agreement_report, write_agreement_report
from llb.goldset.verify_multi.common import (
    ADJUDICATION_FILENAME,
    ADJUDICATOR_ID,
    PRIOR_DECISIONS_COL,
    decision,
    edit,
    load_reviewer_worksheets,
    reviewer_worksheets_from_manifest,
    rows_by_item,
)

_LOG = logging.getLogger(__name__)


def prior_decisions_note(item_id: str, by_reviewer: dict[str, list[dict[str, str]]]) -> str:
    parts: list[str] = []
    for reviewer in sorted(by_reviewer):
        row = rows_by_item(by_reviewer[reviewer]).get(item_id)
        if row is None:
            continue
        verdict = decision(row) or "undecided"
        suffix = ""
        code = (row.get("reject_code") or "").strip()
        if verdict == REJECT and code:
            suffix = f":{code}"
        elif verdict == ACCEPT and edit(row):
            suffix = f":edit={edit(row)}"
        parts.append(f"{reviewer}={verdict}{suffix}")
    return ";".join(parts)


def build_adjudication_worksheet(
    base_ws: Path, by_reviewer: dict[str, list[dict[str, str]]], disagreements: Sequence[str]
) -> tuple[Path, int]:
    path = Path(base_ws).with_name(ADJUDICATION_FILENAME)
    reviewers = sorted(by_reviewer)
    if not reviewers:
        raise ValueError("no reviewer worksheets to adjudicate")
    source_rows = rows_by_item(by_reviewer[reviewers[0]])
    existing: dict[str, dict[str, str]] = {}
    if path.is_file():
        prior_rows, _ = load_worksheet(path)
        existing = rows_by_item(prior_rows)
    fieldnames = worksheet_fieldnames()
    if PRIOR_DECISIONS_COL not in fieldnames:
        fieldnames.append(PRIOR_DECISIONS_COL)
    rows: list[dict[str, str]] = []
    for item_id in disagreements:
        source = source_rows.get(item_id)
        if source is None:
            continue
        row = dict(source)
        for column in HUMAN_COLS:
            row[column] = ""
        row[REVIEWER_COL] = ADJUDICATOR_ID
        row[PRIOR_DECISIONS_COL] = prior_decisions_note(item_id, by_reviewer)
        carried = existing.get(item_id)
        if carried is not None:
            for column in HUMAN_COLS:
                row[column] = carried.get(column, "")
        rows.append(row)
    write_worksheet_rows(path, rows, fieldnames)
    return path, len(rows)


def run_adjudicate(bundle: Path, base_ws: Path | None = None) -> int:
    bundle = Path(bundle)
    base = Path(base_ws) if base_ws is not None else bundle / "verify_sample.csv"
    worksheets = reviewer_worksheets_from_manifest(base)
    if len(worksheets) < 2:
        _LOG.error("[verify] no multi-reviewer worksheets recorded beside %s", base)
        return 1
    by_reviewer = load_reviewer_worksheets(worksheets)
    report = agreement_report(by_reviewer)
    report_path = write_agreement_report(base, report)
    kappa = report["kappa"]
    _LOG.info(
        "[verify] agreement: %s reviewers, %s jointly decided, observed=%.3f kappa=%s (%s) -> %s",
        len(by_reviewer),
        report["jointly_decided"],
        report["observed_agreement"],
        f"{kappa:.3f}" if isinstance(kappa, float) else "n/a",
        report["kappa_method"],
        report_path,
    )
    disagreements = report["disagreements"]
    assert isinstance(disagreements, list)
    path, count = build_adjudication_worksheet(base, by_reviewer, disagreements)
    if count:
        _LOG.info("[verify] %d disagreement row(s) -> %s", count, path)
    else:
        _LOG.info("[verify] no disagreements -- %s is empty", path)
    return 0
