"""Acceptance arithmetic and accepted-ledger emission for the human verification gate.

Coded rejection reasons, accept-with-edit re-grounding, the global/per-stratum/weighted policies,
and ledger emission live here. Shared constants and worksheet I/O live in `verify_base.py`.
"""

import json
import logging
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from llb.core.fsutil import atomic_write_text
from llb.goldset.chains import CHAINS_FILENAME, ChainItem, dump_chains, load_chains
from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset, load_goldset
from llb.goldset.verify_base import (
    ACCEPT,
    ACCEPT_POLICIES,
    CHECK_COLS,
    CHECK_REJECT_CODES,
    CORPUS_DIRNAME,
    DEFAULT_TOLERANCE,
    FAIL,
    GOLDSET_FILENAME,
    KIND_CHAINS,
    POLICY_GLOBAL,
    POLICY_PER_STRATUM,
    POLICY_WEIGHTED,
    REJECT,
    REJECTION_REASONS_FILENAME,
    _worksheet_sample_kind,
    find_chains,
    find_goldset,
    load_worksheet,
)
from llb.goldset.verify_sampling.confidence import row_confidence

_LOG = logging.getLogger(__name__)


# --- coded rejection reasons ----------------------------------------------------------------


def infer_reject_code(row: dict[str, str]) -> str:
    """The reject code implied by the first failed check, else `other`."""
    for col in CHECK_COLS:
        if (row.get(col) or "").strip() == FAIL:
            return CHECK_REJECT_CODES[col]
    return "other"


def rejection_reasons_summary(rows: Sequence[dict[str, str]]) -> dict[str, object]:
    """Aggregate rejected rows by code for draft feedback (`rejection_reasons.json`).

    Rows rejected before the code column existed fall back to the inferred code, so older
    worksheets still export a useful summary.
    """
    by_code: dict[str, dict[str, object]] = {}
    rejected = 0
    for row in rows:
        if (row.get("decision") or "").strip() != REJECT:
            continue
        rejected += 1
        code = (row.get("reject_code") or "").strip() or infer_reject_code(row)
        cell = by_code.setdefault(code, {"count": 0, "items": []})
        cell["count"] = cast(int, cell["count"]) + 1
        entry: dict[str, str] = {"item_id": (row.get("item_id") or "").strip()}
        note = (row.get("human_note") or "").strip()
        if note:
            entry["note"] = note
        cast(list[dict[str, str]], cell["items"]).append(entry)
    return {"rejected": rejected, "by_code": dict(sorted(by_code.items()))}


# --- accept-with-edit re-grounding ----------------------------------------------------------


def ground_answer(doc_text: str, answer: str, *, hint_start: int = 0) -> tuple[int, int] | None:
    """Locate `answer` verbatim in `doc_text`, preferring the occurrence nearest `hint_start`.

    Returns `(char_start, char_end)` or None when the text does not contain the answer -- the
    caller must then BLOCK the edit until the reviewer re-words it to a verbatim span.
    """
    answer = answer.strip()
    if not answer:
        return None
    starts = [m.start() for m in re.finditer(re.escape(answer), doc_text)]
    if not starts:
        return None
    best = min(starts, key=lambda s: abs(s - hint_start))
    return best, best + len(answer)


def worksheet_edits(rows: Sequence[dict[str, str]]) -> dict[str, str]:
    """Item id -> edited reference answer, for rows carrying a non-empty `edited_answer`."""
    return {
        (row.get("item_id") or "").strip(): (row.get("edited_answer") or "").strip()
        for row in rows
        if (row.get("edited_answer") or "").strip() and (row.get("item_id") or "").strip()
    }


# --- acceptance arithmetic ----------------------------------------------------------------


def _is_decided(row: dict[str, str]) -> bool:
    return (row.get("decision") or "").strip() in (ACCEPT, REJECT)


def _failed_any_check(row: dict[str, str]) -> bool:
    return any((row.get(col) or "").strip() == FAIL for col in CHECK_COLS)


def confidence_weighted_reject_rate(rows: Sequence[dict[str, str]]) -> float:
    """Reject rate where each decided row weighs `1 + max(row_confidence, 0)`.

    A reject on a row the automated signals (cross-check verdict + retrieval rank) rated
    CONFIDENT is worse than a reject those signals already flagged: it means the pipeline's own
    quality signals mispredict, so it counts more against the bundle. Deterministic from the
    worksheet columns alone.
    """
    weighted_total = weighted_rejected = 0.0
    for row in rows:
        if not _is_decided(row):
            continue
        weight = 1.0 + max(row_confidence(row), 0.0)
        weighted_total += weight
        if (row.get("decision") or "").strip() == REJECT:
            weighted_rejected += weight
    return (weighted_rejected / weighted_total) if weighted_total else 0.0


def _stratum_tolerance(key: str, tolerance: float, overrides: dict[str, float] | None) -> float:
    if overrides and key in overrides:
        return overrides[key]
    return tolerance


def acceptance_report(
    rows: Sequence[dict[str, str]],
    tolerance: float = DEFAULT_TOLERANCE,
    *,
    policy: str = POLICY_GLOBAL,
    stratum_tolerances: dict[str, float] | None = None,
) -> dict[str, object]:
    """Acceptance-sampling summary: per-stratum + overall decided/reject counts and pass/fail.

    A decided item is a `reject` defect; the reject RATE over decided items is compared to
    `tolerance`. Items with a failed check but no explicit decision are surfaced as
    `undecided_with_failures` so nothing silently slips through. Pure -- the caller decides
    what to emit.

    `policy` selects the acceptance arithmetic (`ACCEPT_POLICIES`): `global` compares the
    overall rate (per-stratum results stay advisory, the original rule); `per-stratum`
    requires EVERY stratum within its own tolerance (`stratum_tolerances` overrides the
    global default per stratum key); `weighted` compares the confidence-weighted rate.
    """
    if policy not in ACCEPT_POLICIES:
        raise ValueError(f"unknown acceptance policy {policy!r}; use one of {ACCEPT_POLICIES}")
    per_stratum, decided, rejected, undecided_with_failures = _tally_decisions(rows)
    _score_strata(per_stratum, tolerance, stratum_tolerances)
    overall_rate = (rejected / decided) if decided else 0.0
    weighted_rate = confidence_weighted_reject_rate(rows)
    if policy == POLICY_PER_STRATUM:
        passed = decided > 0 and all(bool(c["passed"]) for c in per_stratum.values())
    elif policy == POLICY_WEIGHTED:
        passed = decided > 0 and weighted_rate <= tolerance
    else:
        passed = decided > 0 and overall_rate <= tolerance
    return {
        "tolerance": tolerance,
        "policy": policy,
        "n": len(rows),
        "decided": decided,
        "rejected": rejected,
        "accepted": decided - rejected,
        "undecided": len(rows) - decided,
        "undecided_with_failures": undecided_with_failures,
        "reject_rate": overall_rate,
        "weighted_reject_rate": weighted_rate,
        "passed": passed,
        "per_stratum": per_stratum,
    }


def _tally_decisions(
    rows: Sequence[dict[str, str]],
) -> tuple[dict[str, dict[str, float]], int, int, int]:
    """Count decided/rejected per stratum plus overall + undecided-with-failed-check rows."""
    per_stratum: dict[str, dict[str, float]] = {}
    decided = rejected = 0
    undecided_with_failures = 0
    for row in rows:
        key = row.get("stratum", "") or "(none)"
        cell = per_stratum.setdefault(key, {"decided": 0, "rejected": 0})
        if _is_decided(row):
            decided += 1
            cell["decided"] += 1
            if (row.get("decision") or "").strip() == REJECT:
                rejected += 1
                cell["rejected"] += 1
        elif _failed_any_check(row):
            undecided_with_failures += 1
    return per_stratum, decided, rejected, undecided_with_failures


def _score_strata(
    per_stratum: dict[str, dict[str, float]],
    tolerance: float,
    stratum_tolerances: dict[str, float] | None,
) -> None:
    """Fill each stratum cell with its reject rate, effective tolerance, and pass flag."""
    for key, cell in per_stratum.items():
        d = cell["decided"]
        cell["reject_rate"] = (cell["rejected"] / d) if d else 0.0
        cell["tolerance"] = _stratum_tolerance(key, tolerance, stratum_tolerances)
        cell["passed"] = float(cell["reject_rate"] <= cell["tolerance"])


# --- accepted-ledger emission (the flip is an ADOPTION, not a boolean edit) -----------------


def accepted_ids(rows: Sequence[dict[str, str]]) -> list[str]:
    """Item ids the human explicitly accepted."""
    return [
        (row.get("item_id") or "").strip()
        for row in rows
        if (row.get("decision") or "").strip() == ACCEPT and (row.get("item_id") or "").strip()
    ]


def _apply_edit(item: GoldItem, edited_answer: str, corpus_root: Path) -> GoldItem:
    """Re-ground `edited_answer` in the item's span doc and return the edited item.

    The edited answer must exist VERBATIM in the corpus doc -- an edit that no longer matches any
    span is refused here (raises), so an un-groundable edit can never certify through the ledger
    even if the worksheet cell was hand-edited after the session's own re-grounding.
    """
    span = item.source_spans[0]
    doc_path = corpus_root / span.doc_id
    if not doc_path.is_file():
        raise FileNotFoundError(f"{item.id}: corpus doc for edited answer not found: {doc_path}")
    text = doc_path.read_text(encoding="utf-8")
    offsets = ground_answer(text, edited_answer, hint_start=span.char_start)
    if offsets is None:
        raise ValueError(
            f"{item.id}: edited answer is not a verbatim span of {span.doc_id}; "
            "re-ground it in the review session before accepting"
        )
    start, end = offsets
    new_span = SourceSpan(doc_id=span.doc_id, char_start=start, char_end=end, text=text[start:end])
    return item.model_copy(
        update={
            "reference_answer": edited_answer,
            "source_spans": [new_span, *item.source_spans[1:]],
        }
    )


def emit_accepted_ledger(
    bundle: Path,
    accepted: Sequence[str],
    out_dir: Path,
    *,
    edits: dict[str, str] | None = None,
) -> int:
    """Write an accepted-ledger bundle (`goldset.jsonl` verified=true + sibling `corpus/`).

    The accepted items are taken VERBATIM from the draft bundle (canonical content + grounded
    spans) with `verified` flipped to true, and every corpus doc they depend on is copied so the
    ledger is self-contained. Feeding this to `ingest_squad --verified-goldset <out_dir>/goldset.jsonl`
    re-adopts those ids by REPLACEMENT, which is what stops a reused id from certifying changed
    content. We never hand-edit the boolean in the draft bundle.

    `edits` (item id -> accept-with-edit reference answer from the worksheet) are applied through
    `_apply_edit`: the edited answer is re-grounded against the bundle corpus and the primary span
    replaced; an edit that no longer grounds raises instead of certifying.
    """
    bundle = Path(bundle)
    out_dir = Path(out_dir)
    keep = set(accepted)
    edits = edits or {}
    src_corpus = bundle / CORPUS_DIRNAME
    dst_corpus = out_dir / CORPUS_DIRNAME
    verified: list[GoldItem] = []
    doc_ids: set[str] = set()
    for item in load_goldset(find_goldset(bundle)):
        if item.id not in keep:
            continue
        edited_answer = edits.get(item.id, "")
        if edited_answer:
            item = _apply_edit(item, edited_answer, src_corpus)
        verified.append(item.model_copy(update={"verified": True}))
        doc_ids.add(item.source_doc_id)
        doc_ids.update(span.doc_id for span in item.source_spans)
    dump_goldset(verified, out_dir / GOLDSET_FILENAME)
    for doc_id in sorted(doc_ids):
        source = src_corpus / doc_id
        if not source.is_file():
            raise FileNotFoundError(f"corpus doc for accepted item not found: {source}")
        dest = dst_corpus / doc_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    return len(verified)


def emit_accepted_chain_ledger(
    bundle: Path,
    accepted: Sequence[str],
    out_dir: Path,
) -> int:
    """Write an accepted chain ledger (`chains.jsonl` verified=true + sibling `corpus/`)."""
    bundle = Path(bundle)
    out_dir = Path(out_dir)
    keep = set(accepted)
    src_corpus = bundle / CORPUS_DIRNAME
    dst_corpus = out_dir / CORPUS_DIRNAME
    verified: list[ChainItem] = []
    doc_ids: set[str] = set()
    for chain in load_chains(find_chains(bundle)):
        if chain.chain_id not in keep:
            continue
        verified.append(chain.model_copy(update={"verified": True}))
        for step in chain.steps:
            doc_ids.add(step.source_doc_id)
            doc_ids.update(span.doc_id for span in step.source_spans)
    dump_chains(verified, out_dir / CHAINS_FILENAME)
    for doc_id in sorted(doc_ids):
        source = src_corpus / doc_id
        if not source.is_file():
            raise FileNotFoundError(f"corpus doc for accepted chain not found: {source}")
        dest = dst_corpus / doc_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    return len(verified)


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


def run_accept(
    worksheet: Path,
    bundle: Path,
    out_dir: Path | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    *,
    policy: str = POLICY_GLOBAL,
    stratum_tolerances: dict[str, float] | None = None,
) -> int:
    """The `accept` subcommand: acceptance report, rejection-reason export, ledger emission."""
    rows = _accept_rows(Path(worksheet))
    report = acceptance_report(
        rows, tolerance, policy=policy, stratum_tolerances=stratum_tolerances
    )
    _log_report(report)
    out_dir = out_dir or (Path(bundle) / "accepted")
    reasons_path = write_rejection_reasons(rows, out_dir)
    if reasons_path is not None:
        _LOG.info("[verify] rejection reasons for draft feedback -> %s", reasons_path)
    accepted = accepted_ids(rows)
    if not accepted:
        _LOG.info("[verify] no accepted items -- nothing to flip; resolve the sample first")
        return 0 if report["passed"] else 1
    kind = _worksheet_sample_kind(Path(worksheet))
    if kind == KIND_CHAINS:
        n = emit_accepted_chain_ledger(bundle, accepted, out_dir)
        _LOG.info("[verify] wrote %d accepted chain(s) -> %s", n, out_dir / CHAINS_FILENAME)
    else:
        edits = {k: v for k, v in worksheet_edits(rows).items() if k in set(accepted)}
        n = emit_accepted_ledger(bundle, accepted, out_dir, edits=edits)
        if edits:
            _LOG.info(
                "[verify] applied %d accept-with-edit answer(s) through re-grounding", len(edits)
            )
        _LOG.info("[verify] wrote %d accepted item(s) -> %s", n, out_dir / GOLDSET_FILENAME)
        _LOG.info(
            "[verify] flip via the ledger: python -m llb.prep.ingest_squad "
            "--squad-json <source> --verified-goldset %s",
            out_dir / GOLDSET_FILENAME,
        )
    return 0 if report["passed"] else 1
