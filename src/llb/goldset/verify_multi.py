"""Multi-annotator support for the human verification gate (adjudication lane).

Extends the single-reviewer worksheet flow in `verify.py` with more than one annotator:

- `build_multi_reviewer_worksheets` draws ONE stratified sample and writes it as `k`
  per-reviewer worksheets (`verify_sample.r<i>.csv`), each row stamped with a `reviewer_id`.
  Every reviewer verifies the SAME items -- inter-annotator agreement is only defined over
  shared ratings.
- `agreement_report` computes observed agreement plus Cohen's kappa (2 reviewers) or
  Fleiss' kappa (3+) over the jointly decided rows, and lists the disagreement item ids.
- `build_adjudication_worksheet` draws EXACTLY the disagreement rows into `adjudication.csv`,
  carrying every prior decision forward in a read-only `prior_decisions` column; the
  adjudicator reviews it with the ordinary `verify-review` session.
- `consensus_rows` merges the per-reviewer worksheets and the adjudication pass into one
  effective row set for acceptance: unanimous decisions stand, adjudicated decisions override
  disagreements, and anything else stays undecided (so acceptance blocks on it).

Everything here is pure worksheet/JSON I/O -- no model, endpoint, or GPU -- so it is fully
unit-tested. The acceptance-policy arithmetic itself (global / per-stratum / weighted) lives
in `verify.py` beside the single-reviewer acceptance report.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path

from llb.core.fsutil import atomic_write_text
from llb.goldset.schema import load_goldset
from llb.goldset.verify import (
    ACCEPT,
    HUMAN_COLS,
    REJECT,
    REVIEWER_COL,
    SAMPLE_MANIFEST,
    bundle_is_synthetic,
    draw_stratified_sample,
    find_goldset,
    load_worksheet,
    stratum_key,
    worksheet_fieldnames,
    write_worksheet_rows,
)

_LOG = logging.getLogger(__name__)

AGREEMENT_FILENAME = "agreement.json"
ADJUDICATION_FILENAME = "adjudication.csv"
# Read-only adjudication column: every prior reviewer decision, e.g. "r1=accept;r2=reject".
PRIOR_DECISIONS_COL = "prior_decisions"
ADJUDICATOR_ID = "adjudicator"


def reviewer_id(index: int) -> str:
    """Reviewer ids are positional and stable: r1, r2, ... rK."""
    return f"r{index}"


def reviewer_worksheet_path(base: Path, index: int) -> Path:
    """Per-reviewer worksheet path derived from the base worksheet name.

    `<dir>/verify_sample.csv` + reviewer 2 -> `<dir>/verify_sample.r2.csv`.
    """
    base = Path(base)
    return base.with_name(f"{base.stem}.{reviewer_id(index)}{base.suffix}")


# --- agreement math (pure) ------------------------------------------------------------------


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa between two equal-length label sequences.

    Chance agreement uses each rater's own marginal label distribution. Degenerate case
    (chance agreement 1.0, i.e. both raters constant): 1.0 on perfect agreement, else 0.0.
    """
    if len(a) != len(b):
        raise ValueError(f"label sequences differ in length: {len(a)} != {len(b)}")
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    expected = sum((list(a).count(c) / n) * (list(b).count(c) / n) for c in labels)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def fleiss_kappa(counts: Sequence[Sequence[int]]) -> float:
    """Fleiss' kappa over an items x categories count matrix (same rater count per item).

    Each row holds how many raters assigned the item to each category; every row must sum to
    the same number of raters (>= 2). Degenerate case (all raters always pick one category):
    1.0 on perfect agreement, else 0.0.
    """
    rows = [list(row) for row in counts if sum(row) > 0]
    if not rows:
        return 0.0
    raters = sum(rows[0])
    if raters < 2:
        raise ValueError("Fleiss' kappa needs at least 2 raters per item")
    if any(sum(row) != raters for row in rows):
        raise ValueError("every item must be rated by the same number of raters")
    n_items = len(rows)
    n_categories = len(rows[0])
    p_item = [(sum(c * c for c in row) - raters) / (raters * (raters - 1)) for row in rows]
    p_bar = sum(p_item) / n_items
    totals = [sum(row[j] for row in rows) for j in range(n_categories)]
    p_cat = [t / (n_items * raters) for t in totals]
    p_expected = sum(p * p for p in p_cat)
    if p_expected >= 1.0:
        return 1.0 if p_bar >= 1.0 else 0.0
    return (p_bar - p_expected) / (1.0 - p_expected)


# --- multi-reviewer sampling ------------------------------------------------------------------


def _load_manifest(base_ws: Path) -> dict[str, object]:
    path = Path(base_ws).with_name(SAMPLE_MANIFEST)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_multi_reviewer_worksheets(
    bundle: Path,
    out_path: Path,
    *,
    n: int,
    annotators: int,
    seed: int = 13,
    kind: str = "auto",
) -> list[Path]:
    """Draw ONE stratified sample and write it as `annotators` per-reviewer worksheets.

    All reviewers get identical rows (same items, same read-only context) with only
    `reviewer_id` differing -- agreement statistics need shared ratings. The sibling
    `sample_manifest.json` records the reviewer worksheets; it intentionally omits the
    single-`worksheet` key so a multi-reviewer bundle can only stamp `--data-verified`
    through its accepted ledger, never through one reviewer's sheet alone.
    """
    from llb.goldset.chains import chain_stratum_key, load_chains
    from llb.goldset.verify import (
        KIND_CHAINS,
        _sample_chain_rows,
        _sample_rows,
        draw_chain_sample,
        find_chains,
        resolve_sample_kind,
    )

    if annotators < 2:
        raise ValueError("multi-reviewer sampling needs --annotators >= 2")
    bundle = Path(bundle)
    out_path = Path(out_path)
    resolved_kind = resolve_sample_kind(bundle, kind)
    synthetic = bundle_is_synthetic(bundle) if resolved_kind != KIND_CHAINS else False
    if resolved_kind == KIND_CHAINS:
        chains = load_chains(find_chains(bundle))
        chain_sample = draw_chain_sample(chains, n, seed=seed)
        rows = _sample_chain_rows(bundle, chain_sample)
        population = len(chains)
        sample_size = len(chain_sample)
        strata_sizes: dict[str, int] = {}
        for chain in chain_sample:
            key = chain_stratum_key(chain)
            strata_sizes[key] = strata_sizes.get(key, 0) + 1
    else:
        items = load_goldset(find_goldset(bundle))
        gold_sample = draw_stratified_sample(items, n, seed=seed)
        rows = _sample_rows(bundle, gold_sample, synthetic=synthetic)
        population = len(items)
        sample_size = len(gold_sample)
        strata_sizes = {}
        for it in gold_sample:
            strata_sizes[stratum_key(it)] = strata_sizes.get(stratum_key(it), 0) + 1

    paths: list[Path] = []
    for i in range(1, annotators + 1):
        ws_path = reviewer_worksheet_path(out_path, i)
        reviewer_rows = [{**row, REVIEWER_COL: reviewer_id(i)} for row in rows]
        write_worksheet_rows(ws_path, reviewer_rows)
        paths.append(ws_path)

    manifest = {
        "bundle": str(bundle),
        "kind": resolved_kind,
        "annotators": annotators,
        "worksheets": [str(p) for p in paths],
        "synthetic": synthetic,
        "seed": seed,
        "requested": n,
        "sample_size": sample_size,
        "population": population,
        "strata": strata_sizes,
    }
    atomic_write_text(
        out_path.with_name(SAMPLE_MANIFEST), json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    return paths


def reviewer_worksheets_from_manifest(base_ws: Path) -> list[Path]:
    """The per-reviewer worksheet paths the sibling `sample_manifest.json` records ([] if none)."""
    manifest = _load_manifest(base_ws)
    worksheets = manifest.get("worksheets")
    if not isinstance(worksheets, list):
        return []
    return [Path(str(p)) for p in worksheets if p]


def load_reviewer_worksheets(paths: Sequence[Path]) -> dict[str, list[dict[str, str]]]:
    """Load per-reviewer worksheets keyed by reviewer id (falls back to the file stem)."""
    by_reviewer: dict[str, list[dict[str, str]]] = {}
    for path in paths:
        rows, _ = load_worksheet(Path(path))
        rid = next(
            ((row.get(REVIEWER_COL) or "").strip() for row in rows if row.get(REVIEWER_COL)),
            Path(path).stem,
        )
        by_reviewer[rid] = rows
    return by_reviewer


# --- agreement report -------------------------------------------------------------------------


def _decision(row: dict[str, str]) -> str:
    return (row.get("decision") or "").strip()


def _edit(row: dict[str, str]) -> str:
    return (row.get("edited_answer") or "").strip()


def _rows_by_item(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        (row.get("item_id") or "").strip(): row
        for row in rows
        if (row.get("item_id") or "").strip()
    }


def _joint_item_ids(by_reviewer: dict[str, list[dict[str, str]]]) -> list[str]:
    """Item ids present in EVERY reviewer worksheet, in first-worksheet order."""
    reviewers = list(by_reviewer)
    if not reviewers:
        return []
    first = [
        (row.get("item_id") or "").strip()
        for row in by_reviewer[reviewers[0]]
        if (row.get("item_id") or "").strip()
    ]
    others = [set(_rows_by_item(by_reviewer[r])) for r in reviewers[1:]]
    return [item_id for item_id in first if all(item_id in ids for ids in others)]


def _is_disagreement(decisions: Sequence[str], edits: Sequence[str]) -> bool:
    """All reviewers decided but the outcomes differ.

    A unanimous accept with DIFFERING accept-with-edit answers is also a disagreement -- the
    edit changes the content the ledger would certify, so it must be adjudicated too.
    """
    if any(d not in (ACCEPT, REJECT) for d in decisions):
        return False
    if len(set(decisions)) > 1:
        return True
    return decisions[0] == ACCEPT and len({e for e in edits}) > 1


def _joint_decisions(
    indexed: dict[str, dict[str, dict[str, str]]],
    reviewers: Sequence[str],
    joint: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Split the joint item ids into (jointly decided, disagreements)."""
    jointly_decided: list[str] = []
    disagreements: list[str] = []
    for item_id in joint:
        decisions = [_decision(indexed[r][item_id]) for r in reviewers]
        edits = [_edit(indexed[r][item_id]) for r in reviewers]
        if all(d in (ACCEPT, REJECT) for d in decisions):
            jointly_decided.append(item_id)
            if _is_disagreement(decisions, edits):
                disagreements.append(item_id)
    return jointly_decided, disagreements


def _joint_kappa(
    indexed: dict[str, dict[str, dict[str, str]]],
    reviewers: Sequence[str],
    jointly_decided: Sequence[str],
) -> float | None:
    """Cohen's kappa for 2 reviewers, Fleiss' for 3+; None below 2 jointly decided rows."""
    if len(reviewers) < 2 or len(jointly_decided) < 2:
        return None
    if len(reviewers) == 2:
        a, b = reviewers
        return cohen_kappa(
            [_decision(indexed[a][i]) for i in jointly_decided],
            [_decision(indexed[b][i]) for i in jointly_decided],
        )
    counts = [
        [
            sum(1 for r in reviewers if _decision(indexed[r][i]) == label)
            for label in (ACCEPT, REJECT)
        ]
        for i in jointly_decided
    ]
    return fleiss_kappa(counts)


def _per_reviewer_stats(
    by_reviewer: dict[str, list[dict[str, str]]], reviewers: Sequence[str]
) -> dict[str, dict[str, int]]:
    return {
        r: {
            "decided": sum(1 for row in by_reviewer[r] if _decision(row) in (ACCEPT, REJECT)),
            "accepted": sum(1 for row in by_reviewer[r] if _decision(row) == ACCEPT),
            "rejected": sum(1 for row in by_reviewer[r] if _decision(row) == REJECT),
        }
        for r in reviewers
    }


def agreement_report(by_reviewer: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    """Inter-annotator agreement over the jointly decided rows.

    Cohen's kappa for exactly 2 reviewers, Fleiss' kappa for 3+; `kappa` is None until at
    least two jointly decided rows exist. Disagreement ids (differing decisions, or unanimous
    accepts whose edited answers differ) feed the adjudication draw.
    """
    reviewers = sorted(by_reviewer)
    indexed = {r: _rows_by_item(by_reviewer[r]) for r in reviewers}
    joint = _joint_item_ids(by_reviewer)
    jointly_decided, disagreements = _joint_decisions(indexed, reviewers, joint)
    observed = (
        (len(jointly_decided) - len(disagreements)) / len(jointly_decided)
        if jointly_decided
        else 0.0
    )
    return {
        "annotators": reviewers,
        "joint_items": len(joint),
        "jointly_decided": len(jointly_decided),
        "observed_agreement": observed,
        "kappa": _joint_kappa(indexed, reviewers, jointly_decided),
        "kappa_method": "cohen" if len(reviewers) == 2 else "fleiss",
        "disagreements": disagreements,
        "per_reviewer": _per_reviewer_stats(by_reviewer, reviewers),
    }


def write_agreement_report(base_ws: Path, report: dict[str, object]) -> Path:
    """Persist `agreement.json` beside the worksheets."""
    path = Path(base_ws).with_name(AGREEMENT_FILENAME)
    atomic_write_text(path, json.dumps(report, ensure_ascii=False, indent=2))
    return path


# --- adjudication worksheet -------------------------------------------------------------------


def prior_decisions_note(item_id: str, by_reviewer: dict[str, list[dict[str, str]]]) -> str:
    """Render every prior decision (with reject code / edit marker) for the adjudication card."""
    parts: list[str] = []
    for rid in sorted(by_reviewer):
        row = _rows_by_item(by_reviewer[rid]).get(item_id)
        if row is None:
            continue
        decision = _decision(row) or "undecided"
        suffix = ""
        code = (row.get("reject_code") or "").strip()
        if decision == REJECT and code:
            suffix = f":{code}"
        elif decision == ACCEPT and _edit(row):
            suffix = f":edit={_edit(row)}"
        parts.append(f"{rid}={decision}{suffix}")
    return ";".join(parts)


def build_adjudication_worksheet(
    base_ws: Path, by_reviewer: dict[str, list[dict[str, str]]], disagreements: Sequence[str]
) -> tuple[Path, int]:
    """Write `adjudication.csv` holding EXACTLY the disagreement rows.

    Read-only context comes from the first reviewer's copy (all copies are identical there);
    the human columns start blank for a fresh, independent decision; `prior_decisions` carries
    every reviewer's verdict forward. Rebuilding preserves adjudicator decisions already made
    (merged back by item id) so re-running after more review never loses work.
    """
    path = Path(base_ws).with_name(ADJUDICATION_FILENAME)
    reviewers = sorted(by_reviewer)
    if not reviewers:
        raise ValueError("no reviewer worksheets to adjudicate")
    first = _rows_by_item(by_reviewer[reviewers[0]])

    existing: dict[str, dict[str, str]] = {}
    if path.is_file():
        prior_rows, _ = load_worksheet(path)
        existing = _rows_by_item(prior_rows)

    fieldnames = worksheet_fieldnames()
    if PRIOR_DECISIONS_COL not in fieldnames:
        fieldnames.append(PRIOR_DECISIONS_COL)
    rows: list[dict[str, str]] = []
    for item_id in disagreements:
        source = first.get(item_id)
        if source is None:
            continue
        row = dict(source)
        for col in HUMAN_COLS:
            row[col] = ""
        row[REVIEWER_COL] = ADJUDICATOR_ID
        row[PRIOR_DECISIONS_COL] = prior_decisions_note(item_id, by_reviewer)
        carried = existing.get(item_id)
        if carried is not None:
            for col in HUMAN_COLS:
                row[col] = carried.get(col, "")
        rows.append(row)
    write_worksheet_rows(path, rows, fieldnames)
    return path, len(rows)


def run_adjudicate(bundle: Path, base_ws: Path | None = None) -> int:
    """The `adjudicate` subcommand: agreement report + adjudication worksheet for a bundle."""
    bundle = Path(bundle)
    base = Path(base_ws) if base_ws is not None else bundle / "verify_sample.csv"
    worksheets = reviewer_worksheets_from_manifest(base)
    if len(worksheets) < 2:
        _LOG.error(
            "[verify] no multi-reviewer worksheets recorded beside %s -- "
            "run make verify-sample VERIFY_ANNOTATORS=<k> first",
            base,
        )
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
    adj_path, n_rows = build_adjudication_worksheet(base, by_reviewer, disagreements)
    if n_rows:
        _LOG.info(
            "[verify] %d disagreement row(s) -> %s; review: make verify-review VERIFY_WS=%s",
            n_rows,
            adj_path,
            adj_path,
        )
    else:
        _LOG.info("[verify] no disagreements -- %s is empty; proceed to verify-accept", adj_path)
    return 0


# --- consensus (what acceptance actually scores) ------------------------------------------------


def consensus_rows(
    by_reviewer: dict[str, list[dict[str, str]]],
    adjudication: Sequence[dict[str, str]] = (),
) -> list[dict[str, str]]:
    """Merge per-reviewer worksheets (+ the adjudication pass) into one effective row set.

    Per item: an adjudicated decision overrides everything; a unanimous decision stands (taking
    the first reviewer's human columns); anything else -- a reviewer still undecided, or an
    unadjudicated disagreement -- yields an UNDECIDED row, so `acceptance_report` blocks on it.
    """
    reviewers = sorted(by_reviewer)
    if not reviewers:
        return []
    indexed = {r: _rows_by_item(by_reviewer[r]) for r in reviewers}
    adjudicated = _rows_by_item(adjudication)
    merged: list[dict[str, str]] = []
    for item_id in _joint_item_ids(by_reviewer):
        adj_row = adjudicated.get(item_id)
        if adj_row is not None and _decision(adj_row) in (ACCEPT, REJECT):
            merged.append(dict(adj_row))
            continue
        merged.append(_reviewer_consensus_row(indexed, reviewers, item_id))
    return merged


def _reviewer_consensus_row(
    indexed: dict[str, dict[str, dict[str, str]]], reviewers: list[str], item_id: str
) -> dict[str, str]:
    """A unanimous decision row (first reviewer's columns), else an UNDECIDED (blanked) row."""
    decisions = [_decision(indexed[r][item_id]) for r in reviewers]
    edits = [_edit(indexed[r][item_id]) for r in reviewers]
    row = dict(indexed[reviewers[0]][item_id])
    unanimous = all(d in (ACCEPT, REJECT) for d in decisions) and not _is_disagreement(
        decisions, edits
    )
    if not unanimous:
        for col in HUMAN_COLS:
            row[col] = ""
    return row


def resolve_multi_reviewer_rows(worksheet: Path) -> list[dict[str, str]] | None:
    """The consensus row set for a multi-reviewer bundle, or None for single-worksheet flows.

    `worksheet` is the BASE path (`<bundle>/verify_sample.csv`); when the sibling manifest
    records reviewer worksheets, acceptance scores the consensus (including `adjudication.csv`
    when present) instead of any single reviewer's sheet.
    """
    worksheets = reviewer_worksheets_from_manifest(Path(worksheet))
    if len(worksheets) < 2:
        return None
    by_reviewer = load_reviewer_worksheets(worksheets)
    adj_path = Path(worksheet).with_name(ADJUDICATION_FILENAME)
    adjudication: list[dict[str, str]] = []
    if adj_path.is_file():
        adjudication, _ = load_worksheet(adj_path)
    return consensus_rows(by_reviewer, adjudication)
