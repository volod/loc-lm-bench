"""Verification-reference validation for `--data-verified`.

Validates the artifact an operator points `--verification-ref` at before it may stamp a category
run as data-verified: a reviewed `verify_sample.csv` whose rows are all decided within tolerance,
a `sample_manifest.json` that points to such a worksheet, or an accepted-ledger dir / `goldset.jsonl`
/ `chains.jsonl` whose entries are all `verified=true`. Renders a failing check with stats and the
operator's next steps. Shared constants and `VerificationRefStatus` live in `verify_base.py`.
"""

import csv
import json
from pathlib import Path

from llb.goldset.chains import CHAINS_FILENAME
from llb.goldset.verify_base import (
    DEFAULT_TOLERANCE,
    GOLDSET_FILENAME,
    SAMPLE_MANIFEST,
    VerificationRefStatus,
    _resolve_ref_path,
    load_worksheet,
)
from llb.goldset.verify_acceptance import acceptance_report
from llb.goldset.verify_ref_format import (
    _generic_verification_instruction,
    _worksheet_bundle_hint,
    _worksheet_instruction,
    _worksheet_stats,
)
from llb.goldset.verify_ref_ledger import _check_accepted_ledger_ref

# --- worksheet / manifest hints -------------------------------------------------------------


# --- stats rendering ------------------------------------------------------------------------


# --- reference checks -----------------------------------------------------------------------


def _check_worksheet_ref(path: Path, tolerance: float) -> VerificationRefStatus:
    bundle = _worksheet_bundle_hint(path)
    instruction = _worksheet_instruction(path, bundle=bundle)
    try:
        rows, _ = load_worksheet(path)
        report = acceptance_report(rows, tolerance=tolerance)
    except (OSError, csv.Error) as exc:
        return VerificationRefStatus(
            False,
            path,
            "worksheet",
            f"unreadable worksheet: {exc}",
            instruction=instruction,
        )
    stats = _worksheet_stats(report)
    if not rows:
        return VerificationRefStatus(
            False,
            path,
            "worksheet",
            "worksheet has no rows",
            stats=stats,
            instruction=instruction,
        )
    if report["undecided"]:
        return VerificationRefStatus(
            False,
            path,
            "worksheet",
            f"worksheet has {report['undecided']} undecided row(s)",
            stats=stats,
            instruction=instruction,
        )
    if report["undecided_with_failures"]:
        return VerificationRefStatus(
            False,
            path,
            "worksheet",
            f"worksheet has {report['undecided_with_failures']} undecided failed check(s)",
            stats=stats,
            instruction=instruction,
        )
    if not report["passed"]:
        return VerificationRefStatus(
            False,
            path,
            "worksheet",
            f"reject rate {report['reject_rate']:.3f} exceeds tolerance {tolerance:g}",
            stats=stats,
            instruction=instruction,
        )
    return VerificationRefStatus(True, path, "worksheet", stats=stats)


def _check_manifest_ref(path: Path, tolerance: float) -> VerificationRefStatus:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return VerificationRefStatus(
            False,
            path,
            "sample_manifest",
            f"unreadable manifest: {exc}",
            instruction=_generic_verification_instruction(path),
        )
    bundle = payload.get("bundle")
    bundle_path = (
        _resolve_ref_path(bundle, base_dir=path.parent) if isinstance(bundle, str) else None
    )
    worksheet = payload.get("worksheet")
    if not worksheet:
        return VerificationRefStatus(
            False,
            path,
            "sample_manifest",
            "manifest lacks worksheet",
            instruction=_generic_verification_instruction(path),
        )
    worksheet_path = _resolve_ref_path(str(worksheet), base_dir=path.parent)
    status = _check_worksheet_ref(worksheet_path, tolerance)
    if not status.valid:
        stats = dict(status.stats)
        stats["worksheet"] = str(worksheet_path)
        return VerificationRefStatus(
            False,
            path,
            "sample_manifest",
            f"worksheet invalid: {status.reason}",
            stats=stats,
            instruction=status.instruction
            or _worksheet_instruction(worksheet_path, bundle=bundle_path),
        )
    stats = dict(status.stats)
    stats["worksheet"] = str(worksheet_path)
    return VerificationRefStatus(True, path, "sample_manifest", stats=stats)


def check_verification_ref(
    ref: Path | str,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    base_dir: Path | None = None,
) -> VerificationRefStatus:
    """Validate a verification artifact before it can stamp a run as data-verified.

    Accepted forms:
    - reviewed `verify_sample.csv` with every sampled row decided and reject rate within tolerance;
    - `sample_manifest.json` whose `worksheet` points to such a reviewed worksheet;
    - accepted-ledger dir or `goldset.jsonl` whose items are all `verified=true`.
    """
    path = _resolve_ref_path(ref, base_dir=base_dir)
    if not path.exists():
        return VerificationRefStatus(
            False,
            path,
            "missing",
            "verification reference not found",
            instruction=_generic_verification_instruction(path),
        )
    if path.is_dir():
        if (path / GOLDSET_FILENAME).is_file():
            return _check_accepted_ledger_ref(path)
        if (path / CHAINS_FILENAME).is_file():
            return _check_accepted_ledger_ref(path)
        if (path / SAMPLE_MANIFEST).is_file():
            return _check_manifest_ref(path / SAMPLE_MANIFEST, tolerance)
        if (path / "verify_sample.csv").is_file():
            return _check_worksheet_ref(path / "verify_sample.csv", tolerance)
        return VerificationRefStatus(
            False,
            path,
            "directory",
            "directory lacks accepted ledger or verification worksheet",
            instruction=_generic_verification_instruction(path),
        )
    if path.name == SAMPLE_MANIFEST:
        return _check_manifest_ref(path, tolerance)
    if path.suffix.lower() == ".csv":
        return _check_worksheet_ref(path, tolerance)
    if path.name in (GOLDSET_FILENAME, CHAINS_FILENAME) or path.suffix.lower() == ".jsonl":
        return _check_accepted_ledger_ref(path)
    return VerificationRefStatus(
        False,
        path,
        "unknown",
        "unsupported verification reference",
        instruction=_generic_verification_instruction(path),
    )
