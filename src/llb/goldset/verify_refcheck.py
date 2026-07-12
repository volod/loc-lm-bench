"""Verification-reference validation for `--data-verified` (the ledger-ref half of `verify.py`).

Validates the artifact an operator points `--verification-ref` at before it may stamp a category
run as data-verified: a reviewed `verify_sample.csv` whose rows are all decided within tolerance,
a `sample_manifest.json` that points to such a worksheet, or an accepted-ledger dir / `goldset.jsonl`
/ `chains.jsonl` whose entries are all `verified=true`. Renders a failing check with stats and the
operator's next steps. Pure -- no model, endpoint, or GPU. Shared constants, `VerificationRefStatus`,
and the ref-path/stat helpers live in `verify.py`, which re-exports these names so
`llb.goldset.verify.<name>` keeps working.
"""

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

from llb.goldset.chains import CHAINS_FILENAME, load_chains
from llb.goldset.schema import load_goldset
from llb.goldset.verify_base import (
    DEFAULT_TOLERANCE,
    GOLDSET_FILENAME,
    SAMPLE_MANIFEST,
    VerificationRefStatus,
    _resolve_ref_path,
    _stat_text,
    load_worksheet,
)
from llb.goldset.verify_acceptance import acceptance_report

# --- worksheet / manifest hints -------------------------------------------------------------


def _worksheet_bundle_hint(path: Path) -> Path | None:
    manifest = path.with_name(SAMPLE_MANIFEST)
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    bundle = payload.get("bundle")
    if isinstance(bundle, str) and bundle:
        return _resolve_ref_path(bundle, base_dir=manifest.parent)
    return None


def _worksheet_instruction(path: Path, *, bundle: Path | None = None) -> str:
    bundle_arg = str(bundle) if bundle is not None else "<bundle>"
    return "\n".join(
        [
            f"Review every undecided row: make verify-review VERIFY_WS={path}",
            (
                "Recompute acceptance and emit the accepted ledger: "
                f"make verify-accept BUNDLE={bundle_arg} VERIFY_WS={path}"
            ),
            (f"Rerun the category command with --data-verified --verification-ref {path}"),
            (
                "You may also use the sibling sample_manifest.json or the accepted-ledger "
                "directory as --verification-ref after they pass this check."
            ),
        ]
    )


def _accepted_ledger_instruction(path: Path) -> str:
    return "\n".join(
        [
            (
                "Use an accepted ledger only after make verify-accept emits a clean "
                "goldset.jsonl whose items are all verified=true."
            ),
            (
                "If this ledger came from a draft bundle, rerun: "
                "make verify-accept BUNDLE=<bundle> VERIFY_WS=<bundle>/verify_sample.csv"
            ),
            (f"Then rerun the category command with --data-verified --verification-ref {path}"),
        ]
    )


def _generic_verification_instruction(ref: Path | None = None) -> str:
    suffix = f" {ref}" if ref is not None else ""
    return "\n".join(
        [
            "Create or point at one of the accepted human verification gate artifacts:",
            "make verify-sample BUNDLE=<bundle> VERIFY_N=<n>",
            "make verify-review VERIFY_WS=<bundle>/verify_sample.csv",
            "make verify-accept BUNDLE=<bundle> VERIFY_WS=<bundle>/verify_sample.csv",
            (
                "Rerun the category command with --data-verified --verification-ref"
                f"{suffix if suffix else ' <artifact>'}"
            ),
            (
                "Accepted artifact forms: reviewed verify_sample.csv, sample_manifest.json "
                "that points to it, or accepted/ or accepted/goldset.jsonl."
            ),
        ]
    )


# --- stats rendering ------------------------------------------------------------------------


def _worksheet_stats(report: dict[str, object]) -> dict[str, object]:
    per_stratum = cast(dict[str, dict[str, float]], report["per_stratum"])
    failing: list[dict[str, object]] = []
    for key, cell in sorted(per_stratum.items()):
        if not bool(cell["passed"]):
            failing.append(
                {
                    "stratum": key,
                    "decided": int(cell["decided"]),
                    "rejected": int(cell["rejected"]),
                    "reject_rate": float(cell["reject_rate"]),
                }
            )
    return {
        "n": report["n"],
        "decided": report["decided"],
        "accepted": report["accepted"],
        "rejected": report["rejected"],
        "undecided": report["undecided"],
        "undecided_with_failures": report["undecided_with_failures"],
        "reject_rate": report["reject_rate"],
        "tolerance": report["tolerance"],
        "failing_strata": failing,
    }


def _format_verification_stats(stats: dict[str, object]) -> list[str]:
    order = [
        "n",
        "decided",
        "accepted",
        "rejected",
        "undecided",
        "undecided_with_failures",
        "reject_rate",
        "tolerance",
        "items",
        "chains",
        "verified",
        "unverified",
    ]
    lines = [f"{key}: {_stat_text(stats[key])}" for key in order if key in stats]
    failing = stats.get("failing_strata")
    if isinstance(failing, list) and failing:
        lines.append("failing_strata:")
        for cell in failing[:10]:
            if not isinstance(cell, dict):
                continue
            lines.append(
                "  "
                f"{cell.get('stratum', '(none)')}: "
                f"{cell.get('rejected', 0)}/{cell.get('decided', 0)} rejected, "
                f"reject_rate={_stat_text(cell.get('reject_rate', 0.0))}"
            )
        if len(failing) > 10:
            lines.append(f"  ... {len(failing) - 10} more failing strata")
    samples = stats.get("unverified_item_ids")
    if isinstance(samples, list) and samples:
        lines.append("unverified_item_ids: " + ", ".join(str(item) for item in samples[:10]))
    worksheet = stats.get("worksheet")
    if worksheet:
        lines.append(f"worksheet: {worksheet}")
    return lines


def format_verification_status(status: VerificationRefStatus) -> str:
    """Render a failed verification-reference check with stats and operator next steps."""
    if status.valid:
        return f"verification reference is valid: {status.path} ({status.kind})"
    lines = [
        "verification reference cannot be used with --data-verified",
        f"path: {status.path}",
        f"kind: {status.kind}",
        f"reason: {status.reason or 'unknown'}",
    ]
    stat_lines = _format_verification_stats(status.stats)
    if stat_lines:
        lines.append("stats:")
        lines.extend(f"  {line}" for line in stat_lines)
    instruction = status.instruction or _generic_verification_instruction(status.path)
    lines.append("next steps:")
    lines.extend(f"  {line}" for line in instruction.splitlines() if line.strip())
    return "\n".join(lines)


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


def _ledger_ref_status(
    ledger: Path,
    kind: str,
    *,
    noun: str,
    stats_key: str,
    unreadable_label: str,
    load_unverified: Callable[[Path], tuple[int, list[str]]],
    instruction: str,
) -> VerificationRefStatus:
    """Shared flat/chain ledger check: readable, non-empty, and every entry verified."""
    try:
        total, unverified_ids = load_unverified(ledger)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return VerificationRefStatus(
            False, ledger, kind, f"unreadable {unreadable_label}: {exc}", instruction=instruction
        )
    stats: dict[str, object] = {
        stats_key: total,
        "verified": total - len(unverified_ids),
        "unverified": len(unverified_ids),
        "unverified_item_ids": unverified_ids[:10],
    }
    if not total:
        return VerificationRefStatus(
            False,
            ledger,
            kind,
            f"ledger has no {stats_key}",
            stats=stats,
            instruction=instruction,
        )
    if unverified_ids:
        return VerificationRefStatus(
            False,
            ledger,
            kind,
            f"ledger contains unverified {noun}(s)",
            stats=stats,
            instruction=instruction,
        )
    return VerificationRefStatus(True, ledger, kind, stats=stats)


def _load_unverified_chains(ledger: Path) -> tuple[int, list[str]]:
    chains = load_chains(ledger)
    return len(chains), [chain.chain_id for chain in chains if not chain.verified]


def _load_unverified_items(ledger: Path) -> tuple[int, list[str]]:
    items = load_goldset(ledger)
    return len(items), [item.id for item in items if not item.verified]


def _check_accepted_ledger_ref(path: Path) -> VerificationRefStatus:
    ledger = path / GOLDSET_FILENAME if path.is_dir() else path
    instruction = _accepted_ledger_instruction(path)
    chain_ledger = path / CHAINS_FILENAME if path.is_dir() else path
    if not ledger.is_file() and chain_ledger.name == CHAINS_FILENAME and chain_ledger.is_file():
        return _ledger_ref_status(
            chain_ledger,
            "accepted_chain_ledger",
            noun="chain",
            stats_key="chains",
            unreadable_label="chain ledger",
            load_unverified=_load_unverified_chains,
            instruction=instruction,
        )
    return _ledger_ref_status(
        ledger,
        "accepted_ledger",
        noun="item",
        stats_key="items",
        unreadable_label="ledger",
        load_unverified=_load_unverified_items,
        instruction=instruction,
    )


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
