"""Focused verify ref format implementation."""

import json
from pathlib import Path
from typing import cast
from llb.goldset.verify_base import (
    SAMPLE_MANIFEST,
    VerificationRefStatus,
    _resolve_ref_path,
    _stat_text,
)


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
