"""Human sample-verification of AI-drafted gold data (human verification gate -- codeable half).

The drafting pipeline (`llb.prep.frontier` / `llb.prep.ontology`) plus the second-frontier
cross-check (`llb.prep.cross_check`) produce an UNVERIFIED bundle (`goldset.jsonl` + a
self-contained `corpus/`, every item `verified=false`). human verification gate is the irreducibly-human gate:
draw a STRATIFIED sample, verify each item against the four checks (grounded / answerable +
non-circular / reference correct / planted labels match), accept if the error rate is within
tolerance, then flip the accepted items to `verified=true` THROUGH THE LEDGER -- never by
hand-editing the boolean (a reused id must re-adopt canonical content, not certify a changed one).

This module is the pure half -- stratification, deterministic sampling, the acceptance-sampling
arithmetic, worksheet I/O, and emitting an accepted-ledger bundle for `ingest_squad
--verified-goldset`. The interactive session lives in `verify_session.py` (mirroring how
`judge/calibration.py` pairs with `judge/rate.py`); it is imported lazily by the `review`
subcommand. Everything here needs no model, endpoint, or GPU, so it is fully unit-tested.
"""

import argparse
import csv
import io
import json
import logging
import random
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from llb.fsutil import atomic_write_text
from llb.goldset.schema import GoldItem, dump_goldset, load_goldset

_LOG = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 0.05  # accept a stratum/bundle whose sampled reject rate is <= this
GOLDSET_FILENAME = "goldset.jsonl"
# Draft bundles name their gold file differently: real-corpus drafts (frontier / ontology) write
# `goldset.jsonl`; the synthetic planter writes `planted_labels.jsonl`. We accept either.
GOLDSET_CANDIDATES = (GOLDSET_FILENAME, "planted_labels.jsonl")
PROVENANCE_FILENAME = "provenance.json"
CORPUS_DIRNAME = "corpus"
CROSS_CHECK_SUFFIX = ".cross_check.json"
SAMPLE_MANIFEST = "sample_manifest.json"
CONTEXT_CHARS = 400  # corpus window rendered on each side of a cited span

# The human-owned columns -- the four per-item checks, the accept/reject decision, a free note,
# and a status. `pass` / `fail` / "" for a check ("" = not yet checked; planted is "" for real
# items); decision is `accept` / `reject` / "". Everything else in the worksheet is read-only
# context the sampler fills in. Kept here so the session and the accept path share one schema.
CHECK_COLS = ["chk_grounded", "chk_answerable", "chk_reference", "chk_planted"]
HUMAN_COLS = [*CHECK_COLS, "decision", "human_note", "human_status"]

# Read-only cross-check columns (the second frontier's verdict). The analog of `judge_rating`:
# HIDDEN from the card by default so it cannot anchor the human; `--show-crosscheck` reveals it.
CROSS_CHECK_COLS = ["cc_grounded", "cc_non_circular", "cc_supported", "cc_answerable", "cc_note"]

WORKSHEET_COLS = [
    "item_id",
    "provenance",
    "split",
    "source_doc_id",
    "synthetic",
    "stratum",
    "question",
    "reference_answer",
    "span_doc_id",
    "span_text",
    "context",
    *CROSS_CHECK_COLS,
    *HUMAN_COLS,
]

PASS = "pass"
FAIL = "fail"
ACCEPT = "accept"
REJECT = "reject"
STATUS_PENDING = "pending"
STATUS_DECIDED = "decided"


@dataclass(frozen=True)
class VerificationRefStatus:
    """Validation result for an artifact used to stamp a category run as data-verified."""

    valid: bool
    path: Path
    kind: str
    reason: str = ""
    stats: dict[str, object] = field(default_factory=dict)
    instruction: str = ""


# --- bundle layout ------------------------------------------------------------------------


def find_goldset(bundle: Path) -> Path:
    """The bundle's gold file (`goldset.jsonl`, else `planted_labels.jsonl`)."""
    bundle = Path(bundle)
    for name in GOLDSET_CANDIDATES:
        candidate = bundle / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no gold file in {bundle} (looked for {', '.join(GOLDSET_CANDIDATES)})"
    )


def bundle_is_synthetic(bundle: Path) -> bool:
    """Whether the bundle is planted-synthetic (a BUNDLE-level fact in `provenance.json`).

    The synthetic flag is recorded once per bundle by the planter, not per item -- the canonical
    `GoldItem` provenance is `frontier-drafted` for both real and planted drafts. A bundle is
    uniformly real or synthetic, so the planted-labels check applies to all of its items or none.
    """
    meta = Path(bundle) / PROVENANCE_FILENAME
    if not meta.is_file():
        return False
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("synthetic", False))


# --- strata -------------------------------------------------------------------------------


def stratum_key(item: GoldItem) -> str:
    """The stratum an item belongs to: provenance x split x source doc.

    These are the axes actually present on a canonical `GoldItem` (the draft pipeline's free-form
    `kind`/`difficulty` tags are dropped on dump; synthetic is a bundle-level constant). Sampling
    per stratum keeps an error concentrated in one cell -- e.g. a single source doc -- from hiding
    behind a clean overall rate."""
    return f"{item.provenance}|{item.split}|{item.source_doc_id}"


def stratify(items: Sequence[GoldItem]) -> dict[str, list[GoldItem]]:
    """Group items by `stratum_key`, preserving input order within each stratum."""
    strata: dict[str, list[GoldItem]] = {}
    for item in items:
        strata.setdefault(stratum_key(item), []).append(item)
    return strata


def draw_stratified_sample(items: Sequence[GoldItem], n: int, *, seed: int = 13) -> list[GoldItem]:
    """Draw ~`n` items spread across strata (deterministic given `seed`).

    Allocates the budget proportionally to each stratum's size with a floor of one per
    non-empty stratum (so every cell is represented), shuffles within each stratum by `seed`,
    then returns the union in canonical (input) order. If `n` >= len(items), returns all items.
    """
    total = len(items)
    if n >= total:
        return list(items)
    strata = stratify(items)
    rng = random.Random(seed)
    picked: set[int] = set()
    index = {id(it): i for i, it in enumerate(items)}
    # Proportional allocation with a floor of 1, largest-remainder rounded up to `n`.
    quotas: dict[str, int] = {}
    for key, group in strata.items():
        quotas[key] = max(1, round(n * len(group) / total))
    for key, group in sorted(strata.items()):
        order = list(group)
        rng.shuffle(order)
        for it in order[: min(quotas[key], len(group))]:
            picked.add(index[id(it)])
    # Trim or top up to land near `n` while keeping determinism.
    ordered = sorted(picked)
    if len(ordered) > n:
        rng.shuffle(ordered)
        ordered = sorted(ordered[:n])
    return [items[i] for i in ordered]


# --- cross-check sidecar ------------------------------------------------------------------


def load_cross_check(bundle: Path) -> dict[str, dict[str, object]]:
    """Index any `*.cross_check.json` verdicts in the bundle by item id (empty if none).

    The verdict is read-only context for the human (the second frontier already ran these
    checks); it is shown only with `--show-crosscheck` so it never anchors the verification.
    """
    verdicts: dict[str, dict[str, object]] = {}
    for path in sorted(Path(bundle).glob(f"*{CROSS_CHECK_SUFFIX}")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for v in payload.get("verdicts", []):
            item_id = v.get("item_id")
            if item_id:
                verdicts[str(item_id)] = v
    return verdicts


# --- corpus windows -----------------------------------------------------------------------


def corpus_window(text: str, char_start: int, char_end: int, ctx: int = CONTEXT_CHARS) -> str:
    """Render the cited span inside its surrounding corpus text, the span delimited by >>><<<.

    The window is what the human reads to confirm grounding without leaving the tool; it is
    captured into the worksheet at sample time so the CSV stays a self-contained artifact.
    """
    lo = max(0, char_start - ctx)
    hi = min(len(text), char_end + ctx)
    before = text[lo:char_start]
    span = text[char_start:char_end]
    after = text[char_end:hi]
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(text) else ""
    return f"{prefix}{before}>>>{span}<<<{after}{suffix}"


def _corpus_text(corpus_root: Path, doc_id: str, cache: dict[str, str | None]) -> str | None:
    if doc_id not in cache:
        path = corpus_root / doc_id
        cache[doc_id] = path.read_text(encoding="utf-8") if path.is_file() else None
    return cache[doc_id]


# --- worksheet I/O (atomic, CSV-as-state -- mirrors judge/calibration.py) ------------------


def worksheet_fieldnames(existing: Sequence[str] | None = None) -> list[str]:
    """Canonical column order: keep any existing columns, then append missing `WORKSHEET_COLS`."""
    names = list(existing) if existing else []
    for col in WORKSHEET_COLS:
        if col not in names:
            names.append(col)
    return names


def load_worksheet(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Load a verification worksheet CSV into `(rows, fieldnames)` with every column present."""
    text = Path(path).read_text(encoding="utf-8")
    reader = csv.DictReader(text.splitlines())
    fieldnames = worksheet_fieldnames(reader.fieldnames)
    rows = [{name: (raw.get(name) or "") for name in fieldnames} for raw in reader]
    return rows, fieldnames


def _resolve_ref_path(ref: Path | str, base_dir: Path | None = None) -> Path:
    """Resolve an operator-provided verification reference without assuming an absolute root."""
    raw = Path(ref).expanduser()
    if raw.is_absolute():
        return raw
    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append(Path(base_dir) / raw)
    candidates.extend([Path.cwd() / raw, raw])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _stat_text(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


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


def _check_accepted_ledger_ref(path: Path) -> VerificationRefStatus:
    ledger = path / GOLDSET_FILENAME if path.is_dir() else path
    instruction = _accepted_ledger_instruction(path)
    try:
        items = load_goldset(ledger)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return VerificationRefStatus(
            False,
            ledger,
            "accepted_ledger",
            f"unreadable ledger: {exc}",
            instruction=instruction,
        )
    unverified_ids = [item.id for item in items if not item.verified]
    stats: dict[str, object] = {
        "items": len(items),
        "verified": len(items) - len(unverified_ids),
        "unverified": len(unverified_ids),
        "unverified_item_ids": unverified_ids[:10],
    }
    if not items:
        return VerificationRefStatus(
            False,
            ledger,
            "accepted_ledger",
            "ledger has no items",
            stats=stats,
            instruction=instruction,
        )
    if unverified_ids:
        return VerificationRefStatus(
            False,
            ledger,
            "accepted_ledger",
            "ledger contains unverified item(s)",
            stats=stats,
            instruction=instruction,
        )
    return VerificationRefStatus(True, ledger, "accepted_ledger", stats=stats)


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
    if path.name == GOLDSET_FILENAME or path.suffix.lower() == ".jsonl":
        return _check_accepted_ledger_ref(path)
    return VerificationRefStatus(
        False,
        path,
        "unknown",
        "unsupported verification reference",
        instruction=_generic_verification_instruction(path),
    )


def write_worksheet_rows(
    out_path: Path, rows: Sequence[dict[str, str]], fieldnames: Sequence[str] | None = None
) -> int:
    """Atomically (re)write the whole worksheet, preserving column order (crash-safe resume)."""
    columns = list(fieldnames) if fieldnames else list(WORKSHEET_COLS)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in columns})
    atomic_write_text(Path(out_path), buf.getvalue())
    return len(rows)


def _row_for(
    item: GoldItem,
    corpus_root: Path,
    cache: dict[str, str | None],
    verdict: dict[str, object],
    *,
    synthetic: bool,
) -> dict[str, str]:
    span = item.source_spans[0]
    text = _corpus_text(corpus_root, span.doc_id, cache)
    context = (
        corpus_window(text, span.char_start, span.char_end) if text is not None else "(missing doc)"
    )

    def _flag(key: str) -> str:
        value = verdict.get(key)
        return "" if value is None else ("true" if value else "false")

    return {
        "item_id": item.id,
        "provenance": item.provenance,
        "split": item.split,
        "source_doc_id": item.source_doc_id,
        "synthetic": "true" if synthetic else "false",
        "stratum": stratum_key(item),
        "question": item.question,
        "reference_answer": item.reference_answer,
        "span_doc_id": span.doc_id,
        "span_text": span.text,
        "context": context,
        "cc_grounded": _flag("grounded"),
        "cc_non_circular": _flag("non_circular"),
        "cc_supported": _flag("supported"),
        "cc_answerable": _flag("answerable"),
        "cc_note": str(verdict.get("note", "")),
    }


def build_sample_worksheet(
    bundle: Path, out_path: Path, *, n: int, seed: int = 13
) -> tuple[int, dict[str, int]]:
    """Draw a stratified sample from a draft bundle and write the verification worksheet.

    Returns `(sample_size, strata_sizes)` and writes a `sample_manifest.json` beside the
    worksheet documenting the bundle, seed, requested/actual size, and per-stratum counts (the
    "document the size + strata" half of the procedure). Human columns are left blank.
    """
    bundle = Path(bundle)
    items = load_goldset(find_goldset(bundle))
    synthetic = bundle_is_synthetic(bundle)
    sample = draw_stratified_sample(items, n, seed=seed)
    verdicts = load_cross_check(bundle)
    corpus_root = bundle / CORPUS_DIRNAME
    cache: dict[str, str | None] = {}
    rows = [
        _row_for(it, corpus_root, cache, verdicts.get(it.id, {}), synthetic=synthetic)
        for it in sample
    ]
    write_worksheet_rows(out_path, rows)

    strata_sizes: dict[str, int] = {}
    for it in sample:
        strata_sizes[stratum_key(it)] = strata_sizes.get(stratum_key(it), 0) + 1
    manifest = {
        "bundle": str(bundle),
        "worksheet": str(out_path),
        "synthetic": synthetic,
        "seed": seed,
        "requested": n,
        "sample_size": len(sample),
        "population": len(items),
        "strata": strata_sizes,
    }
    atomic_write_text(
        Path(out_path).with_name(SAMPLE_MANIFEST),
        json.dumps(manifest, ensure_ascii=False, indent=2),
    )
    return len(sample), strata_sizes


# --- acceptance arithmetic ----------------------------------------------------------------


def _is_decided(row: dict[str, str]) -> bool:
    return (row.get("decision") or "").strip() in (ACCEPT, REJECT)


def _failed_any_check(row: dict[str, str]) -> bool:
    return any((row.get(col) or "").strip() == FAIL for col in CHECK_COLS)


def acceptance_report(
    rows: Sequence[dict[str, str]], tolerance: float = DEFAULT_TOLERANCE
) -> dict[str, object]:
    """Acceptance-sampling summary: per-stratum + overall decided/reject counts and pass/fail.

    A decided item is a `reject` defect; the reject RATE over decided items is compared to
    `tolerance`. A stratum (and the bundle) PASSES when its rate is within tolerance. Items with
    a failed check but no explicit decision are surfaced as `undecided_with_failures` so nothing
    silently slips through. Pure -- the caller decides what to emit.
    """
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
    for cell in per_stratum.values():
        d = cell["decided"]
        cell["reject_rate"] = (cell["rejected"] / d) if d else 0.0
        cell["passed"] = float(cell["reject_rate"] <= tolerance)
    overall_rate = (rejected / decided) if decided else 0.0
    return {
        "tolerance": tolerance,
        "n": len(rows),
        "decided": decided,
        "rejected": rejected,
        "accepted": decided - rejected,
        "undecided": len(rows) - decided,
        "undecided_with_failures": undecided_with_failures,
        "reject_rate": overall_rate,
        "passed": overall_rate <= tolerance and decided > 0,
        "per_stratum": per_stratum,
    }


# --- accepted-ledger emission (the flip is an ADOPTION, not a boolean edit) -----------------


def accepted_ids(rows: Sequence[dict[str, str]]) -> list[str]:
    """Item ids the human explicitly accepted."""
    return [
        (row.get("item_id") or "").strip()
        for row in rows
        if (row.get("decision") or "").strip() == ACCEPT and (row.get("item_id") or "").strip()
    ]


def emit_accepted_ledger(bundle: Path, accepted: Sequence[str], out_dir: Path) -> int:
    """Write an accepted-ledger bundle (`goldset.jsonl` verified=true + sibling `corpus/`).

    The accepted items are taken VERBATIM from the draft bundle (canonical content + grounded
    spans) with `verified` flipped to true, and every corpus doc they depend on is copied so the
    ledger is self-contained. Feeding this to `ingest_squad --verified-goldset <out_dir>/goldset.jsonl`
    re-adopts those ids by REPLACEMENT, which is what stops a reused id from certifying changed
    content. We never hand-edit the boolean in the draft bundle.
    """
    bundle = Path(bundle)
    out_dir = Path(out_dir)
    keep = set(accepted)
    src_corpus = bundle / CORPUS_DIRNAME
    dst_corpus = out_dir / CORPUS_DIRNAME
    verified: list[GoldItem] = []
    doc_ids: set[str] = set()
    for item in load_goldset(find_goldset(bundle)):
        if item.id not in keep:
            continue
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


def _log_report(report: dict[str, object]) -> None:
    _LOG.info(
        "[verify] decided=%s accepted=%s rejected=%s reject_rate=%.3f tolerance=%s -> %s",
        report["decided"],
        report["accepted"],
        report["rejected"],
        report["reject_rate"],
        report["tolerance"],
        "PASS" if report["passed"] else "FAIL",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="human verification gate human sample-verification of draft data.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sa = sub.add_parser("sample", help="draw a stratified sample from a draft bundle -> worksheet")
    sa.add_argument(
        "--bundle", required=True, type=Path, help="draft dir (goldset.jsonl + corpus/)"
    )
    sa.add_argument("--out", required=True, type=Path, help="verification worksheet CSV to write")
    sa.add_argument("-n", "--size", type=int, default=30, help="target sample size")
    sa.add_argument("--seed", type=int, default=13)

    rv = sub.add_parser("review", help="interactively verify the sampled items")
    rv.add_argument("--worksheet", required=True, type=Path)
    rv.add_argument("--start", type=int, default=None, help="begin at this 1-based item")
    rv.add_argument(
        "--show-crosscheck",
        action="store_true",
        help="reveal the second-frontier verdict (post-hoc only; anchors the human -- off by default)",
    )
    rv.add_argument("--clear", action="store_true", help="wipe ALL human columns first (gated)")

    ac = sub.add_parser("accept", help="acceptance report + emit the accepted-ledger bundle")
    ac.add_argument("--worksheet", required=True, type=Path)
    ac.add_argument(
        "--bundle", required=True, type=Path, help="the draft bundle the sample came from"
    )
    ac.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="accepted-ledger dir (default: <bundle>/accepted)",
    )
    ac.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)

    args = parser.parse_args(argv)

    if args.cmd == "sample":
        size, strata = build_sample_worksheet(args.bundle, args.out, n=args.size, seed=args.seed)
        _LOG.info("[verify] sampled %d item(s) across %d strata -> %s", size, len(strata), args.out)
        _LOG.info("[verify] review: make verify-review WS=%s", args.out)
        return 0

    if args.cmd == "review":
        from llb.goldset.verify_session import run_session

        run_session(
            args.worksheet,
            start=args.start,
            show_crosscheck=args.show_crosscheck,
            clear=args.clear,
        )
        return 0

    rows, _ = load_worksheet(args.worksheet)
    report = acceptance_report(rows, args.tolerance)
    _log_report(report)
    accepted = accepted_ids(rows)
    if not accepted:
        _LOG.info("[verify] no accepted items -- nothing to flip; resolve the sample first")
        return 0 if report["passed"] else 1
    out_dir = args.out_dir or (Path(args.bundle) / "accepted")
    n = emit_accepted_ledger(args.bundle, accepted, out_dir)
    _LOG.info("[verify] wrote %d accepted item(s) -> %s", n, out_dir / GOLDSET_FILENAME)
    _LOG.info(
        "[verify] flip via the ledger: python -m llb.prep.ingest_squad "
        "--squad-json <source> --verified-goldset %s",
        out_dir / GOLDSET_FILENAME,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    from llb.runtime import run

    sys.exit(run(main))
