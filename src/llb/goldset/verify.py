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
import re
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from llb.core.fsutil import atomic_write_text
from llb.goldset.chains import (
    CHAINS_FILENAME,
    ChainItem,
    chain_stratum_key,
    dump_chains,
    load_chains,
)
from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset, load_goldset
from llb.rag.page_metadata import intersect_pages, load_page_citations

_LOG = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 0.05  # accept a stratum/bundle whose sampled reject rate is <= this
GOLDSET_FILENAME = "goldset.jsonl"
# Draft bundles name their gold file differently: real-corpus drafts (frontier / ontology) write
# `goldset.jsonl`; the synthetic planter writes `planted_labels.jsonl`. We accept either.
GOLDSET_CANDIDATES = (GOLDSET_FILENAME, "planted_labels.jsonl")
PROVENANCE_FILENAME = "provenance.json"
CORPUS_DIRNAME = "corpus"
KIND_AUTO = "auto"
KIND_GOLDSET = "goldset"
KIND_CHAINS = "chains"
SAMPLE_KINDS = (KIND_AUTO, KIND_GOLDSET, KIND_CHAINS)
CROSS_CHECK_SUFFIX = ".cross_check.json"
SAMPLE_MANIFEST = "sample_manifest.json"
REJECTION_REASONS_FILENAME = "rejection_reasons.json"
CONTEXT_CHARS = 400  # corpus window rendered on each side of a cited span

# Bundle files that may carry a per-item `retrieval_rank` (rows keyed by `id`): the ontology
# lane's needle report and the external-import provenance sidecar. Missing files are fine.
RETRIEVAL_RANK_SOURCES = ("needle_items.jsonl", "item_provenance.jsonl")

# The human-owned columns -- the four per-item checks, the accept/reject decision, a coded
# rejection reason, an optional edited answer (must re-ground before it can certify), a free
# note, and a status. `pass` / `fail` / "" for a check ("" = not yet checked; planted is "" for
# real items); decision is `accept` / `reject` / "". Everything else in the worksheet is
# read-only context the sampler fills in. Kept here so the session and the accept path share
# one schema.
CHECK_COLS = ["chk_grounded", "chk_answerable", "chk_reference", "chk_planted"]
HUMAN_COLS = [*CHECK_COLS, "decision", "reject_code", "edited_answer", "human_note", "human_status"]

# Read-only cross-check columns (the second frontier's verdict). The analog of `judge_rating`:
# HIDDEN from the card by default so it cannot anchor the human; `--show-crosscheck` reveals it.
CROSS_CHECK_COLS = ["cc_grounded", "cc_non_circular", "cc_supported", "cc_answerable", "cc_note"]

# Sampler-owned reviewer id for multi-annotator worksheets (see `verify_multi.py`). Appended
# LAST so older worksheets stay column-compatible; blank on single-reviewer worksheets.
REVIEWER_COL = "reviewer_id"

WORKSHEET_COLS = [
    "item_kind",
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
    "retrieval_rank",
    "page_citation",
    "chain_steps",
    *CROSS_CHECK_COLS,
    *HUMAN_COLS,
    REVIEWER_COL,
]

# Acceptance-arithmetic policies (`--policy` on the accept subcommand):
# - global: one reject rate over all decided rows vs the tolerance (the original rule);
# - per-stratum: EVERY stratum must be within its own tolerance (overridable per stratum);
# - weighted: confidence-weighted reject rate -- a reject on a row the automated signals
#   rated confident counts more, because it means those signals cannot be trusted either.
POLICY_GLOBAL = "global"
POLICY_PER_STRATUM = "per-stratum"
POLICY_WEIGHTED = "weighted"
ACCEPT_POLICIES = (POLICY_GLOBAL, POLICY_PER_STRATUM, POLICY_WEIGHTED)

PASS = "pass"
FAIL = "fail"
ACCEPT = "accept"
REJECT = "reject"
STATUS_PENDING = "pending"
STATUS_DECIDED = "decided"

# Coded rejection reasons, exported to `rejection_reasons.json` so the drafting pipeline can read
# WHY items were rejected and tighten its prompts. The first four mirror the four checks (a
# failed check infers its code when the reviewer rejects without naming one).
CHECK_REJECT_CODES = {
    "chk_grounded": "ungrounded",
    "chk_answerable": "circular",
    "chk_reference": "wrong_reference",
    "chk_planted": "label_mismatch",
}
REJECT_CODES = (*CHECK_REJECT_CODES.values(), "bad_question", "other")


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


def find_chains(bundle: Path) -> Path:
    """The bundle's chain file (`chains.jsonl`)."""
    path = Path(bundle) / CHAINS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no chain file in {bundle} (looked for {CHAINS_FILENAME})")
    return path


def resolve_sample_kind(bundle: Path, requested: str = KIND_AUTO) -> str:
    """Resolve auto/goldset/chains to the concrete worksheet kind for this bundle."""
    requested = requested.strip().lower() or KIND_AUTO
    if requested not in SAMPLE_KINDS:
        raise ValueError(f"unknown verification sample kind {requested!r}")
    bundle = Path(bundle)
    if requested == KIND_CHAINS:
        find_chains(bundle)
        return KIND_CHAINS
    if requested == KIND_GOLDSET:
        find_goldset(bundle)
        return KIND_GOLDSET
    if (bundle / CHAINS_FILENAME).is_file():
        return KIND_CHAINS
    find_goldset(bundle)
    return KIND_GOLDSET


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


def stratum_quotas(strata_sizes: dict[str, int], n: int) -> dict[str, int]:
    """Exact per-stratum allocation summing to `min(n, population)` (deterministic).

    Floor of one per non-empty stratum first (largest strata first when `n` cannot cover them
    all), then a largest-remainder top-up distributes the remaining budget proportionally,
    each stratum capped at its own size -- so proportional rounding can neither overshoot nor
    undershoot the requested sample size (verify-sample-exact-allocation).
    """
    total = sum(strata_sizes.values())
    budget = min(n, total)
    quotas = {key: 0 for key in strata_sizes}
    allocated = 0
    for key in sorted(strata_sizes, key=lambda k: (-strata_sizes[k], k)):
        if allocated >= budget:
            break
        if strata_sizes[key] > 0:
            quotas[key] = 1
            allocated += 1
    while allocated < budget:
        open_keys = [key for key in strata_sizes if quotas[key] < strata_sizes[key]]
        winner = max(
            sorted(open_keys),
            key=lambda k: (n * strata_sizes[k] / total) - quotas[k],
        )
        quotas[winner] += 1
        allocated += 1
    return quotas


def draw_stratified_sample(items: Sequence[GoldItem], n: int, *, seed: int = 13) -> list[GoldItem]:
    """Draw exactly `min(n, population)` items spread across strata (deterministic given `seed`).

    Allocates the budget with `stratum_quotas` (floor of one per non-empty stratum, exact
    largest-remainder proportional top-up), shuffles within each stratum by `seed`, then
    returns the union in canonical (input) order. If `n` >= len(items), returns all items.
    """
    total = len(items)
    if n >= total:
        return list(items)
    strata = stratify(items)
    rng = random.Random(seed)
    picked: set[int] = set()
    index = {id(it): i for i, it in enumerate(items)}
    quotas = stratum_quotas({key: len(group) for key, group in strata.items()}, n)
    for key, group in sorted(strata.items()):
        order = list(group)
        rng.shuffle(order)
        for it in order[: quotas[key]]:
            picked.add(index[id(it)])
    return [items[i] for i in sorted(picked)]


def draw_chain_sample(chains: Sequence[ChainItem], n: int, *, seed: int = 13) -> list[ChainItem]:
    """Draw exactly `min(n, population)` chain items spread across chain strata."""
    total = len(chains)
    if n >= total:
        return list(chains)
    strata: dict[str, list[ChainItem]] = {}
    for chain in chains:
        strata.setdefault(chain_stratum_key(chain), []).append(chain)
    rng = random.Random(seed)
    picked: set[int] = set()
    index = {id(chain): i for i, chain in enumerate(chains)}
    quotas = stratum_quotas({key: len(group) for key, group in strata.items()}, n)
    for key, group in sorted(strata.items()):
        order = list(group)
        rng.shuffle(order)
        for chain in order[: quotas[key]]:
            picked.add(index[id(chain)])
    return [chains[i] for i in sorted(picked)]


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


# --- retrieval-rank + page-citation context (read-only reviewer signals) --------------------


def load_retrieval_ranks(bundle: Path) -> dict[str, int]:
    """Index per-item `retrieval_rank` from any bundle sidecar that records it (empty if none).

    Reads `RETRIEVAL_RANK_SOURCES` (rows keyed by `id`); a null / missing rank means the item's
    gold span was NOT retrieved within top-k and is simply absent from the result.
    """
    ranks: dict[str, int] = {}
    for name in RETRIEVAL_RANK_SOURCES:
        path = Path(bundle) / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_id = row.get("id")
            rank = row.get("retrieval_rank")
            if item_id and isinstance(rank, int) and rank > 0:
                ranks[str(item_id)] = rank
    return ranks


def page_citation_for_span(
    corpus_root: Path,
    doc_id: str,
    char_start: int,
    char_end: int,
    cache: dict[str, tuple[str | None, list[dict[str, object]]] | None],
) -> str:
    """Render a `<source.pdf> p.N[-M]` citation for a span, "" when no PDF sidecar covers it.

    Reuses the PDF lane's `*.citations.json` sidecar (the same page join `build-index` uses), so
    the reviewer can check the original PDF page without hunting for offsets.
    """
    if doc_id not in cache:
        cache[doc_id] = load_page_citations(Path(corpus_root), doc_id)
    cite = cache[doc_id]
    if cite is None:
        return ""
    source, spans = cite
    pages = intersect_pages(char_start, char_end, spans)
    if not pages:
        return ""
    label = f"p.{pages[0]}" if pages[0] == pages[-1] else f"p.{pages[0]}-{pages[-1]}"
    name = Path(source).name if source else ""
    return f"{name} {label}".strip()


# --- confidence ordering (review queue) -----------------------------------------------------


def row_confidence(row: dict[str, str]) -> float:
    """A draft item's prior plausibility from the read-only signals on its worksheet row.

    Each cross-check verdict flag contributes +1 (true) / -1 (false); a top-k retrieval rank
    contributes 1/rank. Higher = more likely fine. Purely heuristic -- it ORDERS the review
    queue and never decides anything.
    """
    score = 0.0
    for col in ("cc_grounded", "cc_non_circular", "cc_supported", "cc_answerable"):
        value = (row.get(col) or "").strip().lower()
        if value == "true":
            score += 1.0
        elif value == "false":
            score -= 1.0
    rank = (row.get("retrieval_rank") or "").strip()
    if rank.isdigit() and int(rank) > 0:
        score += 1.0 / int(rank)
    return score


def confidence_order(rows: Sequence[dict[str, str]]) -> list[int]:
    """Row indices ordered LEAST-confident first (ties keep worksheet order).

    Suspicious items meet the reviewer's fresh attention first; the tail becomes quick
    confirmations -- that is the throughput win. The worksheet itself is never reordered.
    """
    return sorted(range(len(rows)), key=lambda i: (row_confidence(rows[i]), i))


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


def _worksheet_sample_kind(path: Path) -> str:
    manifest = path.with_name(SAMPLE_MANIFEST)
    if not manifest.is_file():
        return KIND_GOLDSET
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return KIND_GOLDSET
    kind = payload.get("kind")
    return str(kind) if kind in (KIND_GOLDSET, KIND_CHAINS) else KIND_GOLDSET


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
    chain_ledger = path / CHAINS_FILENAME if path.is_dir() else path
    if not ledger.is_file() and chain_ledger.name == CHAINS_FILENAME and chain_ledger.is_file():
        try:
            chains = load_chains(chain_ledger)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return VerificationRefStatus(
                False,
                chain_ledger,
                "accepted_chain_ledger",
                f"unreadable chain ledger: {exc}",
                instruction=instruction,
            )
        unverified_ids = [chain.chain_id for chain in chains if not chain.verified]
        chain_stats: dict[str, object] = {
            "chains": len(chains),
            "verified": len(chains) - len(unverified_ids),
            "unverified": len(unverified_ids),
            "unverified_item_ids": unverified_ids[:10],
        }
        if not chains:
            return VerificationRefStatus(
                False,
                chain_ledger,
                "accepted_chain_ledger",
                "ledger has no chains",
                stats=chain_stats,
                instruction=instruction,
            )
        if unverified_ids:
            return VerificationRefStatus(
                False,
                chain_ledger,
                "accepted_chain_ledger",
                "ledger contains unverified chain(s)",
                stats=chain_stats,
                instruction=instruction,
            )
        return VerificationRefStatus(True, chain_ledger, "accepted_chain_ledger", stats=chain_stats)
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
    retrieval_rank: int | None = None,
    page_cache: dict[str, tuple[str | None, list[dict[str, object]]] | None] | None = None,
) -> dict[str, str]:
    span = item.source_spans[0]
    text = _corpus_text(corpus_root, span.doc_id, cache)
    context = (
        corpus_window(text, span.char_start, span.char_end) if text is not None else "(missing doc)"
    )
    page = (
        page_citation_for_span(corpus_root, span.doc_id, span.char_start, span.char_end, page_cache)
        if page_cache is not None
        else ""
    )

    def _flag(key: str) -> str:
        value = verdict.get(key)
        return "" if value is None else ("true" if value else "false")

    return {
        "item_kind": KIND_GOLDSET,
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
        "retrieval_rank": "" if retrieval_rank is None else str(retrieval_rank),
        "page_citation": page,
        "chain_steps": "",
        "cc_grounded": _flag("grounded"),
        "cc_non_circular": _flag("non_circular"),
        "cc_supported": _flag("supported"),
        "cc_answerable": _flag("answerable"),
        "cc_note": str(verdict.get("note", "")),
    }


def _chain_step_contexts(
    chain: ChainItem,
    corpus_root: Path,
    cache: dict[str, str | None],
    page_cache: dict[str, tuple[str | None, list[dict[str, object]]] | None],
) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []
    for step in chain.steps:
        span = step.source_spans[0]
        text = _corpus_text(corpus_root, span.doc_id, cache)
        context = (
            corpus_window(text, span.char_start, span.char_end)
            if text is not None
            else "(missing doc)"
        )
        page = page_citation_for_span(
            corpus_root, span.doc_id, span.char_start, span.char_end, page_cache
        )
        steps.append(
            {
                "order": str(step.order),
                "question": step.question,
                "reference_answer": step.reference_answer,
                "dependency_note": step.dependency_note,
                "span_doc_id": span.doc_id,
                "span_text": span.text,
                "context": context,
                "page_citation": page,
            }
        )
    return steps


def _row_for_chain(
    chain: ChainItem,
    corpus_root: Path,
    cache: dict[str, str | None],
    page_cache: dict[str, tuple[str | None, list[dict[str, object]]] | None],
) -> dict[str, str]:
    steps = _chain_step_contexts(chain, corpus_root, cache, page_cache)
    final = steps[-1] if steps else {}
    first_doc = chain.steps[0].source_doc_id if chain.steps else ""
    return {
        "item_kind": KIND_CHAINS,
        "item_id": chain.chain_id,
        "provenance": chain.provenance,
        "split": chain.split,
        "source_doc_id": first_doc,
        "synthetic": "false",
        "stratum": chain_stratum_key(chain),
        "question": " -> ".join(step.question for step in chain.steps),
        "reference_answer": chain.steps[-1].reference_answer if chain.steps else "",
        "span_doc_id": final.get("span_doc_id", ""),
        "span_text": final.get("span_text", ""),
        "context": final.get("context", ""),
        "retrieval_rank": "",
        "page_citation": final.get("page_citation", ""),
        "chain_steps": json.dumps(steps, ensure_ascii=False),
    }


def _sample_rows(
    bundle: Path, sample: Sequence[GoldItem], *, synthetic: bool
) -> list[dict[str, str]]:
    """Build worksheet rows for `sample`, joining cross-check, retrieval rank, and page cites."""
    verdicts = load_cross_check(bundle)
    ranks = load_retrieval_ranks(bundle)
    corpus_root = bundle / CORPUS_DIRNAME
    cache: dict[str, str | None] = {}
    page_cache: dict[str, tuple[str | None, list[dict[str, object]]] | None] = {}
    return [
        _row_for(
            it,
            corpus_root,
            cache,
            verdicts.get(it.id, {}),
            synthetic=synthetic,
            retrieval_rank=ranks.get(it.id),
            page_cache=page_cache,
        )
        for it in sample
    ]


def _sample_chain_rows(bundle: Path, sample: Sequence[ChainItem]) -> list[dict[str, str]]:
    """Build worksheet rows for chain samples, with every step rendered into chain_steps JSON."""
    corpus_root = bundle / CORPUS_DIRNAME
    cache: dict[str, str | None] = {}
    page_cache: dict[str, tuple[str | None, list[dict[str, object]]] | None] = {}
    return [_row_for_chain(chain, corpus_root, cache, page_cache) for chain in sample]


def _write_sample_manifest(
    out_path: Path,
    bundle: Path,
    manifest_update: dict[str, object],
    *,
    merge_existing: bool = False,
) -> None:
    """Write the sibling `sample_manifest.json` (merge into the existing one on enlargement)."""
    manifest_path = Path(out_path).with_name(SAMPLE_MANIFEST)
    manifest: dict[str, object] = {}
    if merge_existing and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    manifest.update({"bundle": str(bundle), "worksheet": str(out_path)})
    manifest.update(manifest_update)
    atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))


def build_sample_worksheet(
    bundle: Path, out_path: Path, *, n: int, seed: int = 13, kind: str = KIND_AUTO
) -> tuple[int, dict[str, int]]:
    """Draw a stratified sample from a draft bundle and write the verification worksheet.

    Returns `(sample_size, strata_sizes)` and writes a `sample_manifest.json` beside the
    worksheet documenting the bundle, seed, requested/actual size, and per-stratum counts (the
    "document the size + strata" half of the procedure). Human columns are left blank.
    """
    bundle = Path(bundle)
    resolved_kind = resolve_sample_kind(bundle, kind)
    synthetic = bundle_is_synthetic(bundle) if resolved_kind == KIND_GOLDSET else False
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
    write_worksheet_rows(out_path, rows)

    _write_sample_manifest(
        out_path,
        bundle,
        {
            "kind": resolved_kind,
            "synthetic": synthetic,
            "seed": seed,
            "requested": n,
            "sample_size": sample_size,
            "population": population,
            "strata": strata_sizes,
        },
    )
    return sample_size, strata_sizes


def merge_sample_worksheet(
    bundle: Path, out_path: Path, *, n: int, seed: int = 13, kind: str = KIND_AUTO
) -> tuple[int, int]:
    """Enlarge an existing worksheet ADDITIVELY to ~`n` rows; returns `(added, total)`.

    Draws a fresh stratified sample of size `n` over the whole bundle and appends rows ONLY for
    item ids the worksheet does not already hold: existing rows -- including every human decision
    -- are rewritten byte-for-byte untouched, and a decided row is never re-shown or re-drawn.
    Falls back to a fresh `build_sample_worksheet` when the worksheet does not exist yet.
    """
    out_path = Path(out_path)
    if not out_path.is_file():
        size, _ = build_sample_worksheet(bundle, out_path, n=n, seed=seed, kind=kind)
        return size, size
    bundle = Path(bundle)
    manifest_kind = _worksheet_sample_kind(out_path)
    resolved_kind = resolve_sample_kind(bundle, manifest_kind if kind == KIND_AUTO else kind)
    existing_rows, fieldnames = load_worksheet(out_path)
    existing_ids = {(row.get("item_id") or "").strip() for row in existing_rows}
    synthetic = bundle_is_synthetic(bundle) if resolved_kind == KIND_GOLDSET else False
    if resolved_kind == KIND_CHAINS:
        chains = load_chains(find_chains(bundle))
        chain_sample = draw_chain_sample(chains, n, seed=seed)
        new_chains = [chain for chain in chain_sample if chain.chain_id not in existing_ids]
        if not new_chains:
            return 0, len(existing_rows)
        new_rows = _sample_chain_rows(bundle, new_chains)
        population = len(chains)
    else:
        items = load_goldset(find_goldset(bundle))
        gold_sample = draw_stratified_sample(items, n, seed=seed)
        new_items = [it for it in gold_sample if it.id not in existing_ids]
        if not new_items:
            return 0, len(existing_rows)
        new_rows = _sample_rows(bundle, new_items, synthetic=synthetic)
        population = len(items)
    if not new_rows:
        return 0, len(existing_rows)
    all_rows = [*existing_rows, *new_rows]
    write_worksheet_rows(out_path, all_rows, fieldnames)
    _write_sample_manifest(
        out_path,
        bundle,
        {
            "kind": resolved_kind,
            "synthetic": synthetic,
            "seed": seed,
            "requested": n,
            "sample_size": len(all_rows),
            "population": population,
            "merged_added": len(new_rows),
        },
        merge_existing=True,
    )
    return len(new_rows), len(all_rows)


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
    for key, cell in per_stratum.items():
        d = cell["decided"]
        cell["reject_rate"] = (cell["rejected"] / d) if d else 0.0
        cell["tolerance"] = _stratum_tolerance(key, tolerance, stratum_tolerances)
        cell["passed"] = float(cell["reject_rate"] <= cell["tolerance"])
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
    from llb.goldset.verify_multi import resolve_multi_reviewer_rows

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="human verification gate human sample-verification of draft data."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sa = sub.add_parser("sample", help="draw a stratified sample from a draft bundle -> worksheet")
    sa.add_argument(
        "--bundle", required=True, type=Path, help="draft dir (goldset.jsonl + corpus/)"
    )
    sa.add_argument("--out", required=True, type=Path, help="verification worksheet CSV to write")
    sa.add_argument("-n", "--size", type=int, default=30, help="target sample size")
    sa.add_argument("--seed", type=int, default=13)
    sa.add_argument(
        "--kind",
        choices=SAMPLE_KINDS,
        default=KIND_AUTO,
        help="sample flat goldset rows, chain rows, or auto-select chains when present",
    )
    sa.add_argument(
        "--merge",
        action="store_true",
        help="enlarge an existing worksheet additively (decided rows preserved byte-for-byte)",
    )
    sa.add_argument(
        "--annotators",
        type=int,
        default=1,
        help="write the SAME sample as k per-reviewer worksheets (multi-annotator gate)",
    )

    rv = sub.add_parser("review", help="interactively verify the sampled items")
    rv.add_argument("--worksheet", required=True, type=Path)
    rv.add_argument("--start", type=int, default=None, help="begin at this 1-based item")
    rv.add_argument(
        "--show-crosscheck",
        action="store_true",
        help="reveal the second-frontier verdict (post-hoc only; anchors the human -- off by default)",
    )
    rv.add_argument("--clear", action="store_true", help="wipe ALL human columns first (gated)")
    rv.add_argument(
        "--order",
        choices=("worksheet", "confidence"),
        default="worksheet",
        help="review queue order: worksheet row order, or least-confident first",
    )

    aj = sub.add_parser(
        "adjudicate",
        help="agreement report (Cohen/Fleiss kappa) + adjudication worksheet from disagreements",
    )
    aj.add_argument(
        "--bundle", required=True, type=Path, help="the draft bundle the samples came from"
    )
    aj.add_argument(
        "--worksheet",
        type=Path,
        default=None,
        help="base worksheet path (default: <bundle>/verify_sample.csv)",
    )

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
    ac.add_argument(
        "--policy",
        choices=ACCEPT_POLICIES,
        default=POLICY_GLOBAL,
        help="acceptance arithmetic: global rate, per-stratum thresholds, or confidence-weighted",
    )
    ac.add_argument(
        "--stratum-tolerance",
        action="append",
        default=[],
        metavar="STRATUM=TOL",
        help="per-stratum tolerance override (repeatable; used by --policy per-stratum)",
    )

    args = parser.parse_args(argv)

    if args.cmd == "sample":
        if args.annotators > 1:
            from llb.goldset.verify_multi import build_multi_reviewer_worksheets

            if args.merge:
                parser.error("--merge is not supported together with --annotators")
            paths = build_multi_reviewer_worksheets(
                args.bundle,
                args.out,
                n=args.size,
                annotators=args.annotators,
                seed=args.seed,
                kind=args.kind,
            )
            for path in paths:
                _LOG.info("[verify] reviewer worksheet: make verify-review VERIFY_WS=%s", path)
            _LOG.info("[verify] after all reviews: make verify-adjudicate BUNDLE=%s", args.bundle)
            return 0
        if args.merge:
            added, total = merge_sample_worksheet(
                args.bundle, args.out, n=args.size, seed=args.seed, kind=args.kind
            )
            _LOG.info("[verify] merged %d new item(s) -> %d total in %s", added, total, args.out)
        else:
            size, strata = build_sample_worksheet(
                args.bundle, args.out, n=args.size, seed=args.seed, kind=args.kind
            )
            _LOG.info(
                "[verify] sampled %d item(s) across %d strata -> %s", size, len(strata), args.out
            )
        _LOG.info("[verify] review: make verify-review VERIFY_WS=%s", args.out)
        return 0

    if args.cmd == "review":
        from llb.goldset.verify_session import run_session

        run_session(
            args.worksheet,
            start=args.start,
            show_crosscheck=args.show_crosscheck,
            clear=args.clear,
            order=args.order,
        )
        return 0

    if args.cmd == "adjudicate":
        from llb.goldset.verify_multi import run_adjudicate

        return run_adjudicate(args.bundle, args.worksheet)

    overrides: dict[str, float] = {}
    for spec in args.stratum_tolerance:
        key, sep, value = spec.rpartition("=")
        if not sep or not key:
            parser.error(f"--stratum-tolerance expects STRATUM=TOL, got {spec!r}")
        try:
            overrides[key] = float(value)
        except ValueError:
            parser.error(f"--stratum-tolerance value is not a number: {spec!r}")
    return run_accept(
        args.worksheet,
        args.bundle,
        args.out_dir,
        args.tolerance,
        policy=args.policy,
        stratum_tolerances=overrides or None,
    )


if __name__ == "__main__":
    from llb.core.runtime import run

    sys.exit(run(main))
