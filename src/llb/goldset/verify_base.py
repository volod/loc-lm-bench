"""Shared schema + bundle layout + worksheet I/O for the human verification gate.

The leaf module of the verification family: the worksheet schema (constants +
`VerificationRefStatus`), the draft-bundle layout helpers, and the atomic CSV-as-state worksheet
I/O. It depends on nothing else in the family, so sampling, acceptance, reference checking, and
the CLI can all build on it without an import cycle.
"""

import csv
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from llb.core.fsutil import atomic_write_text
from llb.goldset.chains import CHAINS_FILENAME

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

# Sampler-owned reviewer id for multi-annotator worksheets. It is blank on single-reviewer rows.
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


# --- worksheet I/O (atomic, CSV-as-state -- mirrors judge/calibration.py) ------------------


def worksheet_fieldnames(existing: Sequence[str] | None = None) -> list[str]:
    """Complete a profile header with the shared verification columns."""
    names = list(existing) if existing else []
    for col in WORKSHEET_COLS:
        if col not in names:
            names.append(col)
    return names


def load_worksheet(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Load a worksheet with shared and profile-specific verification columns."""
    text = Path(path).read_text(encoding="utf-8")
    reader = csv.DictReader(text.splitlines())
    fieldnames = worksheet_fieldnames(reader.fieldnames)
    rows = [{name: (raw.get(name) or "") for name in fieldnames} for raw in reader]
    return rows, fieldnames


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


# --- verification-reference helpers (shared by the refcheck submodule) ----------------------


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
