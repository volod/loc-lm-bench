"""Adapter garbage collection guarded by durable-artifact citations.

An adapter is `superseded` once a newer adapter exists for the same base model. Superseded
adapters are the only GC candidates, and even those are refused while any durable artifact
still cites them -- deleting one would strand evidence that can no longer be reproduced. The
citation scan covers published run bundles (`$DATA_DIR/run-eval/*/manifest.json`) AND the
orchestrator journals that also link adapter directories: self-improvement `state.json`
(`rounds[].adapter_dir`) and campaign `campaign.progress.jsonl` (`entry.adapter_dir`). Each
citation carries its artifact kind so the refusal names what still points at the adapter.
`--force` overrides the citation refusal; it never overrides the safety rule that GC only ever
deletes directories inside `$DATA_DIR`, which keeps committed fixtures out of reach.
"""

import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from llb.core.config import RUN_EVAL_METHOD
from llb.core.contracts import JsonObject
from llb.core.paths import resolve_project_path
from llb.finetune.registry import (
    AdapterEntry,
    load_registry,
    merged_root,
    record_delete,
    registry_path,
)

_LOG = logging.getLogger(__name__)

RUN_MANIFEST = "manifest.json"
SELF_IMPROVE_METHOD = "self-improve"
SELF_IMPROVE_STATE = "state.json"
CAMPAIGN_METHOD = "finetune-campaign"
CAMPAIGN_PROGRESS = "campaign.progress.jsonl"

# Citation kinds, prefixed onto every `cited_by` entry as `<kind>:<artifact-path>` so a refusal
# can name exactly which durable artifact still links the adapter.
CITED_BY_RUN_BUNDLE = "run-bundle"
CITED_BY_SELF_IMPROVE = "self-improve-state"
CITED_BY_CAMPAIGN = "campaign-journal"

GC_DELETE = "delete"
GC_REFUSE = "refuse"
GC_KEEP = "keep"

REASON_NEWEST = "newest adapter for its base model"
REASON_OUTSIDE_DATA_DIR = "adapter directory lives outside $DATA_DIR"

_MAX_NAMED_CITATIONS = 2  # citations spelled out in a refusal reason before "+N more"


@dataclass(frozen=True)
class GcDecision:
    """What GC decided for one registered adapter, and why.

    `cited_by` holds every durable artifact still linking the adapter as
    `<kind>:<artifact-path>` (kinds: run-bundle, self-improve-state, campaign-journal).
    """

    entry: AdapterEntry
    action: str
    reason: str
    cited_by: tuple[str, ...] = ()
    superseded_by: str | None = None


@dataclass
class GcPlan:
    decisions: list[GcDecision] = field(default_factory=list)

    def _of(self, action: str) -> list[GcDecision]:
        return [decision for decision in self.decisions if decision.action == action]

    @property
    def deleted(self) -> list[GcDecision]:
        return self._of(GC_DELETE)

    @property
    def refused(self) -> list[GcDecision]:
        return self._of(GC_REFUSE)

    @property
    def kept(self) -> list[GcDecision]:
        return self._of(GC_KEEP)


def cited_adapters(
    run_root: Path | str,
    entries: dict[str, AdapterEntry],
    *,
    data_dir: Path | str | None = None,
) -> dict[str, tuple[str, ...]]:
    """Map adapter id -> durable artifacts citing it, each as `<kind>:<artifact-path>`.

    Scans published run bundles under `run_root` (by recorded digest or served adapter path)
    and, when `data_dir` is given, the orchestrator journals that also link `adapter_dir`
    paths: `<data_dir>/self-improve/*/state.json` and
    `<data_dir>/finetune-campaign/*/campaign.progress.jsonl` (resolved through the registry's
    adapter-dir index the way the `adapter_path` match already is).
    """
    root = Path(run_root)
    dir_to_id = {entry.resolved_dir: entry.adapter_id for entry in entries.values()}
    citations: dict[str, list[str]] = defaultdict(list)
    if root.is_dir():
        for manifest_path in sorted(root.glob(f"*/{RUN_MANIFEST}")):
            if manifest_path.parent.name.startswith("."):
                continue
            for adapter_id in _cited_ids(manifest_path, dir_to_id):
                citations[adapter_id].append(f"{CITED_BY_RUN_BUNDLE}:{manifest_path.parent}")
    if data_dir is not None:
        for adapter_id, citation in _journal_citations(Path(data_dir), dir_to_id):
            citations[adapter_id].append(citation)
    return {adapter_id: tuple(runs) for adapter_id, runs in citations.items()}


def _journal_citations(data_dir: Path, dir_to_id: dict[Path, str]) -> list[tuple[str, str]]:
    """`(adapter_id, "<kind>:<journal-path>")` pairs from the orchestrator journals."""
    found: list[tuple[str, str]] = []
    for state_path in sorted((data_dir / SELF_IMPROVE_METHOD).glob(f"*/{SELF_IMPROVE_STATE}")):
        rows = _state_rounds(state_path)
        for adapter_id in _dir_cited_ids(rows, dir_to_id):
            found.append((adapter_id, f"{CITED_BY_SELF_IMPROVE}:{state_path}"))
    for progress_path in sorted((data_dir / CAMPAIGN_METHOD).glob(f"*/{CAMPAIGN_PROGRESS}")):
        rows = _progress_entries(progress_path)
        for adapter_id in _dir_cited_ids(rows, dir_to_id):
            found.append((adapter_id, f"{CITED_BY_CAMPAIGN}:{progress_path}"))
    return found


def _state_rounds(state_path: Path) -> list[JsonObject]:
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("[gc-adapters] unreadable self-improve state %s: %s", state_path, exc)
        return []
    rounds = payload.get("rounds") if isinstance(payload, dict) else None
    return [row for row in rounds if isinstance(row, dict)] if isinstance(rounds, list) else []


def _progress_entries(progress_path: Path) -> list[JsonObject]:
    try:
        lines = progress_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _LOG.warning("[gc-adapters] unreadable campaign journal %s: %s", progress_path, exc)
        return []
    rows: list[JsonObject] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _dir_cited_ids(rows: list[JsonObject], dir_to_id: dict[Path, str]) -> set[str]:
    """Adapter ids whose registered directory a journal row's `adapter_dir` resolves to."""
    found: set[str] = set()
    for row in rows:
        adapter_dir = row.get("adapter_dir")
        if not adapter_dir:
            continue
        adapter_id = dir_to_id.get(resolve_project_path(str(adapter_dir)))
        if adapter_id is not None:
            found.add(adapter_id)
    return found


def plan_gc(
    entries: dict[str, AdapterEntry],
    *,
    cited: dict[str, tuple[str, ...]],
    data_dir: Path | str,
    force: bool = False,
) -> GcPlan:
    """Decide delete/refuse/keep per adapter without touching the filesystem."""
    superseded = supersession(entries)
    root = Path(data_dir).resolve()
    plan = GcPlan()
    for entry in sorted(entries.values(), key=lambda item: item.adapter_id):
        newer = superseded.get(entry.adapter_id)
        runs = cited.get(entry.adapter_id, ())
        if newer is None:
            plan.decisions.append(GcDecision(entry, GC_KEEP, REASON_NEWEST, runs))
            continue
        if not _within(entry.resolved_dir, root):
            plan.decisions.append(
                GcDecision(entry, GC_REFUSE, REASON_OUTSIDE_DATA_DIR, runs, newer)
            )
            continue
        if runs and not force:
            reason = (
                f"cited by {len(runs)} durable artifact(s): {_citation_note(runs)}; "
                "pass --force to delete anyway"
            )
            plan.decisions.append(GcDecision(entry, GC_REFUSE, reason, runs, newer))
            continue
        reason = f"superseded by {newer[:12]}"
        if runs:
            reason += f"; forced past {len(runs)} citation(s)"
        plan.decisions.append(GcDecision(entry, GC_DELETE, reason, runs, newer))
    return plan


def gc_adapters(
    *,
    data_dir: Path | str,
    run_root: Path | str | None = None,
    registry: Path | str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> GcPlan:
    """Delete superseded, uncited adapter directories and tombstone them in the registry."""
    registry_file = Path(registry) if registry is not None else registry_path(data_dir)
    entries = load_registry(registry_file)
    runs = Path(run_root) if run_root is not None else Path(data_dir) / RUN_EVAL_METHOD
    cited = cited_adapters(runs, entries, data_dir=data_dir)
    plan = plan_gc(entries, cited=cited, data_dir=data_dir, force=force)
    if dry_run:
        return plan
    merged = merged_root(data_dir)
    for decision in plan.deleted:
        shutil.rmtree(decision.entry.resolved_dir, ignore_errors=True)
        shutil.rmtree(merged / decision.entry.short_id, ignore_errors=True)
        record_delete(
            registry=registry_file, adapter_id=decision.entry.adapter_id, reason=decision.reason
        )
    return plan


def supersession(entries: dict[str, AdapterEntry]) -> dict[str, str]:
    """Map each superseded adapter id to the newest adapter id for the same base model."""
    by_model: dict[str, list[AdapterEntry]] = defaultdict(list)
    for entry in entries.values():
        by_model[entry.base_model].append(entry)
    superseded: dict[str, str] = {}
    for group in by_model.values():
        ordered = sorted(group, key=lambda entry: entry.recency, reverse=True)
        for entry in ordered[1:]:
            superseded[entry.adapter_id] = ordered[0].adapter_id
    return superseded


def _cited_ids(manifest_path: Path, dir_to_id: dict[Path, str]) -> set[str]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("[gc-adapters] unreadable run bundle %s: %s", manifest_path.parent, exc)
        return set()
    config = manifest.get("config") if isinstance(manifest, dict) else None
    if not isinstance(config, dict):
        return set()
    found: set[str] = set()
    adapter = config.get("adapter")
    if isinstance(adapter, dict) and adapter.get("adapter_digest"):
        found.add(str(adapter["adapter_digest"]))
    served = config.get("adapter_path")
    if served:
        adapter_id = dir_to_id.get(resolve_project_path(str(served)))
        if adapter_id is not None:
            found.add(adapter_id)
    return found


def _within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _citation_note(citations: tuple[str, ...]) -> str:
    """Name the first few citing artifacts (`kind:path`), truncating the rest to a count."""
    head = "; ".join(citations[:_MAX_NAMED_CITATIONS])
    extra = len(citations) - _MAX_NAMED_CITATIONS
    return head + (f"; +{extra} more" if extra > 0 else "")


def _citation_kinds(citations: tuple[str, ...]) -> list[str]:
    """The distinct artifact kinds among `kind:path` citations, sorted."""
    return sorted({c.split(":", 1)[0] for c in citations if c})


def gc_rows(plan: GcPlan) -> list[JsonObject]:
    """`llb gc-adapters` rows: one per registered adapter, delete decisions first."""
    order = {GC_DELETE: 0, GC_REFUSE: 1, GC_KEEP: 2}
    ordered = sorted(plan.decisions, key=lambda d: (order.get(d.action, 9), d.entry.adapter_id))
    return [
        {
            "adapter_id": decision.entry.short_id,
            "base_model": decision.entry.base_model,
            "action": decision.action,
            "reason": decision.reason,
            "cited_by": len(decision.cited_by),
            "cited_kinds": ",".join(_citation_kinds(decision.cited_by)),
            "adapter_dir": str(decision.entry.adapter_dir),
        }
        for decision in ordered
    ]
