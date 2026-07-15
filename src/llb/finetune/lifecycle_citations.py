"""Focused lifecycle citations implementation."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from llb.core.contracts.common import JsonObject
from llb.core.paths import resolve_project_path
from llb.finetune.registry.model import AdapterEntry

_LOG = logging.getLogger(__name__)

RUN_MANIFEST = "manifest.json"

SELF_IMPROVE_METHOD = "self-improve"

SELF_IMPROVE_STATE = "state.json"

CAMPAIGN_METHOD = "finetune-campaign"

CAMPAIGN_PROGRESS = "campaign.progress.jsonl"

CITED_BY_RUN_BUNDLE = "run-bundle"

CITED_BY_SELF_IMPROVE = "self-improve-state"

CITED_BY_CAMPAIGN = "campaign-journal"


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
