"""Prompt-system comparison loading for the board."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

from llb.contracts import BoardRow
from llb.scoring.aggregate import ModelResult, format_board, rank_board, ranking_policy_note

from llb.board.categories import AGENTIC_METHOD
from llb.board.harnesses import harness_record_from_manifest
from llb.board.runs import record_from_manifest


@dataclass
class PromptSystemRunRecord:
    """One agentic run tagged by its prompt-system id."""

    model: str
    harness: str
    prompt_system: str
    result: ModelResult
    run_dir: str


@dataclass
class RagPromptSystemRunRecord:
    """One final-split RAG run tagged by prompt-system id."""

    model: str
    prompt_system: str
    result: ModelResult
    run_dir: str


def load_prompt_system_records(data_dir: Path | str) -> list[PromptSystemRunRecord]:
    """Load agentic run bundles that carry a prompt-system id."""
    root = Path(data_dir) / AGENTIC_METHOD
    best: dict[tuple[str, str, str], PromptSystemRunRecord] = {}
    if not root.exists():
        return []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        if manifest_path.parent.name.startswith("."):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        config = manifest.get("config") or {}
        prompt_system = config.get("prompt_system")
        if not prompt_system:
            continue
        record = harness_record_from_manifest(manifest, manifest_path.parent)
        if record is None:
            continue
        key = (record.model, record.harness, str(prompt_system))
        current = best.get(key)
        if current is None or record.result.objective_score > current.result.objective_score:
            best[key] = PromptSystemRunRecord(
                model=record.model,
                harness=record.harness,
                prompt_system=str(prompt_system),
                result=record.result,
                run_dir=record.run_dir,
            )
    return list(best.values())


def prompt_system_comparison(
    data_dir: Path | str, model: str, harness: str | None = None
) -> tuple[list[BoardRow], str, list[str]]:
    """Rank one model, optionally one harness, across prompt-system ids."""
    records = [
        r
        for r in load_prompt_system_records(data_dir)
        if r.model == model and (harness is None or r.harness == harness)
    ]
    if not records:
        return [], "", []
    ordered = sorted(records, key=lambda r: r.prompt_system)
    results = [replace(r.result, model=r.prompt_system) for r in ordered]
    rows = rank_board(results)
    table = format_board(rows, policy=ranking_policy_note(results, judge_trusted=False))
    return rows, table, [r.prompt_system for r in ordered]


def load_rag_prompt_system_records(data_dir: Path | str) -> list[RagPromptSystemRunRecord]:
    """Load final-split `run-eval` bundles tagged with prompt-system provenance."""
    root = Path(data_dir) / "run-eval"
    best: dict[tuple[str, str], RagPromptSystemRunRecord] = {}
    if not root.exists():
        return []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        if manifest_path.parent.name.startswith("."):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        provenance = manifest.get("prompt_system_provenance") or {}
        if not isinstance(provenance, dict):
            continue
        prompt_system = provenance.get("prompt_system_id")
        if not prompt_system:
            continue
        record = record_from_manifest(manifest, manifest_path.parent)
        if record is None:
            continue
        key = (record.result.model, str(prompt_system))
        current = best.get(key)
        if current is None or record.result.objective_score > current.result.objective_score:
            best[key] = RagPromptSystemRunRecord(
                model=record.result.model,
                prompt_system=str(prompt_system),
                result=record.result,
                run_dir=record.run_dir,
            )
    return list(best.values())


def rag_prompt_system_comparison(
    data_dir: Path | str, model: str
) -> tuple[list[BoardRow], str, list[str]]:
    """Rank one baseline RAG model across prompt-system ids."""
    records = [r for r in load_rag_prompt_system_records(data_dir) if r.model == model]
    if not records:
        return [], "", []
    ordered = sorted(records, key=lambda r: r.prompt_system)
    results = [replace(r.result, model=r.prompt_system) for r in ordered]
    rows = rank_board(results)
    table = format_board(rows, policy=ranking_policy_note(results, judge_trusted=False))
    return rows, table, [r.prompt_system for r in ordered]
