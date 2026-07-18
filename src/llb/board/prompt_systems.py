"""Prompt-system comparison loading for the board."""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from llb.core.contracts.results import BoardRow
from llb.scoring.aggregate import rank_board
from llb.scoring.board_format import format_board, ranking_policy_note
from llb.scoring.leaderboard import ModelResult, bootstrap_mean_ci

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
    knowledge_tree: dict[str, object]


@dataclass
class KnowledgeTreeABResult:
    """Best tree candidate versus best no-tree control over aligned cases."""

    baseline_id: str
    tree_id: str
    delta: float
    ci: tuple[float, float] | None
    conclusion: str
    depth: int
    budget_tokens: int


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


def _add_best_rag_prompt_system_record(
    manifest: dict[str, Any],
    manifest_path: Path,
    best: dict[tuple[str, str], RagPromptSystemRunRecord],
    prompt_system: str,
    knowledge_tree: dict[str, object],
) -> None:
    record = record_from_manifest(manifest, manifest_path.parent)
    if record is None:
        return
    key = (record.result.model, str(prompt_system))
    current = best.get(key)
    if current is None or record.result.objective_score > current.result.objective_score:
        best[key] = RagPromptSystemRunRecord(
            model=record.result.model,
            prompt_system=str(prompt_system),
            result=record.result,
            run_dir=record.run_dir,
            knowledge_tree=knowledge_tree,
        )


def _add_best_rag_prompt_system(
    best: dict[tuple[str, str], RagPromptSystemRunRecord], root: Path
) -> None:
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

        tree = provenance.get("knowledge_tree") or {}
        _add_best_rag_prompt_system_record(
            manifest,
            manifest_path,
            best,
            prompt_system,
            tree if isinstance(tree, dict) else {},
        )


def load_rag_prompt_system_records(data_dir: Path | str) -> list[RagPromptSystemRunRecord]:
    """Load final-split `run-eval` bundles tagged with prompt-system provenance."""
    root = Path(data_dir) / "run-eval"
    best: dict[tuple[str, str], RagPromptSystemRunRecord] = {}
    if not root.exists():
        return []

    _add_best_rag_prompt_system(best, root)

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


def knowledge_tree_ab_comparison(data_dir: Path | str, model: str) -> KnowledgeTreeABResult | None:
    """Compare the best evaluated tree with the best evaluated no-tree control."""
    records = [
        record for record in load_rag_prompt_system_records(data_dir) if record.model == model
    ]
    baselines = [record for record in records if not record.knowledge_tree]
    trees = [record for record in records if record.knowledge_tree]
    if not baselines or not trees:
        return None
    tree = max(trees, key=lambda record: record.result.objective_score)
    baseline_id = str(tree.knowledge_tree.get("baseline_prompt_system_id", ""))
    baseline = next(
        (record for record in baselines if record.prompt_system == baseline_id),
        max(baselines, key=lambda record: record.result.objective_score),
    )
    base_cases = baseline.result.case_objectives
    tree_cases = tree.result.case_objectives
    if base_cases and len(base_cases) == len(tree_cases):
        differences = [candidate - control for candidate, control in zip(tree_cases, base_cases)]
        delta = sum(differences) / len(differences)
        ci = bootstrap_mean_ci(differences)
    else:
        delta = tree.result.objective_score - baseline.result.objective_score
        ci = None
    conclusion = "inconclusive"
    if ci is not None and ci[0] > 0:
        conclusion = "helps"
    elif ci is not None and ci[1] < 0:
        conclusion = "hurts"
    return KnowledgeTreeABResult(
        baseline_id=baseline.prompt_system,
        tree_id=tree.prompt_system,
        delta=round(delta, 4),
        ci=ci,
        conclusion=conclusion,
        depth=_tree_int(tree.knowledge_tree.get("depth")),
        budget_tokens=_tree_int(tree.knowledge_tree.get("budget_tokens")),
    )


def _tree_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    return 0
