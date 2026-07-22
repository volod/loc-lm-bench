"""Production wiring: score one item set end to end under every context lane, then compare.

Each lane is an ORDINARY `run-eval` bundle under `$DATA_DIR/run-eval/` -- nothing about a lane's
scoring is special-cased here, so its numbers are reproducible by re-running `run-eval
--context-strategy <lane>` with the same config. The item set is selected ONCE and handed to every
lane, which is what makes the per-item pairing in the comparison legitimate.

`run_lane` is injectable, so the whole orchestration runs in CI with fake bundles -- no backend, no
store, no GPU.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from llb.board.io import read_case_rows
from llb.core.config import RunConfig
from llb.eval.answer_quality.run import (
    GROUNDING_DRAFTED,
    GROUNDING_VERIFIED,
    select_items,
)
from llb.eval.context_ablation.compare import compare_context_strategies
from llb.eval.context_ablation.lanes import default_lanes, lane_config
from llb.eval.context_ablation.models import (
    LANE_CLOSED_BOOK,
    ContextAblationReport,
)
from llb.eval.context_ablation.report import format_report
from llb.eval.paired_cases import CaseRows
from llb.goldset.schema import GoldItem
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED
from llb.rag.question_types import load_question_types

METHOD = "context-ablation"
RUN_NAME_PREFIX = "context-ablation"

# One lane config + one split's items -> that (lane, split) bundle's persisted `scores.jsonl`.
LaneRunner = Callable[[RunConfig, list[GoldItem], str], Path]


@dataclass(frozen=True)
class ContextAblationRun:
    report: ContextAblationReport
    out_dir: Path
    paths: Mapping[str, str]


def eval_lane_runner(*, verified_only: bool = True) -> LaneRunner:
    """The default lane runner: one ordinary `run-eval` bundle per (lane, split)."""

    def run_lane(config: RunConfig, items: list[GoldItem], split: str) -> Path:
        from llb.executor.runner import run_eval

        result = run_eval(config, items=items, split=split, verified_only=verified_only)
        return Path(str(result["paths"]["scores"]))

    return run_lane


def score_lanes(
    config: RunConfig,
    lanes: Sequence[str],
    items_by_split: Mapping[str, list[GoldItem]],
    *,
    run_lane: LaneRunner,
) -> tuple[dict[str, CaseRows], dict[str, list[str]]]:
    """Run every lane over the SAME items, then read back its per-case rows.

    Several splits pool into ONE compared item set (one run bundle each, so every bundle stays an
    ordinary per-split run).
    """
    rows: dict[str, CaseRows] = {}
    run_dirs: dict[str, list[str]] = {}
    for lane in lanes:
        config_for_lane = lane_config(config, lane, run_name_prefix=RUN_NAME_PREFIX)
        lane_rows: CaseRows = []
        lane_dirs: list[str] = []
        for split, items in items_by_split.items():
            scores = run_lane(config_for_lane, items, split)
            lane_rows.extend(read_case_rows(scores))
            lane_dirs.append(str(scores.parent))
        rows[lane] = lane_rows
        run_dirs[lane] = lane_dirs
    return rows, run_dirs


def run_context_ablation(
    config: RunConfig,
    lanes: Sequence[str] | None = None,
    *,
    splits: Sequence[str] = ("final",),
    limit: int | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    out_dir: Path | None = None,
    verified_only: bool = True,
    run_lane: LaneRunner | None = None,
) -> ContextAblationRun:
    """Score the selected items under every context lane and persist the comparison."""
    selection = list(lanes) if lanes else default_lanes()
    if LANE_CLOSED_BOOK not in selection:
        raise ValueError(
            f"the ablation needs the {LANE_CLOSED_BOOK!r} lane: every derived number is stated "
            "against it"
        )
    if len(selection) < 2:
        raise ValueError("the comparison needs the baseline lane and at least one other lane")
    if not splits:
        raise ValueError("name at least one gold split to score")
    items_by_split = select_items(config, splits, limit, verified_only)
    rows, run_dirs = score_lanes(
        config,
        selection,
        items_by_split,
        run_lane=run_lane or eval_lane_runner(verified_only=verified_only),
    )
    report = compare_context_strategies(
        rows,
        load_question_types(config.goldset_path),
        baseline=LANE_CLOSED_BOOK,
        run_dirs=run_dirs,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    target = Path(out_dir) if out_dir is not None else default_out_dir(config)
    paths = write_artifacts(report, target, metadata=_metadata(config, splits, verified_only))
    return ContextAblationRun(report, target, paths)


def default_out_dir(config: RunConfig) -> Path:
    """`$DATA_DIR/context-ablation/<timestamp>/`."""
    from llb.core.store_generations import generation_timestamp

    return config.data_dir / METHOD / generation_timestamp()


def _metadata(config: RunConfig, splits: Sequence[str], verified_only: bool) -> dict[str, object]:
    return {
        "model": config.model,
        "backend": config.backend,
        "split": ",".join(splits),
        "goldset": str(config.goldset_path),
        "corpus": str(config.corpus_root),
        "grounding": GROUNDING_VERIFIED if verified_only else GROUNDING_DRAFTED,
    }


def write_artifacts(
    report: ContextAblationReport, out_dir: Path, *, metadata: Mapping[str, object]
) -> dict[str, str]:
    """Persist `report.md` + `comparison.json` under the comparison directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {**report, "metadata": dict(metadata)}
    (out_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(format_report(report, metadata=metadata), encoding="utf-8")
    return {
        "report": str(out_dir / "report.md"),
        "comparison": str(out_dir / "comparison.json"),
    }
