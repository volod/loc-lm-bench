"""Production wiring: score one item set end to end under every context lane, then compare.

Each lane is an ORDINARY `run-eval` bundle under `$DATA_DIR/run-eval/` -- nothing about a lane's
scoring is special-cased here, so its numbers are reproducible by re-running `run-eval
--context-strategy <lane>` with the same config. The item set is selected ONCE and handed to every
lane, which is what makes the per-item pairing in the comparison legitimate.

`run_lane` is injectable, so the whole orchestration runs in CI with fake bundles -- no backend, no
store, no GPU.

`repeats` scores every lane more than once with the IDENTICAL config on the IDENTICAL items. That
is not more evidence -- the comparison is still taken over the first repeat -- it is the decode's
own band, which no bootstrap over the item sample can see
(`llb.eval.context_ablation.decoding_stability`).
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from llb.board.io import read_case_rows
from llb.core.config import RunConfig
from llb.eval.answer_quality.models import GROUNDING_DRAFTED, GROUNDING_VERIFIED
from llb.eval.answer_quality.run import select_items
from llb.eval.context_ablation.compare import compare_context_strategies
from llb.eval.context_ablation.decoding_stability import MIN_REPEATS, measure_decoding_stability
from llb.eval.context_ablation.lanes import default_lanes, lane_config
from llb.eval.context_ablation.models import (
    LANE_CLOSED_BOOK,
    ContextAblationReport,
    ContextWindowBinding,
    DecodingStabilityReport,
    LongContextPowerAnalysis,
)
from llb.eval.context_ablation.power import (
    DEFAULT_TARGET_POWER,
    plan_from_artifact,
    resolve_power_analysis,
    write_power_plan,
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


def lane_context_windows(
    run_dirs: Mapping[str, Sequence[str]],
) -> dict[str, ContextWindowBinding | None]:
    """Which window each lane's skips were measured against, read back off its run manifests.

    The binding is recorded by the run that did the skipping, not by the comparison, so it is read
    here rather than recomputed: a comparison assembled on another host, or months later, still
    reports the window the lane actually ran under. A lane with no manifest (an injected runner in
    CI) or no recorded binding (no document was ever checked) reports None.
    """
    windows: dict[str, ContextWindowBinding | None] = {}
    for label, dirs in run_dirs.items():
        windows[label] = next(
            (
                binding
                for run_dir in dirs
                if (binding := _manifest_context_window(Path(run_dir))) is not None
            ),
            None,
        )
    return windows


def _manifest_context_window(run_dir: Path) -> ContextWindowBinding | None:
    manifest = run_dir / "manifest.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    binding = payload.get("context_window") if isinstance(payload, dict) else None
    return cast(ContextWindowBinding, binding) if isinstance(binding, dict) else None


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
    power_reference: Path | None = None,
    minimum_detectable_delta: float | None = None,
    target_power: float = DEFAULT_TARGET_POWER,
    repeats: int = 1,
) -> ContextAblationRun:
    """Score the selected items under every context lane and persist the comparison."""
    selection = _selection(lanes, splits, repeats)
    items_by_split = select_items(config, splits, limit, verified_only)
    target = Path(out_dir) if out_dir is not None else default_out_dir(config)
    power_plan = _prepare_power_plan(
        power_reference,
        minimum_detectable_delta,
        target_power,
        confidence,
        sum(len(items) for items in items_by_split.values()),
    )
    if power_plan is not None:
        write_power_plan(power_plan, target / "power-plan.json")
    runner = run_lane or eval_lane_runner(verified_only=verified_only)
    passes = [
        score_lanes(config, selection, items_by_split, run_lane=runner) for _ in range(repeats)
    ]
    rows, run_dirs = passes[0]
    report = compare_context_strategies(
        rows,
        load_question_types(config.goldset_path),
        baseline=LANE_CLOSED_BOOK,
        run_dirs=run_dirs,
        context_windows=lane_context_windows(run_dirs),
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    if power_plan is not None:
        report["power_analysis"] = resolve_power_analysis(report, power_plan)
    if len(passes) >= MIN_REPEATS:
        report["decoding_stability"] = _stability(passes, selection, report)
    paths = write_artifacts(report, target, metadata=_metadata(config, splits, verified_only))
    if power_plan is not None:
        paths["power_plan"] = str(target / "power-plan.json")
    return ContextAblationRun(report, target, paths)


def _selection(lanes: Sequence[str] | None, splits: Sequence[str], repeats: int) -> list[str]:
    """The lane selection this run will score, or the reason it cannot be compared."""
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
    if repeats < 1:
        raise ValueError("a lane is scored at least once")
    return selection


def _stability(
    passes: Sequence[tuple[dict[str, CaseRows], dict[str, list[str]]]],
    selection: Sequence[str],
    report: ContextAblationReport,
) -> DecodingStabilityReport:
    """Band every lane's own numbers occupy across the repeated passes of this run."""
    return measure_decoding_stability(
        {label: [lane_rows[label] for lane_rows, _ in passes] for label in selection},
        report["item_ids"],
        report["derived"],
        run_dirs={label: [dirs[label] for _, dirs in passes] for label in selection},
        baseline=LANE_CLOSED_BOOK,
    )


def _prepare_power_plan(
    reference: Path | None,
    minimum_detectable_delta: float | None,
    target_power: float,
    confidence: float,
    planned_n: int,
) -> LongContextPowerAnalysis | None:
    if reference is None and minimum_detectable_delta is None:
        return None
    if reference is None or minimum_detectable_delta is None:
        raise ValueError("power planning needs both power_reference and minimum_detectable_delta")
    return plan_from_artifact(
        reference,
        minimum_detectable_delta=minimum_detectable_delta,
        target_power=target_power,
        confidence=confidence,
        planned_n=planned_n,
    )


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
