"""Production wiring: score one item set end to end under several retrieval lanes, then compare.

Each lane is an ORDINARY `run-eval` bundle under `$DATA_DIR/run-eval/` -- nothing about the lane's
scoring is special-cased here, so a lane's numbers are reproducible by re-running `run-eval` with
that lane's config. The item set is selected ONCE and handed to every lane, which is what makes the
per-item pairing in the comparison legitimate.

`run_lane` is injectable, so the whole orchestration runs in CI with fake bundles -- no backend, no
store, no GPU.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from llb.board.io import read_case_rows
from llb.core.config import RunConfig
from llb.eval.answer_quality.budgets import conversion_baselines, expand_budget_lanes
from llb.eval.answer_quality.bundle_match import RUN_NAME_PREFIX
from llb.eval.answer_quality.compare import CaseRows, compare_answer_quality
from llb.eval.answer_quality.conversion import budget_conversion
from llb.eval.answer_quality.coverage import read_case_coverage, with_coverage
from llb.eval.answer_quality.lanes import lane_config
from llb.eval.answer_quality.models import (
    FOCUS_SLICE,
    GROUNDING_DRAFTED,
    GROUNDING_VERIFIED,
    AnswerQualityReport,
    LaneSpec,
)
from llb.eval.answer_quality.report import format_report
from llb.executor.runner_setup import _select_eval_items
from llb.goldset.schema import GoldItem
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED
from llb.rag.question_types import load_question_types

METHOD = "graph-vector-fusion-multihop"
ARTIFACT_SUBDIR = "answer-quality"

# One lane config + one split's items -> that (lane, split) bundle's persisted `scores.jsonl`.
LaneRunner = Callable[[RunConfig, list[GoldItem], str], Path]


@dataclass(frozen=True)
class AnswerQualityRun:
    report: AnswerQualityReport
    out_dir: Path
    paths: Mapping[str, str]


def eval_lane_runner(*, verified_only: bool = True) -> LaneRunner:
    """The default lane runner: one ordinary `run-eval` bundle per (lane, split)."""

    def run_lane(config: RunConfig, items: list[GoldItem], split: str) -> Path:
        from llb.executor.runner import run_eval

        result = run_eval(config, items=items, split=split, verified_only=verified_only)
        return Path(str(result["paths"]["scores"]))

    return run_lane


def select_items(
    config: RunConfig, splits: Sequence[str], limit: int | None, verified_only: bool
) -> dict[str, list[GoldItem]]:
    """The scored items per split; a named split that selects nothing is an error, not an empty
    lane, because the pooled comparison would then silently be a different item set."""
    selected = {
        split: _select_eval_items(config, None, split, limit, verified_only) for split in splits
    }
    empty = [split for split, items in selected.items() if not items]
    if empty:
        raise SystemExit(
            f"no {'verified ' if verified_only else ''}items in split(s) "
            f"{', '.join(empty)} of {config.goldset_path}"
        )
    return selected


def score_lanes(
    config: RunConfig,
    lanes: Sequence[LaneSpec],
    items_by_split: Mapping[str, list[GoldItem]],
    *,
    run_lane: LaneRunner,
) -> tuple[dict[str, CaseRows], dict[str, list[str]]]:
    """Run every lane over the SAME items, then read back its per-case rows plus coverage.

    Several splits pool into ONE compared item set (one run bundle each, so every bundle stays an
    ordinary per-split run), which is how the comparison can cover exactly the ledger a retrieval
    sweep measured rather than a third of it.

    The multi-span coverage columns are recomputed from each bundle's retrieval sidecar, so the
    comparison can pair the metric the fusion sweep decides on (`all-spans@k`) against the answers
    the model produced from that same context.
    """
    rows: dict[str, CaseRows] = {}
    run_dirs: dict[str, list[str]] = {}
    for lane in lanes:
        config_for_lane = lane_config(config, lane, run_name_prefix=RUN_NAME_PREFIX)
        lane_rows: CaseRows = []
        lane_dirs: list[str] = []
        for split, items in items_by_split.items():
            scores = run_lane(config_for_lane, items, split)
            # The lane's OWN budget: a budget sweep's cells differ in exactly this, so reading
            # every cell's coverage at the base config's `top_k` would erase the thing measured.
            coverage = read_case_coverage(scores.parent, config_for_lane.top_k)
            lane_rows.extend(with_coverage(list(read_case_rows(scores)), coverage))
            lane_dirs.append(str(scores.parent))
        rows[lane.label] = lane_rows
        run_dirs[lane.label] = lane_dirs
    return rows, run_dirs


def run_answer_quality(
    config: RunConfig,
    lanes: Sequence[LaneSpec],
    *,
    splits: Sequence[str] = ("final",),
    limit: int | None = None,
    focus_slice: str = FOCUS_SLICE,
    budgets: Sequence[int] = (),
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    out_dir: Path | None = None,
    verified_only: bool = True,
    run_lane: LaneRunner | None = None,
) -> AnswerQualityRun:
    """Score the selected items under every lane and persist the per-slice comparison.

    `budgets` turns the run into a `(lane x top_k)` sweep: every row is scored at every named
    retrieval budget and each raised-budget cell is additionally read against the SAME row at the
    smallest one, which is the reading that says whether a retrieval-budget coverage gain converts
    into answers. An empty selection leaves the config's own `top_k` alone.

    `verified_only=False` grounds the comparison on a DRAFTED ledger, which is what makes it
    readable beside a retrieval sweep measured on the same drafted set; the grounding is recorded
    in every artifact so the two can never be confused.
    """
    if len(lanes) < 2:
        raise ValueError("the comparison needs a baseline lane and at least one candidate lane")
    if not splits:
        raise ValueError("name at least one gold split to score")
    cells = expand_budget_lanes(lanes, budgets) if budgets else list(lanes)
    cross = conversion_baselines(lanes, budgets) if budgets else {}
    items_by_split = select_items(config, splits, limit, verified_only)
    rows, run_dirs = score_lanes(
        config,
        cells,
        items_by_split,
        run_lane=run_lane or eval_lane_runner(verified_only=verified_only),
    )
    report = compare_answer_quality(
        rows,
        load_question_types(config.goldset_path),
        baseline=cells[0].label,
        run_dirs=run_dirs,
        focus_slice=focus_slice,
        cross_baselines=cross,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    if cross:
        report["budget_conversion"] = budget_conversion(
            report["cross_readings"],
            budgets=budgets,
            focus_slice=focus_slice,
            coverage=report["verdict"]["coverage_metric"],
            confidence=confidence,
        )
    target = Path(out_dir) if out_dir is not None else default_out_dir(config)
    metadata = _metadata(config, splits, verified_only)
    metadata["top_k"] = ",".join(str(budget) for budget in budgets) if budgets else config.top_k
    paths = write_artifacts(report, target, metadata=metadata)
    return AnswerQualityRun(report, target, paths)


def default_out_dir(config: RunConfig) -> Path:
    """`$DATA_DIR/graph-vector-fusion-multihop/<timestamp>/answer-quality/`."""
    from llb.core.store_generations import generation_timestamp

    return config.data_dir / METHOD / generation_timestamp() / ARTIFACT_SUBDIR


def _metadata(config: RunConfig, splits: Sequence[str], verified_only: bool) -> dict[str, object]:
    return {
        "model": config.model,
        "backend": config.backend,
        "split": ",".join(splits),
        "goldset": str(config.goldset_path),
        "grounding": GROUNDING_VERIFIED if verified_only else GROUNDING_DRAFTED,
    }


def write_artifacts(
    report: AnswerQualityReport, out_dir: Path, *, metadata: Mapping[str, object]
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
