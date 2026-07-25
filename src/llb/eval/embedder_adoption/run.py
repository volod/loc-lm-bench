"""Production wiring: score one item set end to end in every (cell, encoder) pair, then compare.

Each pair is an ORDINARY `run-eval` bundle under its encoder's own `$DATA_DIR/run-eval/` -- nothing
about its scoring is special-cased here, so any cell's numbers are reproducible by re-running
`run-eval` with that cell's config. The item set is selected ONCE from the baseline encoder's
config and handed to every cell and lane, which is what makes the pairing legitimate both inside a
cell and across cells.

`run_lane` is injectable, so the whole orchestration runs in CI with fake bundles -- no backend, no
store, no GPU.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from llb.board.io import read_case_rows
from llb.core.config import RunConfig
from llb.eval.answer_quality.run import GROUNDING_DRAFTED, GROUNDING_VERIFIED, LaneRunner
from llb.eval.embedder_adoption.cells import cell_config
from llb.eval.embedder_adoption.compare import CellRows, compare_cells
from llb.eval.embedder_adoption.models import AdoptionBarReport, CellSpec, EmbedderLane
from llb.eval.embedder_adoption.report import format_report
from llb.eval.paired_cases import CaseRows
from llb.goldset.schema import GoldItem
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED

if TYPE_CHECKING:
    from llb.eval.embedder_adoption.cross_model import CrossModelReport
    from llb.eval.embedder_adoption.roster import ModelProfile, RosterReport
    from llb.eval.embedder_adoption.screen import ScreenReport

METHOD = "embedder-adoption-bar"


@dataclass(frozen=True)
class AdoptionBarRun:
    report: AdoptionBarReport
    out_dir: Path
    paths: Mapping[str, str]


@dataclass(frozen=True)
class ScoredCells:
    """Every cell's per-encoder rows plus the run bundle each lane's rows were read from."""

    cells: list[tuple[CellSpec, CellRows]]
    run_dirs: dict[str, dict[str, list[str]]]


def score_cells(
    config: RunConfig,
    cells: Sequence[CellSpec],
    lanes: Sequence[EmbedderLane],
    items_by_split: Mapping[str, list[GoldItem]],
    *,
    run_lane: LaneRunner,
) -> ScoredCells:
    """Run every (cell, encoder) pair over the SAME items and read back its per-case rows.

    Several splits pool into ONE compared item set (one bundle each, so every bundle stays an
    ordinary per-split run), which is how the sweep can cover exactly the ledger the retrieval
    bake-off measured rather than a third of it.
    """
    scored: list[tuple[CellSpec, CellRows]] = []
    run_dirs: dict[str, dict[str, list[str]]] = {}
    for cell in cells:
        cell_rows: dict[str, CaseRows] = {}
        cell_dirs: dict[str, list[str]] = {}
        for lane in lanes:
            config_for_lane = cell_config(config, cell, lane)
            rows: CaseRows = []
            dirs: list[str] = []
            for split, items in items_by_split.items():
                scores = run_lane(config_for_lane, items, split)
                rows.extend(read_case_rows(scores))
                dirs.append(str(scores.parent))
            cell_rows[lane.model] = rows
            cell_dirs[lane.model] = dirs
        scored.append((cell, cell_rows))
        run_dirs[cell.label] = cell_dirs
    return ScoredCells(scored, run_dirs)


def run_adoption_bar_sweep(
    config: RunConfig,
    cells: Sequence[CellSpec],
    lanes: Sequence[EmbedderLane],
    *,
    splits: Sequence[str] = ("final",),
    limit: int | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    out_dir: Path | None = None,
    verified_only: bool = True,
    run_lane: LaneRunner | None = None,
) -> AdoptionBarRun:
    """Score every cell under both encoders and persist the per-cell comparison plus the verdict.

    `lanes[0]` is the incumbent the deltas are measured against, exactly as `--baseline` is in the
    retrieval bake-off: the question is whether to REPLACE that row.
    """
    from llb.eval.answer_quality.run import eval_lane_runner, select_items

    if len(lanes) != 2:
        raise ValueError("the sweep compares exactly two encoders: a baseline and a candidate")
    if not cells:
        raise ValueError("the sweep needs at least one cell")
    if not splits:
        raise ValueError("name at least one gold split to score")
    baseline, candidate = lanes
    if baseline.model == candidate.model:
        raise ValueError("the baseline and candidate encoders must differ")
    items_by_split = select_items(config, splits, limit, verified_only)
    scored = score_cells(
        config,
        cells,
        lanes,
        items_by_split,
        run_lane=run_lane or eval_lane_runner(verified_only=verified_only),
    )
    report = compare_cells(
        scored.cells,
        scored.run_dirs,
        baseline=baseline.model,
        candidate=candidate.model,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    target = Path(out_dir) if out_dir is not None else default_out_dir(config)
    paths = write_artifacts(report, target, metadata=_metadata(config, splits, verified_only))
    return AdoptionBarRun(report, target, paths)


def default_out_dir(config: RunConfig) -> Path:
    """`$DATA_DIR/embedder-adoption-bar/<timestamp>/`."""
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
    report: AdoptionBarReport, out_dir: Path, *, metadata: Mapping[str, object]
) -> dict[str, str]:
    """Persist `report.md` + `comparison.json` under the sweep directory."""
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


@dataclass(frozen=True)
class CrossModelRun:
    report: "CrossModelReport"
    out_dir: Path
    paths: Mapping[str, str]


def load_report(path: Path) -> AdoptionBarReport:
    """Load one finished sweep's `comparison.json` (or its directory) as an `AdoptionBarReport`."""
    target = Path(path)
    if target.is_dir():
        target = target / "comparison.json"
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{target}: cannot load an adoption-bar comparison: {exc}") from None
    if not isinstance(data, dict) or "cells" not in data or "verdict" not in data:
        raise ValueError(f"{target}: not an adoption-bar comparison.json (no cells/verdict)")
    return data  # type: ignore[return-value]


def run_cross_model_comparison(report_paths: Sequence[Path], *, out_dir: Path) -> CrossModelRun:
    """Read two finished sweeps and persist the per-cell cross-model agreement report."""
    from llb.eval.embedder_adoption.cross_model import (
        _cross_metadata,
        compare_models,
        format_cross_model,
    )

    reports = [load_report(path) for path in report_paths]
    cross = compare_models(reports)
    metadata = _cross_metadata(reports)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = {**cross, "metadata": dict(metadata)}
    (target / "cross_model.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "cross_model.md").write_text(
        format_cross_model(cross, metadata=metadata), encoding="utf-8"
    )
    paths = {
        "report": str(target / "cross_model.md"),
        "comparison": str(target / "cross_model.json"),
    }
    return CrossModelRun(cross, target, paths)


@dataclass(frozen=True)
class RosterRun:
    report: "RosterReport"
    out_dir: Path
    paths: Mapping[str, str]


def load_profiles(path: Path | None) -> dict[str, "ModelProfile"]:
    """Load the declared per-model profiles (`{model id: {params_b, family}}`) from JSON.

    Profiles are DECLARED, never inferred from the model id: a wrong guess at a parameter count
    would silently turn into a wrong claim about what predicts the reranker gain.
    """
    from llb.eval.embedder_adoption.roster import PROPERTIES

    if path is None:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot load model profiles: {exc}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path}: model profiles must be an object keyed by model id")
    profiles: dict[str, ModelProfile] = {}
    for model, profile in data.items():
        if not isinstance(profile, dict):
            raise ValueError(f"{path}: profile for {model!r} must be an object")
        unknown = sorted(set(profile) - set(PROPERTIES))
        if unknown:
            raise ValueError(
                f"{path}: profile for {model!r} declares unknown propert(ies) "
                f"{', '.join(unknown)}; expected any of {', '.join(PROPERTIES)}"
            )
        profiles[str(model)] = profile  # type: ignore[assignment]
    return profiles


def run_roster_comparison(
    report_paths: Sequence[Path],
    *,
    out_dir: Path,
    profiles_path: Path | None = None,
    focus_cell: str | None = None,
) -> RosterRun:
    """Read N finished sweeps and persist the roster property reading."""
    from llb.eval.embedder_adoption.cross_model import _cross_metadata
    from llb.eval.embedder_adoption.roster import (
        DEFAULT_FOCUS_CELL,
        compare_roster,
        format_roster,
    )

    reports = [load_report(path) for path in report_paths]
    roster = compare_roster(
        reports,
        load_profiles(profiles_path),
        focus_cell=focus_cell or DEFAULT_FOCUS_CELL,
    )
    metadata = _cross_metadata(reports)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = {**roster, "metadata": dict(metadata)}
    (target / "roster.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "roster.md").write_text(format_roster(roster, metadata=metadata), encoding="utf-8")
    paths = {
        "report": str(target / "roster.md"),
        "comparison": str(target / "roster.json"),
    }
    return RosterRun(roster, target, paths)


@dataclass(frozen=True)
class ScreenRun:
    report: "ScreenReport"
    out_dir: Path
    paths: Mapping[str, str]


def run_screen_study_over_paths(
    report_paths: Sequence[Path], *, out_dir: Path, focus_cell: str | None = None, **kwargs: object
) -> ScreenRun:
    """Read finished sweeps and persist the per-model screen cost study."""
    from llb.eval.embedder_adoption.cross_model import _cross_metadata
    from llb.eval.embedder_adoption.roster import DEFAULT_FOCUS_CELL
    from llb.eval.embedder_adoption.screen import format_screen, run_screen_study

    reports = [load_report(path) for path in report_paths]
    study = run_screen_study(
        reports,
        focus_cell=focus_cell or DEFAULT_FOCUS_CELL,
        **kwargs,  # type: ignore[arg-type]
    )
    metadata = _cross_metadata(reports)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = {**study, "metadata": dict(metadata)}
    (target / "screen.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "screen.md").write_text(format_screen(study, metadata=metadata), encoding="utf-8")
    paths = {
        "report": str(target / "screen.md"),
        "comparison": str(target / "screen.json"),
    }
    return ScreenRun(study, target, paths)
