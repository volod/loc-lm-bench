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

from llb.artifacts.runs.bundle import read_case_rows
from llb.core.config import RunConfig
from llb.eval.answer_quality.models import GROUNDING_DRAFTED, GROUNDING_VERIFIED
from llb.eval.answer_quality.run import LaneRunner
from llb.eval.embedder_adoption.cells import cell_config
from llb.eval.embedder_adoption.compare import CellRows, compare_cells
from llb.eval.embedder_adoption.models import AdoptionBarReport, CellSpec, EmbedderLane
from llb.eval.embedder_adoption.report import format_report
from llb.eval.paired_cases import CaseRows
from llb.goldset.schema import GoldItem
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED

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
