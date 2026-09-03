"""Production wiring: score one item set under each validation lane, then compare.

Every lane is an ORDINARY `run-eval` bundle under `$DATA_DIR/run-eval/`, so a lane's numbers are
reproducible by re-running `run-eval` with that lane's config and nothing about its scoring is
special-cased here. The `off` lane is literally the shipped free-text path with no new knob set,
which is what lets it reproduce a recorded run bundle rather than merely resemble one.

`run_lane` is injectable, so the whole orchestration runs in CI with fake bundles -- no backend,
no store, no GPU.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.artifacts.runs.bundle import read_case_rows
from llb.core.config import RunConfig
from llb.eval.answer_validation.constants import (
    COMPARISON_FILENAME,
    LANE_OFF,
    LANE_PYDANTIC_ONTOLOGY,
    METHOD_DIR,
    REPORT_FILENAME,
    RUN_NAME_PREFIX,
    VALIDATION_LANES,
)
from llb.eval.answer_validation.report import format_report
from llb.eval.answer_validation.study import analyze, with_references
from llb.eval.paired_cases import CaseRows
from llb.goldset.schema import GoldItem
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED

# One lane config + one split's items -> that (lane, split) bundle's persisted `scores.jsonl`.
LaneRunner = Callable[[RunConfig, list[GoldItem], str], Path]


@dataclass(frozen=True)
class AnswerValidationRun:
    report: dict[str, Any]
    out_dir: Path
    paths: Mapping[str, str]


def parse_lanes(spec: str) -> list[str]:
    """Parse the comma-separated lane selection, de-duplicated in the order given."""
    labels = [token.strip() for token in spec.split(",") if token.strip()]
    unknown = [label for label in labels if label not in VALIDATION_LANES]
    if unknown:
        raise ValueError(
            f"unknown validation lane(s) {unknown}; expected any of {list(VALIDATION_LANES)}"
        )
    ordered = list(dict.fromkeys(labels))
    if len(ordered) < 2:
        raise ValueError("name at least two lanes: a baseline and a candidate")
    if ordered[0] != LANE_OFF:
        raise ValueError(f"the first lane must be the {LANE_OFF!r} baseline, got {ordered[0]!r}")
    return ordered


def lane_config(
    config: RunConfig,
    lane: str,
    axioms: Path | None,
    ledger: Path | None,
    overlay: Path | None = None,
) -> RunConfig:
    """`config` with this lane's answer-contract knobs applied and a lane-identifying run name.

    Built by revalidating an explicit field mapping rather than `with_overrides`, because the `off`
    lane must be able to set the two ontology paths back to `None` -- `with_overrides` drops `None`
    by design, and a baseline carrying the gate's paths is refused by `RunConfig` itself.
    """
    values = config.model_dump()
    values.update(
        run_name=f"{RUN_NAME_PREFIX}-{lane}",
        answer_format="envelope" if lane != LANE_OFF else "free_text",
        answer_validation="ontology" if lane == LANE_PYDANTIC_ONTOLOGY else "off",
        ontology_axioms=axioms if lane == LANE_PYDANTIC_ONTOLOGY else None,
        ontology_ledger=ledger if lane == LANE_PYDANTIC_ONTOLOGY else None,
        ontology_overlay=overlay if lane == LANE_PYDANTIC_ONTOLOGY else None,
    )
    return RunConfig.model_validate(values)


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
    axioms: Path | None,
    ledger: Path | None,
    overlay: Path | None,
    run_lane: LaneRunner,
) -> tuple[dict[str, CaseRows], dict[str, list[str]]]:
    """Run every lane over the SAME items and read back its per-case rows."""
    rows: dict[str, CaseRows] = {}
    run_dirs: dict[str, list[str]] = {}
    for lane in lanes:
        config_for_lane = lane_config(config, lane, axioms, ledger, overlay)
        lane_rows: CaseRows = []
        lane_dirs: list[str] = []
        for split, items in items_by_split.items():
            scores = run_lane(config_for_lane, items, split)
            lane_rows.extend(read_case_rows(scores))
            lane_dirs.append(str(scores.parent))
        rows[lane] = lane_rows
        run_dirs[lane] = lane_dirs
    return rows, run_dirs


def run_answer_validation(
    config: RunConfig,
    lanes: Sequence[str],
    *,
    axioms: Path | None = None,
    ledger: Path | None = None,
    overlay: Path | None = None,
    splits: Sequence[str] = ("final",),
    limit: int | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    out_dir: Path | None = None,
    verified_only: bool = True,
    run_lane: LaneRunner | None = None,
) -> AnswerValidationRun:
    """Score the selected items under every validation lane and persist the comparison."""
    from llb.eval.answer_quality.run import select_items

    if LANE_PYDANTIC_ONTOLOGY in lanes and (axioms is None or ledger is None):
        raise ValueError(
            f"the {LANE_PYDANTIC_ONTOLOGY!r} lane needs a SIGNED axiom file and the corpus "
            "extraction ledger; name both rather than running an ungated lane under its name"
        )
    items_by_split = select_items(config, splits, limit, verified_only)
    references = {
        item.id: item.reference_answer for items in items_by_split.values() for item in items
    }
    rows, run_dirs = score_lanes(
        config,
        lanes,
        items_by_split,
        axioms=axioms,
        ledger=ledger,
        overlay=overlay,
        run_lane=run_lane or eval_lane_runner(verified_only=verified_only),
    )
    report = analyze(
        {lane: with_references(lane_rows, references) for lane, lane_rows in rows.items()},
        baseline=LANE_OFF,
        run_dirs=run_dirs,
        gated_lane=LANE_PYDANTIC_ONTOLOGY if LANE_PYDANTIC_ONTOLOGY in rows else None,
        references=references,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    target = Path(out_dir) if out_dir is not None else default_out_dir(config)
    metadata = _metadata(config, lanes, splits, axioms, ledger, overlay)
    paths = write_artifacts(report, target, metadata=metadata)
    return AnswerValidationRun(report, target, paths)


def default_out_dir(config: RunConfig) -> Path:
    """`$DATA_DIR/answer-validation/<timestamp>/`."""
    from llb.core.store_generations import generation_timestamp

    return config.data_dir / METHOD_DIR / generation_timestamp()


def _metadata(
    config: RunConfig,
    lanes: Sequence[str],
    splits: Sequence[str],
    axioms: Path | None,
    ledger: Path | None,
    overlay: Path | None,
) -> dict[str, Any]:
    return {
        "model": config.model,
        "backend": config.backend,
        "split": ",".join(splits),
        "goldset": str(config.goldset_path),
        "lanes": ",".join(lanes),
        "axioms": str(axioms) if axioms is not None else "-",
        "ledger": str(ledger) if ledger is not None else "-",
        "overlay": str(overlay) if overlay is not None else "-",
        "max_tokens": config.max_tokens,
        "top_k": config.top_k,
    }


def write_artifacts(
    report: Mapping[str, Any], out_dir: Path, *, metadata: Mapping[str, Any]
) -> dict[str, str]:
    """Persist `report.md` + `comparison.json` under the comparison directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {**report, "metadata": dict(metadata)}
    (out_dir / COMPARISON_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / REPORT_FILENAME).write_text(
        format_report(report, metadata=metadata), encoding="utf-8"
    )
    return {
        "report": str(out_dir / REPORT_FILENAME),
        "comparison": str(out_dir / COMPARISON_FILENAME),
    }
