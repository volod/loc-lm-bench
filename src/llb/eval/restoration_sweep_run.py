"""Production wiring for the restoration constraint sweep: measure, decide, publish."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from llb.bench.common import new_run_timestamp
from llb.core.config import RunConfig
from llb.eval.restoration_sweep import (
    METHOD,
    SWEEP_STEPS,
    SWEEP_VARIANT_CLASSES,
    run_restoration_sweep,
    sweep_config,
)
from llb.eval.restoration_sweep_lanes import SweepResult
from llb.eval.restoration_sweep_report import write_sweep_artifacts
from llb.eval.restoration_sweep_verdict import ConstantVerdict, constant_verdicts
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES
from llb.rag.query_prep.restore_policy import DEFAULT_RESTORATION_POLICY, RestorationPolicy


@dataclass(frozen=True)
class RestorationSweepRun:
    """One published sweep: what was measured, what it decided, and where it landed."""

    result: SweepResult
    verdicts: tuple[ConstantVerdict, ...]
    out_dir: Path
    paths: Mapping[str, str]


def run_and_publish_sweep(
    config: RunConfig,
    *,
    split: str = "final",
    limit: int | None = None,
    typo_rate: float = 0.08,
    variant_classes: Sequence[str] = SWEEP_VARIANT_CLASSES,
    policies: Sequence[RestorationPolicy] = (DEFAULT_RESTORATION_POLICY,),
    dense_case: bool = False,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    progress: Callable[[str], None] | None = None,
) -> RestorationSweepRun:
    """Run every setting, read each constant's verdict, and publish the bundle atomically."""
    lane_config = sweep_config(config, dense_case)
    result = run_restoration_sweep(
        config,
        split=split,
        limit=limit,
        typo_rate=typo_rate,
        variant_classes=variant_classes,
        policies=policies,
        dense_case=dense_case,
        progress=progress,
    )
    verdicts = constant_verdicts(
        result, resamples=resamples, confidence=confidence, seed=lane_config.seed
    )
    _, stamp = new_run_timestamp()
    out_dir = lane_config.data_dir / METHOD / stamp
    metadata: dict[str, object] = {
        "goldset": str(lane_config.goldset_path),
        "corpus_root": str(lane_config.corpus_root),
        "split": split,
        "limit": limit,
        "embedding_model": lane_config.embedding_model,
        "seed": lane_config.seed,
        "top_k": lane_config.top_k,
        "typo_rate": typo_rate,
        "lane": ",".join(SWEEP_STEPS),
        "typo_guard": lane_config.query_prep_typo_guard,
        "query_prep_dense_case": lane_config.query_prep_dense_case,
        "variant_classes": list(result.variant_classes),
        "settings": [policy.as_metadata() for policy in result.policies],
        "resamples": resamples,
        "confidence": confidence,
    }
    paths = write_sweep_artifacts(result, verdicts, out_dir, metadata)
    return RestorationSweepRun(result, tuple(verdicts), out_dir, paths)
