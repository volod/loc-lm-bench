"""How cheaply can an operator decide the reranker question for THEIR model?

The roster established that whether `bge-m3`'s first-hit-rank gain reaches the answer under a
cross-encoder is not predictable from a model card, so the advice is "measure it on the model you
ship". That advice is only useful if measuring is cheap. A full sweep is 24 `run-eval` bundles
(4 cells x 2 encoders x 3 splits); the reranker question lives in ONE cell, so dropping the other
three is an exact 4x saving with no statistical cost at all -- the surviving cell is byte-identical
to the one the full sweep would have run.

What is NOT free is shrinking the ITEM set, and this module measures that honestly. A subsample can
either lose or invent a separation under the calibrated randomization rule. The study resamples
the recorded per-item deltas at a range of item counts and reports, per model, how often a screen
of that size reproduces the full-set reading -- so the answer is a measured floor rather than a
guess.

Per-item deltas are re-derived from each sweep's own `run-eval` bundles (the sweep artifact stores
aggregates only), and the re-derivation is CHECKED against the sweep's recorded reading before any
subsampling: a pipeline that cannot reproduce the full-set answer has no business estimating a
smaller one.

Pure Python and dependency-free, like `fusion_evidence.stats`, so the study imports and is
unit-tested in the lightweight CI install -- no numpy, no backend, no GPU.
"""

from collections.abc import Sequence
from random import Random

from llb.eval.embedder_adoption.cross_model import model_id
from llb.eval.embedder_adoption.models import (
    DEFAULT_FOCUS_CELL,
    AdoptionBarReport,
    ItemDeltas,
)
from llb.eval.embedder_adoption.screen_data import cell_item_deltas
from llb.eval.embedder_adoption.screen_models import (
    DECISION_FULL_SET_REQUIRED,
    DECISION_SCREEN_SUPPORTED,
    DEFAULT_DRAWS,
    DEFAULT_SCREEN_SIZES,
    DEFAULT_STUDY_RESAMPLES,
    DEFAULT_TARGET_AGREEMENT,
    ModelScreen,
    ScreenReport,
    ScreenVerdict,
    SizeAgreement,
)
from llb.eval.embedder_adoption.stability import reading_from_deltas
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_SEED, bootstrap_index_sets


def screen_model(
    model: str,
    deltas: ItemDeltas,
    recorded_reading: str,
    *,
    sizes: Sequence[int] = DEFAULT_SCREEN_SIZES,
    draws: int = DEFAULT_DRAWS,
    resamples: int = DEFAULT_STUDY_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    target: float = DEFAULT_TARGET_AGREEMENT,
) -> ModelScreen:
    """Measure how often a screen of each size reproduces this model's full-set reading."""
    n = len(deltas)
    full_reading = reading_from_deltas(deltas, bootstrap_index_sets(n, resamples, seed), confidence)
    rng = Random(seed)
    curves: list[SizeAgreement] = []
    for size in sizes:
        if size >= n:
            continue
        index_sets = bootstrap_index_sets(size, resamples, seed)
        counts: dict[str, int] = {}
        for _ in range(draws):
            reading = reading_from_deltas(
                deltas.take(rng.sample(range(n), size)), index_sets, confidence
            )
            counts[reading] = counts.get(reading, 0) + 1
        curves.append(
            {
                "size": size,
                "agreement": counts.get(full_reading, 0) / draws if draws else 0.0,
                "readings": counts,
            }
        )
    clearing = [entry["size"] for entry in curves if entry["agreement"] >= target]
    return {
        "model": model,
        "n": n,
        "full_reading": full_reading,
        "recorded_reading": recorded_reading,
        "reproduced": full_reading == recorded_reading,
        "sizes": curves,
        "min_size": min(clearing) if clearing else None,
    }


def decide_screen(
    models: Sequence[ModelScreen],
    *,
    focus_cell: str,
    target: float,
    bundles_full_grid: int,
    bundles_focus_cell: int,
) -> ScreenVerdict:
    """State the cheapest honest per-model screen: how many cells, and how many items.

    The cell saving is exact and unconditional, so it is reported either way. The ITEM saving is
    only claimed when every model's reading survives it -- a screen that reproduces four models and
    loses the fifth is exactly the screen that would tell an operator the reranker does not pay when
    it does.
    """
    full_n = max((entry["n"] for entry in models), default=0)
    per_model = [entry["min_size"] for entry in models]
    verdict: ScreenVerdict = {
        "decision": DECISION_FULL_SET_REQUIRED,
        "focus_cell": focus_cell,
        "target_agreement": target,
        "min_size": None,
        "full_n": full_n,
        "bundles_full_grid": bundles_full_grid,
        "bundles_focus_cell": bundles_focus_cell,
        "reason": "",
    }
    saving = (
        f"scoring only `{focus_cell}` cuts the run from {bundles_full_grid} to "
        f"{bundles_focus_cell} run-eval bundles "
        f"({bundles_full_grid // bundles_focus_cell if bundles_focus_cell else 0}x) with no "
        "statistical cost, since it is the same cell the full grid would have scored"
    )
    if models and all(size is not None for size in per_model):
        smallest = max(size for size in per_model if size is not None)
        verdict["decision"] = DECISION_SCREEN_SUPPORTED
        verdict["min_size"] = smallest
        verdict["reason"] = (
            f"{smallest} of {full_n} items reproduces every model's `{focus_cell}` reading at "
            f"least {target:.0%} of the time, so a screen may drop to {smallest} items; {saving}"
        )
        return verdict
    lost = [entry["model"] for entry in models if entry["min_size"] is None]
    verdict["reason"] = (
        f"no measured item count below {full_n} reproduces the `{focus_cell}` reading at "
        f"{target:.0%} for {', '.join(f'`{m}`' for m in lost)}, so the ITEM set cannot be cut -- "
        f"the decision needs the whole ledger. What is free is the grid: {saving}"
    )
    return verdict


def run_screen_study(
    reports: Sequence[AdoptionBarReport],
    *,
    focus_cell: str = DEFAULT_FOCUS_CELL,
    sizes: Sequence[int] = DEFAULT_SCREEN_SIZES,
    draws: int = DEFAULT_DRAWS,
    resamples: int = DEFAULT_STUDY_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    target: float = DEFAULT_TARGET_AGREEMENT,
) -> ScreenReport:
    """Measure the screen's agreement curve on every recorded sweep and decide what it costs."""
    from llb.eval.embedder_adoption.cross_model import cell_reading

    if not reports:
        raise ValueError("the screen study needs at least one recorded sweep")
    screens: list[ModelScreen] = []
    for report in reports:
        cell = next((c for c in report["cells"] if c["label"] == focus_cell), None)
        if cell is None:
            raise ValueError(
                f"cell {focus_cell!r} is not in the sweep for {model_id(report)!r} "
                f"({', '.join(c['label'] for c in report['cells'])})"
            )
        screens.append(
            screen_model(
                model_id(report),
                cell_item_deltas(report, focus_cell),
                cell_reading(cell),
                sizes=sizes,
                draws=draws,
                resamples=resamples,
                confidence=confidence,
                seed=seed,
                target=target,
            )
        )
    drifted = [entry["model"] for entry in screens if not entry["reproduced"]]
    if drifted:
        raise ValueError(
            "re-derived per-item deltas do not reproduce the recorded reading for "
            f"{', '.join(drifted)}; the study cannot estimate a smaller set from vectors that "
            "disagree with the sweep they came from"
        )
    reference = reports[0]
    return {
        "focus_cell": focus_cell,
        "baseline": reference["baseline"],
        "candidate": reference["candidate"],
        "sizes": [size for size in sizes if size < max(e["n"] for e in screens)],
        "draws": draws,
        "resamples": resamples,
        "confidence": confidence,
        "seed": seed,
        "models": screens,
        "verdict": decide_screen(
            screens,
            focus_cell=focus_cell,
            target=target,
            bundles_full_grid=_bundle_count(reference),
            bundles_focus_cell=_bundle_count(reference, focus_cell),
        ),
    }


def _bundle_count(report: AdoptionBarReport, only_cell: str | None = None) -> int:
    """How many `run-eval` bundles a sweep costs, in total or for one cell."""
    return sum(
        len(lane["run_dirs"])
        for cell in report["cells"]
        if only_cell is None or cell["label"] == only_cell
        for lane in cell["lanes"].values()
    )
