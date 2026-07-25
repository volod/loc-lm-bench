"""How cheaply can an operator decide the reranker question for THEIR model?

The roster established that whether `bge-m3`'s first-hit-rank gain reaches the answer under a
cross-encoder is not predictable from a model card, so the advice is "measure it on the model you
ship". That advice is only useful if measuring is cheap. A full sweep is 24 `run-eval` bundles
(4 cells x 2 encoders x 3 splits); the reranker question lives in ONE cell, so dropping the other
three is an exact 4x saving with no statistical cost at all -- the surviving cell is byte-identical
to the one the full sweep would have run.

What is NOT free is shrinking the ITEM set, and this module measures that honestly. Fewer items
widen the paired interval, and the reading rule is "the interval clears zero", so a screen can only
lose detections, never invent them. The study resamples the recorded per-item deltas at a range of
item counts and reports, per model, how often a screen of that size reproduces the full-set reading
-- so the answer is a measured floor rather than a guess.

Per-item deltas are re-derived from each sweep's own `run-eval` bundles (the sweep artifact stores
aggregates only), and the re-derivation is CHECKED against the sweep's recorded reading before any
subsampling: a pipeline that cannot reproduce the full-set answer has no business estimating a
smaller one.

Pure Python and dependency-free, like `fusion_evidence.stats`, so the study imports and is
unit-tested in the lightweight CI install -- no numpy, no backend, no GPU.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from random import Random

from typing_extensions import NotRequired, TypedDict

from llb.eval.embedder_adoption.compare import with_reciprocal_rank
from llb.eval.embedder_adoption.cross_model import model_id
from llb.eval.embedder_adoption.models import (
    DEFAULT_FOCUS_CELL,
    METRIC_OBJECTIVE,
    METRIC_RECIPROCAL_RANK,
    AdoptionBarReport,
    ItemDeltas,
)
from llb.eval.embedder_adoption.stability import reading_from_deltas
from llb.rag.fusion_evidence.stability import reading_label
from llb.rag.fusion_evidence.stats import DEFAULT_CONFIDENCE, DEFAULT_SEED, bootstrap_index_sets

# Item counts the screen is measured at. The full accepted ledger is 40, so these span "one split's
# worth" (~13) up to nearly the whole set.
DEFAULT_SCREEN_SIZES = (10, 15, 20, 25, 30, 35)
# Subsamples drawn per size, and resamples per subsample. Deliberately modest: this is a pure-Python
# nested bootstrap, and the agreement rate it estimates is a coarse quantity.
DEFAULT_DRAWS = 120
DEFAULT_STUDY_RESAMPLES = 300
# A screen size is "good enough" when it reproduces the full-set reading at least this often.
DEFAULT_TARGET_AGREEMENT = 0.9

# Some size below the full item set reproduces every model's full-set reading at the target rate.
DECISION_SCREEN_SUPPORTED = "screen_supported"
# No size below the full set clears the target: the item count cannot be cut, only the cell grid.
DECISION_FULL_SET_REQUIRED = "full_set_required"

METRICS = (METRIC_OBJECTIVE, METRIC_RECIPROCAL_RANK)


class SizeAgreement(TypedDict):
    """How often a screen of `size` items reproduces the full-set reading."""

    size: int
    agreement: float
    readings: dict[str, int]  # READING_* -> how many draws produced it


class ModelScreen(TypedDict):
    """One model's full-set reading plus the agreement curve over screen sizes."""

    model: str
    n: int
    full_reading: str
    # The sweep's own recorded reading; equal to `full_reading` unless the re-derivation drifted.
    recorded_reading: str
    reproduced: bool
    sizes: list[SizeAgreement]
    # Smallest measured size clearing the target agreement, or None.
    min_size: int | None


class ScreenVerdict(TypedDict):
    """What the study says the per-model decision actually costs."""

    decision: str
    focus_cell: str
    target_agreement: float
    # Smallest size clearing the target for EVERY model, or None when some model never does.
    min_size: int | None
    full_n: int
    bundles_full_grid: int
    bundles_focus_cell: int
    reason: str


class ScreenReport(TypedDict):
    """The screen study artifact: per-model agreement curves and the cost verdict."""

    focus_cell: str
    baseline: str
    candidate: str
    sizes: list[int]
    draws: int
    resamples: int
    confidence: float
    seed: int
    models: list[ModelScreen]
    verdict: ScreenVerdict
    metadata: NotRequired[dict[str, object]]


def cell_item_deltas(report: AdoptionBarReport, cell_label: str) -> ItemDeltas:
    """Re-derive one cell's per-item deltas from the sweep's own `run-eval` bundles.

    The sweep artifact persists aggregates, so the item-level vectors come back from the bundles it
    names. Rows are keyed by item id and intersected across the two encoders' bundles, so a lane
    that scored a different set fails loudly rather than pairing mismatched items.
    """
    cell = next((entry for entry in report["cells"] if entry["label"] == cell_label), None)
    if cell is None:
        raise ValueError(
            f"cell {cell_label!r} is not in this sweep "
            f"({', '.join(entry['label'] for entry in report['cells'])})"
        )
    baseline, candidate = report["baseline"], report["candidate"]
    rows = {
        model: _bundle_rows(cell["lanes"][model]["run_dirs"]) for model in (baseline, candidate)
    }
    # The two lanes are handed the SAME items by construction, so a difference means a bundle is
    # partial. Intersecting it away would compute the delta on a different set than the sweep
    # published, so it is refused rather than quietly narrowed.
    if set(rows[baseline]) != set(rows[candidate]):
        missing = sorted(set(rows[baseline]) - set(rows[candidate]))
        extra = sorted(set(rows[candidate]) - set(rows[baseline]))
        raise ValueError(
            f"cell {cell_label!r}: the encoders scored different item sets -- "
            f"{candidate} did not score {missing[:3]}, {baseline} did not score {extra[:3]}"
        )
    ids = sorted(rows[baseline])
    if not ids:
        raise ValueError(f"cell {cell_label!r}: the bundles carry no scored item")
    return ItemDeltas(
        item_ids=ids,
        objective=[
            _value(rows[candidate][i], METRIC_OBJECTIVE)
            - _value(rows[baseline][i], METRIC_OBJECTIVE)
            for i in ids
        ],
        reciprocal_rank=[
            _value(rows[candidate][i], METRIC_RECIPROCAL_RANK)
            - _value(rows[baseline][i], METRIC_RECIPROCAL_RANK)
            for i in ids
        ],
    )


def _value(row: Mapping[str, object], metric: str) -> float:
    return float(row.get(metric) or 0.0)  # type: ignore[arg-type]


def _bundle_rows(run_dirs: Sequence[str]) -> dict[str, Mapping[str, object]]:
    """Every per-case row of one lane's bundles, keyed by item id and carrying reciprocal rank."""
    from llb.board.io import read_case_rows

    rows: dict[str, Mapping[str, object]] = {}
    for run_dir in run_dirs:
        scores = Path(run_dir) / "scores.jsonl"
        if not scores.is_file():
            raise ValueError(f"missing run bundle scores: {scores}")
        for row in with_reciprocal_rank(list(read_case_rows(scores))):
            rows[str(row["item_id"])] = row
    return rows


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


def format_screen(
    report: ScreenReport,
    *,
    metadata: Mapping[str, object] | None = None,
    title: str = "Embedder adoption bar: what does a per-model screen cost?",
) -> str:
    """The screen Markdown artifact: the cost verdict, the agreement curves, the reading check."""
    verdict = report["verdict"]
    meta = dict(metadata or {})
    lines = [f"# {title}", ""]
    for key in ("split", "grounding", "goldset", "corpus"):
        if key in meta:
            lines.append(f"- {key}: `{meta[key]}`")
    lines += [
        f"- encoder pair: `{report['candidate']}` vs `{report['baseline']}`",
        f"- focus cell: `{report['focus_cell']}` (the reranker question)",
        f"- full item set: {verdict['full_n']} items; "
        f"full grid {verdict['bundles_full_grid']} bundles, focus cell alone "
        f"{verdict['bundles_focus_cell']}",
        f"- study: {report['draws']} subsamples per size, {report['resamples']} resamples each, "
        f"seed {report['seed']}, target agreement {verdict['target_agreement']:.0%}",
        f"- verdict: **{verdict['decision']}** -- {verdict['reason']}",
        "",
        "### Screen agreement with the full-set reading",
        "",
        "How often a screen of N items reproduces the same `k10+rerank` reading the whole ledger "
        'gives. Fewer items widen the paired interval and the rule is "the interval clears zero", '
        "so a screen can only LOSE a detection, never invent one.",
        "",
        "| model | full reading | "
        + " | ".join(f"n={size}" for size in report["sizes"])
        + " | min size |",
        "| --- | :-: | " + " | ".join(["---:"] * len(report["sizes"])) + " | :-: |",
    ]
    for entry in report["models"]:
        by_size = {curve["size"]: curve["agreement"] for curve in entry["sizes"]}
        cells = " | ".join(
            f"{by_size[size]:.0%}" if size in by_size else "-" for size in report["sizes"]
        )
        floor = str(entry["min_size"]) if entry["min_size"] is not None else "none"
        short = entry["model"].split("/")[-1]
        reading = reading_label(entry["full_reading"])
        lines.append(f"| `{short}` | {reading} | {cells} | {floor} |")
    lines += [
        "",
        "Every row's re-derived full-set reading matches the reading its own sweep recorded, so "
        "the curves above are estimated from vectors that reproduce the published answer.",
        "",
    ]
    return "\n".join(lines) + "\n"


def format_screen_summary(report: ScreenReport) -> str:
    """One-screen ASCII summary for the terminal."""
    verdict = report["verdict"]
    lines = [
        f"[compare-adoption-screen] focus={report['focus_cell']} "
        f"full_n={verdict['full_n']} models={len(report['models'])}",
        f"  {'model'.ljust(34)} {'reading':>10} "
        + " ".join(f"{'n=' + str(s):>6}" for s in report["sizes"])
        + "   min",
    ]
    for entry in report["models"]:
        by_size = {curve["size"]: curve["agreement"] for curve in entry["sizes"]}
        cells = " ".join(
            f"{by_size[size]:>5.0%}" if size in by_size else f"{'-':>6}" for size in report["sizes"]
        )
        floor = str(entry["min_size"]) if entry["min_size"] is not None else "none"
        lines.append(
            f"  {entry['model'].split('/')[-1][:34].ljust(34)} "
            f"{reading_label(entry['full_reading']):>10} "
            f"{cells}   {floor:>4}"
        )
    lines.append(f"  verdict: {verdict['decision'].upper()} -- {verdict['reason']}")
    return "\n".join(lines)
