"""Carve the held-out dev sub-slice from inside the tuning split -- plain (seeded shuffle) or
base-score stratified (every non-empty bucket represented, answerable items guaranteed).

No trial ever trains on the dev ids; the objective scores them straight from the goldset. The
`_dev_items` resolver here loads those dev ids back as `GoldItem`s for the default objective.
"""

import json
from pathlib import Path
from random import Random

from llb.finetune.hparam_search.model import (
    BUCKET_PRIORITY,
    BUCKET_UNSCORED,
    BUCKET_ZERO,
    DEFAULT_DEV_FRACTION,
    DEFAULT_SEED,
    HIGH_SCORE_BOUNDARY,
    MIN_SLICE_ITEMS,
    SCORES_FILENAME,
    BUCKET_HIGH,
    BUCKET_LOW,
    DevSlice,
)
from llb.core.contracts.common import JsonObject
from llb.goldset.schema import GoldItem, load_goldset


def carve_dev_slice(
    item_ids: list[str] | tuple[str, ...],
    *,
    seed: int = DEFAULT_SEED,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
) -> DevSlice:
    """Split tuning item ids into disjoint train/dev sub-slices, deterministically for a seed."""
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError(f"dev_fraction must lie strictly between 0 and 1, got {dev_fraction}")
    unique = sorted({str(item_id) for item_id in item_ids})
    if len(unique) < MIN_SLICE_ITEMS:
        raise ValueError(
            f"a held-out dev slice needs at least {MIN_SLICE_ITEMS} tuning items, got {len(unique)}"
        )
    shuffled = list(unique)
    Random(seed).shuffle(shuffled)
    # Always leave at least one item on each side, whatever the fraction rounds to.
    n_dev = min(len(unique) - 1, max(1, round(len(unique) * dev_fraction)))
    return DevSlice(
        train_ids=tuple(sorted(shuffled[n_dev:])),
        dev_ids=tuple(sorted(shuffled[:n_dev])),
        seed=seed,
        dev_fraction=dev_fraction,
    )


def load_base_scores(run_dir: Path | str) -> dict[str, float]:
    """Per-item base-model `objective_score` from a scored run bundle's `scores.jsonl`."""
    path = Path(run_dir) / SCORES_FILENAME
    if not path.is_file():
        raise ValueError(f"--stratify-by-base-score run has no {SCORES_FILENAME}: {run_dir}")
    scores: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        item_id = row.get("item_id")
        value = row.get("objective_score")
        if item_id is not None and value is not None:
            scores[str(item_id)] = float(value)
    return scores


def base_score_bucket(score: float | None) -> str:
    """The base-score stratum an item falls into (see `BUCKET_PRIORITY`)."""
    if score is None:
        return BUCKET_UNSCORED
    if score <= 0.0:
        return BUCKET_ZERO
    return BUCKET_HIGH if score >= HIGH_SCORE_BOUNDARY else BUCKET_LOW


def _dev_quota_per_bucket(buckets: dict[str, list[str]], n_dev: int, total: int) -> dict[str, int]:
    """Proportional dev quota per bucket: floor of one (answerable buckets first), then
    largest-remainder top-up, each bucket capped at its own size. Deterministic."""
    quotas = {bucket: 0 for bucket in buckets}
    remaining = n_dev
    for bucket in BUCKET_PRIORITY:
        if remaining > 0 and buckets.get(bucket):
            quotas[bucket] = 1
            remaining -= 1
    while remaining > 0:
        ideal = {b: len(ids) * n_dev / total for b, ids in buckets.items()}
        open_buckets = [b for b in BUCKET_PRIORITY if b in buckets and quotas[b] < len(buckets[b])]
        if not open_buckets:
            break
        winner = max(open_buckets, key=lambda b: ideal[b] - quotas[b])
        quotas[winner] += 1
        remaining -= 1
    return quotas


def carve_stratified_dev_slice(
    item_ids: list[str] | tuple[str, ...],
    base_scores: dict[str, float],
    *,
    seed: int = DEFAULT_SEED,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    base_score_run: str | None = None,
) -> DevSlice:
    """Carve the dev slice proportionally per base-score bucket (answerable items guaranteed).

    Same guarantees as `carve_dev_slice` -- train/dev disjoint, deterministic for a seed, at
    least one item on each side -- plus: every non-empty bucket is represented (floor of one,
    answerable buckets first), so a small dev slice still carries items the base model can
    answer and the trial objective can discriminate. Refuses a population with NO answerable
    item: a study cannot rank trials against a constant objective.
    """
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError(f"dev_fraction must lie strictly between 0 and 1, got {dev_fraction}")
    unique = sorted({str(item_id) for item_id in item_ids})
    if len(unique) < MIN_SLICE_ITEMS:
        raise ValueError(
            f"a held-out dev slice needs at least {MIN_SLICE_ITEMS} tuning items, got {len(unique)}"
        )
    if not any(base_scores.get(item_id, 0.0) > 0.0 for item_id in unique):
        raise ValueError(
            "stratified dev slice refused: the base model scores 0.0 on every tuning item, so "
            "no dev slice can discriminate between trials; grow or diversify the dataset "
            "(or drop --stratify-by-base-score to accept a constant objective knowingly)"
        )
    buckets: dict[str, list[str]] = {}
    for item_id in unique:
        buckets.setdefault(base_score_bucket(base_scores.get(item_id)), []).append(item_id)

    n_dev = min(len(unique) - 1, max(1, round(len(unique) * dev_fraction)))
    quotas = _dev_quota_per_bucket(buckets, n_dev, len(unique))
    rng = Random(seed)
    dev: set[str] = set()
    for bucket in BUCKET_PRIORITY:
        ids = list(buckets.get(bucket, []))
        rng.shuffle(ids)
        dev.update(ids[: quotas.get(bucket, 0)])

    strata: JsonObject = {
        bucket: {
            "population": len(ids),
            "dev": sum(1 for item_id in ids if item_id in dev),
            "mean_base_score": round(sum(base_scores.get(i, 0.0) for i in ids) / len(ids), 6),
        }
        for bucket, ids in sorted(buckets.items())
    }
    if base_score_run is not None:
        strata = {"base_score_run": base_score_run, "buckets": strata}
    return DevSlice(
        train_ids=tuple(sorted(set(unique) - dev)),
        dev_ids=tuple(sorted(dev)),
        seed=seed,
        dev_fraction=dev_fraction,
        strata=strata,
    )


def _carve_dev_slice(
    dataset_manifest: JsonObject,
    stratify_by_base_score: Path | str | None,
    *,
    seed: int,
    dev_fraction: float,
) -> DevSlice:
    """The frozen dev slice: base-score-stratified when a base run is given, plain otherwise."""
    item_ids = [str(item_id) for item_id in dataset_manifest.get("item_ids") or []]
    if stratify_by_base_score is not None:
        return carve_stratified_dev_slice(
            item_ids,
            load_base_scores(stratify_by_base_score),
            seed=seed,
            dev_fraction=dev_fraction,
            base_score_run=str(stratify_by_base_score),
        )
    return carve_dev_slice(item_ids, seed=seed, dev_fraction=dev_fraction)


def _dev_items(goldset_path: Path | str, dev_slice: DevSlice) -> list[GoldItem]:
    wanted = set(dev_slice.dev_ids)
    items = [item for item in load_goldset(goldset_path) if item.id in wanted]
    if not items:
        raise SystemExit(
            f"[finetune-hparams] none of the {len(wanted)} dev-slice ids exist in {goldset_path}"
        )
    return items
