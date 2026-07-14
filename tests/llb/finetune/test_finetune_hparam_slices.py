"""Tests for finetune hparam slices."""

import json
from pathlib import Path
import pytest
from llb.finetune.hparam_search.dev_slice import (
    carve_dev_slice,
    carve_stratified_dev_slice,
    load_base_scores,
)
from llb.finetune.hparam_search.search import search_hyperparameters
from finetune_hparam_helpers import (
    BASE_SCORE_RUN,
    MODEL,
    STRATIFY_IDS,
    TUNING_IDS,
    _config,
    _dataset,
    _rank_objective,
    _trainer_fn,
)


def test_dev_slice_is_disjoint_and_deterministic():
    first = carve_dev_slice(TUNING_IDS, seed=13, dev_fraction=0.25)
    again = carve_dev_slice(TUNING_IDS, seed=13, dev_fraction=0.25)
    other = carve_dev_slice(TUNING_IDS, seed=99, dev_fraction=0.25)

    assert set(first.train_ids) & set(first.dev_ids) == set()
    assert set(first.train_ids) | set(first.dev_ids) == set(TUNING_IDS)
    assert len(first.dev_ids) == 2
    assert (first.train_ids, first.dev_ids) == (again.train_ids, again.dev_ids)
    assert first.dev_ids != other.dev_ids, "a different seed must draw a different slice"


def test_dev_slice_always_leaves_an_item_on_each_side():
    tiny = carve_dev_slice(["a", "b"], dev_fraction=0.9)

    assert len(tiny.train_ids) == 1 and len(tiny.dev_ids) == 1

    with pytest.raises(ValueError, match="at least 2 tuning items"):
        carve_dev_slice(["only-one"])


def test_stratified_slice_holds_an_answerable_item_at_every_seed():
    scores = load_base_scores(BASE_SCORE_RUN)
    assert sum(1 for value in scores.values() if value > 0) == 3
    for seed in range(25):
        dev_slice = carve_stratified_dev_slice(STRATIFY_IDS, scores, seed=seed, dev_fraction=0.25)
        assert set(dev_slice.train_ids) & set(dev_slice.dev_ids) == set()
        assert set(dev_slice.train_ids) | set(dev_slice.dev_ids) == set(STRATIFY_IDS)
        assert any(scores.get(item_id, 0.0) > 0 for item_id in dev_slice.dev_ids), (
            f"seed {seed}: dev slice holds no answerable item"
        )
        again = carve_stratified_dev_slice(STRATIFY_IDS, scores, seed=seed, dev_fraction=0.25)
        assert (dev_slice.train_ids, dev_slice.dev_ids) == (again.train_ids, again.dev_ids)


def test_stratified_slice_beats_uniform_on_answerable_coverage():
    """The uniform slice can land on zero answerable items; the stratified one never does."""
    scores = load_base_scores(BASE_SCORE_RUN)

    def answerable_in_dev(dev_ids: tuple[str, ...]) -> int:
        return sum(1 for item_id in dev_ids if scores.get(item_id, 0.0) > 0)

    uniform_misses = sum(
        1
        for seed in range(25)
        if answerable_in_dev(carve_dev_slice(STRATIFY_IDS, seed=seed, dev_fraction=0.25).dev_ids)
        == 0
    )
    stratified_misses = sum(
        1
        for seed in range(25)
        if answerable_in_dev(
            carve_stratified_dev_slice(STRATIFY_IDS, scores, seed=seed, dev_fraction=0.25).dev_ids
        )
        == 0
    )
    assert uniform_misses > 0, "the fixture must exhibit the uniform-slice failure"
    assert stratified_misses == 0


def test_stratified_slice_refuses_an_all_zero_population():
    with pytest.raises(ValueError, match="constant objective|discriminate"):
        carve_stratified_dev_slice(STRATIFY_IDS, {}, seed=13, dev_fraction=0.25)


def test_load_base_scores_requires_a_scored_run():
    with pytest.raises(ValueError, match="scores.jsonl"):
        load_base_scores(Path("samples/finetune"))  # a dir without scores.jsonl


def test_manifest_records_dev_slice_strata(tmp_path: Path):
    dataset = _dataset(tmp_path, item_ids=STRATIFY_IDS)
    result = search_hyperparameters(
        _config(tmp_path),
        model=MODEL,
        dataset_dir=dataset,
        max_trials=1,
        seed=7,
        trainer="fake",
        out_dir=tmp_path / "study",
        stratify_by_base_score=BASE_SCORE_RUN,
        trainer_fn=_trainer_fn,
        objective_fn=_rank_objective,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    strata = manifest["dev_slice"]["strata"]
    assert strata["base_score_run"] == str(BASE_SCORE_RUN)
    buckets = strata["buckets"]
    assert buckets["zero"]["population"] == 9
    assert buckets["high"]["population"] == 2 and buckets["low"]["population"] == 1
    assert buckets["high"]["dev"] >= 1  # the answerable floor landed
    assert buckets["high"]["mean_base_score"] == pytest.approx(0.85)
