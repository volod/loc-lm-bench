"""Tests for finetune hparam resources."""

import json
from pathlib import Path
import pytest
from llb.core.config import RunConfig
from llb.finetune.hparam_search.model import (
    STATE_COMPLETE,
    STATE_PRUNED,
)
from llb.finetune.hparam_search.search import search_hyperparameters
from llb.finetune.hparam_search.space import assert_tuning_only
from llb.goldset.schema import load_goldset
from finetune_hparam_helpers import (
    FIXTURE_ARCH,
    MODEL,
    TUNING_IDS,
    _config,
    _dataset,
    _goldset,
    _rank_objective,
    _trainer_fn,
)


def test_adapter_footprint_estimate_is_hand_computable():
    from llb.finetune.hparam_search.space import adapter_param_estimate, estimated_adapter_train_mib

    params = {"lora_r": 8, "target_modules": ["q_proj", "v_proj"]}
    # 32 layers x 2 modules x 2 matrices x 4096 hidden x rank 8 = 4,194,304 params
    assert adapter_param_estimate(params, hidden_size=4096, n_layers=32) == 4_194_304
    # x 16 bytes / MiB = 64 MiB
    assert estimated_adapter_train_mib(params, hidden_size=4096, n_layers=32) == 64.0


@pytest.mark.slow
def test_infeasible_point_prunes_before_the_trainer_runs(tmp_path: Path):
    """A rank-64 x attn_mlp point on a no-headroom host never reaches the trainer."""
    dataset = _dataset(tmp_path)
    trained_ranks: list[int] = []

    def counting_trainer(dataset_dir, model, adapter_dir, seed, params):
        trained_ranks.append(int(params["lora_r"]))
        return _trainer_fn(dataset_dir, model, adapter_dir, seed, params)

    # Headroom fits rank 8 on the widest preset (~1.8 GiB for rank 64) but not rank >= 16.
    headroom = 300.0
    result = search_hyperparameters(
        _config(tmp_path),
        model=MODEL,
        dataset_dir=dataset,
        max_trials=10,
        seed=7,
        trainer="fake",
        out_dir=tmp_path / "study",
        vram_headroom_mib=headroom,
        model_arch=FIXTURE_ARCH,
        trainer_fn=counting_trainer,
        objective_fn=_rank_objective,
    )
    pruned = [t for t in result.trials if t.state == STATE_PRUNED]
    complete = [t for t in result.trials if t.state == STATE_COMPLETE]
    assert pruned, "the fixture space must contain an infeasible point"
    assert complete, "a small-rank point must still train"
    assert len(trained_ranks) == len(complete)  # the trainer never saw a pruned point
    for trial in result.trials:
        assert trial.estimated_adapter_mib is not None
        if trial.state == STATE_PRUNED:
            assert trial.estimated_adapter_mib > headroom
        if trial.state == STATE_COMPLETE:
            assert trial.estimated_adapter_mib <= headroom
    # The manifest carries the estimate on every trial row, pruned ones included.
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert all(row["estimated_adapter_mib"] is not None for row in manifest["trials"])


@pytest.mark.slow
def test_prune_is_off_without_a_headroom(tmp_path: Path):
    dataset = _dataset(tmp_path)
    result = search_hyperparameters(
        _config(tmp_path),
        model=MODEL,
        dataset_dir=dataset,
        max_trials=4,
        seed=7,
        trainer="fake",
        out_dir=tmp_path / "study",
        model_arch=FIXTURE_ARCH,
        trainer_fn=_trainer_fn,
        objective_fn=_rank_objective,
    )
    assert all(t.state == STATE_COMPLETE for t in result.trials)
    assert all(t.estimated_adapter_mib is not None for t in result.trials)


def test_guard_refuses_a_dataset_manifest_declaring_a_protected_split():
    with pytest.raises(SystemExit, match="non-tuning splits: final"):
        assert_tuning_only({"split_counts": {"tuning": 3, "final": 1}, "item_ids": []})


def test_guard_refuses_protected_item_ids_even_when_the_manifest_looks_clean(tmp_path: Path):
    """A dataset manifest is operator-writable, so its split counts alone are not proof."""
    goldset = _goldset(tmp_path)
    laundered = {"split_counts": {"tuning": 1}, "item_ids": ["final-1"]}

    with pytest.raises(SystemExit, match="protected-split item ids: final-1"):
        assert_tuning_only(laundered, goldset_path=goldset)


def test_no_calibration_or_final_id_can_enter_a_trial(tmp_path: Path):
    goldset = _goldset(tmp_path)
    dataset = _dataset(tmp_path)
    trained_on: list[set[str]] = []

    def spy_trainer(dataset_dir: Path, model: str, adapter_dir: Path, seed: int, params):
        manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
        trained_on.append({str(item_id) for item_id in manifest["item_ids"]})
        return _trainer_fn(dataset_dir, model, adapter_dir, seed, params)

    result = search_hyperparameters(
        _config(tmp_path, goldset),
        model=MODEL,
        dataset_dir=dataset,
        max_trials=3,
        trainer="fake",
        out_dir=tmp_path / "study",
        goldset_path=goldset,
        trainer_fn=spy_trainer,
        objective_fn=_rank_objective,
    )

    protected = {item.id for item in load_goldset(goldset) if item.split != "tuning"}
    assert protected == {"cal-1", "final-1"}
    assert trained_on, "the study must have trained at least one trial"
    dev = set(result.dev_slice.dev_ids)
    for seen in trained_on:
        assert seen & protected == set(), "a trial trained on a protected-split item"
        assert seen & dev == set(), "a trial trained on its own held-out dev slice"
    assert set(result.dev_slice.train_ids) | dev <= set(TUNING_IDS)


def test_a_backend_that_cannot_serve_a_lora_refuses_before_any_trial_trains(tmp_path: Path):
    """The first trial fine-tunes before it scores, so this must be caught up front, not after."""
    goldset = _goldset(tmp_path)
    dataset = _dataset(tmp_path)
    config = RunConfig(data_dir=tmp_path, model=MODEL, backend="ollama", goldset_path=goldset)
    trained: list[str] = []

    with pytest.raises(SystemExit, match="needs the vllm backend"):
        search_hyperparameters(
            config,
            model=MODEL,
            dataset_dir=dataset,
            max_trials=2,
            trainer="fake",
            out_dir=tmp_path / "study",
            goldset_path=goldset,
            trainer_fn=lambda *args: trained.append("trained") or {},
        )

    assert trained == []
    assert not (tmp_path / "study" / "study.db").exists(), "no study is created on a bad backend"


def test_the_default_objective_needs_a_goldset(tmp_path: Path):
    dataset = _dataset(tmp_path)

    with pytest.raises(SystemExit, match="needs a goldset"):
        search_hyperparameters(
            _config(tmp_path),
            model=MODEL,
            dataset_dir=dataset,
            max_trials=1,
            trainer="fake",
            out_dir=tmp_path / "study",
        )
