"""Budgeted LoRA hyperparameter search: split discipline, determinism, budget, and consumption."""

import json
from pathlib import Path

import pytest

from llb.core.config import RunConfig
from llb.core.contracts import EvalResult, JsonObject
from llb.finetune.dataset import subset_dataset
from llb.finetune.hparam_search import (
    HPARAMS_MANIFEST,
    HPARAMS_METHOD,
    STATE_COMPLETE,
    STATE_PRUNED,
    assert_tuning_only,
    carve_dev_slice,
    latest_hparams_manifest,
    search_hyperparameters,
    trainer_defaults,
)
from llb.finetune.loop import run_self_improve
from llb.finetune.naming import model_slug
from llb.finetune.trainer import fake_train_adapter, load_adapter_manifest
from llb.goldset.schema import GoldItem, dump_goldset, load_goldset

MODEL = "base-model"
TUNING_IDS = [f"tune-{index}" for index in range(8)]


def _item(item_id: str, split: str) -> GoldItem:
    return GoldItem(
        id=item_id,
        question=f"Question {item_id}?",
        reference_answer=f"Answer {item_id}",
        source_doc_id=f"{item_id}.txt",
        source_spans=[
            {"doc_id": f"{item_id}.txt", "char_start": 0, "char_end": 5, "text": "alpha"}
        ],
        provenance="human-authored",
        verified=True,
        split=split,
    )


def _goldset(tmp_path: Path) -> Path:
    path = tmp_path / "goldset.jsonl"
    items = [_item(item_id, "tuning") for item_id in TUNING_IDS]
    items += [_item("cal-1", "calibration"), _item("final-1", "final")]
    dump_goldset(items, path)
    return path


def _dataset(tmp_path: Path, *, item_ids: list[str] | None = None, digest: str = "d0") -> Path:
    """A tuning-split export, as `export-finetune-set` would leave it."""
    ids = item_ids or TUNING_IDS
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "item_id": item_id,
            "split": "tuning",
            "weight": 1.0,
            "messages": [{"role": "user", "content": item_id}],
            "response": f"Answer {item_id}",
        }
        for item_id in ids
    ]
    (dataset_dir / "sft.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (dataset_dir / "dpo.jsonl").write_text("", encoding="utf-8")
    (dataset_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "kind": "llb.finetune.dataset",
                "dataset_digest": digest,
                "item_ids": ids,
                "split_counts": {"tuning": len(ids)},
                "n_sft": len(ids),
                "n_dpo": 0,
                "prompt_template": "eval.rag.chat",
            }
        ),
        encoding="utf-8",
    )
    return dataset_dir


def _config(tmp_path: Path, goldset: Path | None = None) -> RunConfig:
    return RunConfig(
        data_dir=tmp_path,
        model=MODEL,
        backend="vllm",
        goldset_path=goldset or (tmp_path / "missing.jsonl"),
    )


def _trainer_fn(dataset_dir: Path, model: str, adapter_dir: Path, seed: int, params: JsonObject):
    return fake_train_adapter(
        dataset_dir=dataset_dir, model=model, out_dir=adapter_dir, seed=seed, hyperparameters=params
    )


def _rank_objective(_adapter_dir: Path, params: JsonObject) -> float:
    """Deterministic synthetic objective peaking at rank 16."""
    return 1.0 - abs(int(params["lora_r"]) - 16) / 64.0


def _fake_clock(step_s: float):
    """A clock that advances `step_s` on every read, so a wall-clock budget is testable."""
    ticks = iter(index * step_s for index in range(10_000))
    return lambda: next(ticks)


def _write_hparams_manifest(tmp_path: Path, model: str, best: JsonObject) -> Path:
    out = tmp_path / HPARAMS_METHOD / model_slug(model) / "20260101-000000"
    out.mkdir(parents=True)
    manifest = out / HPARAMS_MANIFEST
    manifest.write_text(
        json.dumps({"kind": "llb.finetune.hparams", "model": model, "best_hyperparameters": best}),
        encoding="utf-8",
    )
    return manifest


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


def test_study_is_deterministic_for_a_seed(tmp_path: Path):
    dataset = _dataset(tmp_path)
    config = _config(tmp_path)

    def run(name: str):
        return search_hyperparameters(
            config,
            model=MODEL,
            dataset_dir=dataset,
            max_trials=4,
            seed=7,
            trainer="fake",
            out_dir=tmp_path / name,
            trainer_fn=_trainer_fn,
            objective_fn=_rank_objective,
        )

    first, second = run("study-a"), run("study-b")

    assert [trial.hyperparameters for trial in first.trials] == [
        trial.hyperparameters for trial in second.trials
    ]
    assert first.best_hyperparameters == second.best_hyperparameters
    assert first.best_objective == second.best_objective


def test_budget_abort_stops_between_trials_and_the_study_resumes(tmp_path: Path):
    dataset = _dataset(tmp_path)
    config = _config(tmp_path)
    study_dir = tmp_path / "study"
    trained: list[int] = []

    def counting_trainer(dataset_dir: Path, model: str, adapter_dir: Path, seed: int, params):
        trained.append(int(params["lora_r"]))
        return _trainer_fn(dataset_dir, model, adapter_dir, seed, params)

    # Every clock read advances 10 minutes, so a 0.25h budget is spent after the first trial.
    aborted = search_hyperparameters(
        config,
        model=MODEL,
        dataset_dir=dataset,
        max_trials=6,
        max_hours=0.25,
        trainer="fake",
        out_dir=study_dir,
        trainer_fn=counting_trainer,
        objective_fn=_rank_objective,
        clock=_fake_clock(600.0),
    )

    assert aborted.budget_exhausted
    assert len(aborted.trials) == 1, "the in-flight trial completes; the next never starts"
    assert json.loads(aborted.manifest_path.read_text(encoding="utf-8"))["budget_exhausted"] is True
    before = len(trained)

    resumed = search_hyperparameters(
        config,
        model=MODEL,
        dataset_dir=dataset,
        max_trials=6,
        trainer="fake",
        resume=study_dir,
        trainer_fn=counting_trainer,
        objective_fn=_rank_objective,
    )

    assert not resumed.budget_exhausted
    assert len(resumed.trials) == 6
    assert [trial.number for trial in resumed.trials] == [0, 1, 2, 3, 4, 5]
    assert len(trained) - before == 5, "a resume retrains only the unfinished trials"


def test_resume_at_the_same_budget_runs_no_further_trial(tmp_path: Path):
    dataset = _dataset(tmp_path)
    config = _config(tmp_path)
    study_dir = tmp_path / "study"
    scored: list[JsonObject] = []

    def counting_objective(adapter_dir: Path, params: JsonObject) -> float:
        scored.append(params)
        return _rank_objective(adapter_dir, params)

    search_hyperparameters(
        config,
        model=MODEL,
        dataset_dir=dataset,
        max_trials=3,
        trainer="fake",
        out_dir=study_dir,
        trainer_fn=_trainer_fn,
        objective_fn=counting_objective,
    )
    assert len(scored) == 3

    result = search_hyperparameters(
        config,
        model=MODEL,
        dataset_dir=dataset,
        max_trials=3,
        trainer="fake",
        resume=study_dir,
        trainer_fn=_trainer_fn,
        objective_fn=counting_objective,
    )

    assert len(scored) == 3, "an exhausted trial budget starts no further trial on resume"
    assert len(result.trials) == 3


def test_a_measured_oom_prunes_the_trial_instead_of_killing_the_study(tmp_path: Path):
    dataset = _dataset(tmp_path)

    def flaky_objective(adapter_dir: Path, params: JsonObject) -> float:
        if int(params["lora_r"]) >= 64:
            raise RuntimeError("CUDA error: out of memory")
        return _rank_objective(adapter_dir, params)

    result = search_hyperparameters(
        _config(tmp_path),
        model=MODEL,
        dataset_dir=dataset,
        max_trials=4,
        trainer="fake",
        out_dir=tmp_path / "study",
        trainer_fn=_trainer_fn,
        objective_fn=flaky_objective,
    )

    states = [trial.state for trial in result.trials]
    assert STATE_PRUNED in states, "the rank-64 trial must prune on the measured OOM"
    assert states.count(STATE_COMPLETE) >= 1
    assert result.best_hyperparameters is not None
    assert int(result.best_hyperparameters["lora_r"]) < 64


def test_a_failed_trial_still_leaves_an_inspectable_manifest(tmp_path: Path):
    """A trial that fails for an unprunable reason has still cost a fine-tune; do not lose the study."""
    dataset = _dataset(tmp_path)
    study_dir = tmp_path / "study"

    def exploding_objective(_adapter_dir: Path, _params: JsonObject) -> float:
        raise RuntimeError("the backend refused this adapter")

    with pytest.raises(RuntimeError, match="refused this adapter"):
        search_hyperparameters(
            _config(tmp_path),
            model=MODEL,
            dataset_dir=dataset,
            max_trials=3,
            trainer="fake",
            out_dir=study_dir,
            trainer_fn=_trainer_fn,
            objective_fn=exploding_objective,
        )

    manifest = json.loads((study_dir / HPARAMS_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["n_trials"] == 1
    assert manifest["trials"][0]["state"] == "failed"
    assert manifest["best_hyperparameters"] is None
    assert (study_dir / "study.db").is_file(), "the study stays resumable"


def test_subset_dataset_recomputes_the_digest(tmp_path: Path):
    dataset = _dataset(tmp_path, digest="parent-digest")

    manifest = subset_dataset(
        dataset_dir=dataset,
        out_dir=tmp_path / "subset",
        item_ids=TUNING_IDS[:3],
        role="train",
    )

    assert manifest["item_ids"] == TUNING_IDS[:3]
    assert manifest["split_counts"] == {"tuning": 3}
    assert manifest["parent_dataset_digest"] == "parent-digest"
    assert manifest["dataset_digest"] != "parent-digest", "a subset is not its parent"
    assert manifest["subset_role"] == "train"


def test_trainer_consumes_the_recorded_best_config(tmp_path: Path):
    best = {"method": "lora", "lora_r": 32, "lora_alpha": 64, "learning_rate": 0.0001}
    manifest_path = _write_hparams_manifest(tmp_path, MODEL, best)
    dataset = _dataset(tmp_path)

    assert latest_hparams_manifest(tmp_path, MODEL) == manifest_path
    defaults = trainer_defaults(tmp_path, MODEL)
    assert defaults["hyperparameters"] == best
    assert defaults["hparams_manifest"] == str(manifest_path)

    fake_train_adapter(dataset_dir=dataset, model=MODEL, out_dir=tmp_path / "adapter", **defaults)

    adapter = load_adapter_manifest(tmp_path / "adapter")
    assert adapter["hyperparameters"]["lora_r"] == 32
    assert adapter["hparams_manifest"] == str(manifest_path)


def test_trainer_defaults_are_empty_without_a_recorded_search(tmp_path: Path):
    assert trainer_defaults(tmp_path, MODEL) == {}
    _write_hparams_manifest(tmp_path, MODEL, {})
    assert trainer_defaults(tmp_path, MODEL) == {}, "a study with no best config sets no default"


def test_self_improve_round_records_the_searched_config_in_provenance(tmp_path: Path):
    best = {"method": "lora", "lora_r": 8, "lora_alpha": 32, "num_train_epochs": 2.0}
    manifest_path = _write_hparams_manifest(tmp_path, MODEL, best)
    goldset = tmp_path / "goldset.jsonl"
    dump_goldset([_item("tune-1", "tuning"), _item("final-1", "final")], goldset)
    config = RunConfig(data_dir=tmp_path, goldset_path=goldset, model=MODEL, backend="vllm")

    def eval_fn(cfg: RunConfig, split: str, _round_dir: Path) -> EvalResult:
        items = [item for item in load_goldset(goldset) if item.split == split]
        objective = 0.8 if cfg.adapter_path is not None and split == "final" else 0.2
        run = _write_bundle(tmp_path, f"{split}-{objective}", split, items, objective)
        return {
            "rows": [],
            "metrics": {"objective_score": objective, "reliability": 1.0, "tokens_per_s": 1.0},
            "retrieval": {"n": len(items), "k": 1, "recall_at_k": 1.0, "mrr": 1.0},
            "paths": {"manifest": str(run / "manifest.json"), "scores": str(run / "scores.jsonl")},
            "table": "",
            "telemetry": None,
            "manifest": None,
            "run_timestamp": run.name,
        }

    result = run_self_improve(
        config, rounds=1, out_dir=tmp_path / "campaign", trainer="fake", eval_fn=eval_fn
    )

    adapter = load_adapter_manifest(result.rounds[0].adapter_dir)
    assert adapter["hyperparameters"]["lora_r"] == 8
    assert adapter["hyperparameters"]["num_train_epochs"] == 2.0
    assert adapter["hparams_manifest"] == str(manifest_path)


def _write_bundle(
    root: Path, name: str, split: str, items: list[GoldItem], objective: float
) -> Path:
    run = root / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "split": split,
                "config": {"model": MODEL, "backend": "vllm", "goldset_path": str(root)},
                "metrics": {"objective_score": objective, "reliability": 1.0, "tokens_per_s": 1.0},
                "n_cases": len(items),
            }
        ),
        encoding="utf-8",
    )
    (run / "scores.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "item_id": item.id,
                    "split": item.split,
                    "status": "ok",
                    "objective_score": objective,
                    "answer_preview": "wrong answer",
                }
            )
            + "\n"
            for item in items
        ),
        encoding="utf-8",
    )
    (run / "retrieval.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "item_id": item.id,
                    "retrieved": [
                        {
                            "doc_id": item.source_doc_id,
                            "char_start": 0,
                            "char_end": 5,
                            "rank": 1,
                            "text_preview": "alpha",
                        }
                    ],
                    "gold_spans": [span.model_dump() for span in item.source_spans],
                }
            )
            + "\n"
            for item in items
        ),
        encoding="utf-8",
    )
    return run
