"""Serving knobs a joint search carries: config file, flag, candidate cells, manifest, ceiling."""

import json
from pathlib import Path

import pytest

from llb.cli.models.search_serving import search_base_config, search_max_model_len
from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec, ResolvedModel
from llb.optimize.joint_search.constants import DEFAULT_SEARCH_MAX_MODEL_LEN, MANIFEST_FILE
from llb.optimize.joint_search.hooks import candidate_config
from llb.optimize.joint_search.models import FinalistTuneResult
from llb.optimize.joint_search.schedule import run_joint_search
from llb.optimize.objectives import TrialMetrics
from llb.optimize.tuning_space import (
    GPU_MEMORY_UTILIZATION_RANGE,
    suggest_gpu_memory_utilization,
    suggest_overrides,
)
from llb.optimize.tuning_space import FINAL_SPLIT

# The validated value for a 16 GiB card; the RunConfig default (0.85) is above it.
HOST_UTILIZATION = 0.80


def _write_config(path: Path, **fields: object) -> Path:
    lines = [f"{key}: {value}" for key, value in fields.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class _CeilingTrial:
    """Records the float ranges it was asked for, and returns the top of each."""

    def __init__(self, values: dict[str, object]):
        self.values = values
        self.float_ranges: dict[str, tuple[float, float]] = {}

    def suggest_categorical(self, name, choices):
        return self.values[name]

    def suggest_int(self, name, lo, hi, step=1):
        return self.values[name]

    def suggest_float(self, name, lo, hi, step=None):
        self.float_ranges[name] = (lo, hi)
        return hi

    def set_user_attr(self, key, value):
        self.values[key] = value


BASE_TRIAL_VALUES: dict[str, object] = {
    "strategy": "recursive",
    "chunk_size": 800,
    "overlap_frac": 0.15,
    "retrieval_mode": "flat",
    "top_k": 5,
    "max_model_len": 8192,
}


def test_search_max_model_len_prefers_the_flag_then_the_config_then_the_default(tmp_path: Path):
    from_file = RunConfig(data_dir=tmp_path, max_model_len=4096)
    assert search_max_model_len(from_file, 16384) == 16384  # flag wins
    assert search_max_model_len(from_file, None) == 4096  # then the config file
    bare = RunConfig(data_dir=tmp_path)
    assert search_max_model_len(bare, None) == DEFAULT_SEARCH_MAX_MODEL_LEN


def test_a_config_file_supplies_the_serving_knobs_and_the_flag_overrides_it(tmp_path: Path):
    path = _write_config(
        tmp_path / "run.yaml", backend="vllm", gpu_memory_utilization=HOST_UTILIZATION
    )
    from_file = search_base_config(path, gpu_memory_utilization=None)
    assert from_file.gpu_memory_utilization == HOST_UTILIZATION
    overridden = search_base_config(path, gpu_memory_utilization=0.70)
    assert overridden.gpu_memory_utilization == 0.70
    # No file and no flag is still the RunConfig default, not an error.
    assert search_base_config(None, gpu_memory_utilization=None).gpu_memory_utilization == 0.85


def test_a_vllm_candidate_cell_inherits_the_declared_utilization(tmp_path: Path):
    base = RunConfig(data_dir=tmp_path, gpu_memory_utilization=HOST_UTILIZATION)
    resolution = ResolvedModel(
        name="alpha",
        chosen_backend="vllm",
        chosen_source="org/alpha",
        verdict="gpu",
        candidates=[],
        note="ok",
    )
    cell = candidate_config(base, resolution, max_model_len=4096, run_name="joint-screen-alpha")
    assert cell.gpu_memory_utilization == HOST_UTILIZATION and cell.max_model_len == 4096


def test_every_screen_cell_is_served_at_the_declared_utilization_and_the_manifest_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The whole point: a vLLM candidate is screened at 0.80, not at the RunConfig default."""
    specs: list[ModelSpec] = [
        {"name": "alpha", "backend": "vllm", "source": "org/alpha"},
        {"name": "bravo", "backend": "vllm", "source": "org/bravo"},
    ]
    served: list[tuple[float, int | None]] = []

    def fake_resolve_all(candidates, vram_mib, ram_mib, *, probes=None, **kwargs):
        del vram_mib, ram_mib, probes, kwargs
        return [
            ResolvedModel(
                name=c["name"],
                chosen_backend="vllm",
                chosen_source=c["source"],
                verdict="gpu",
                candidates=[],
                note="ok",
            )
            for c in candidates
        ]

    monkeypatch.setattr("llb.backends.resolver.resolve_all", fake_resolve_all)
    monkeypatch.setattr(
        "llb.backends.readiness.local_backend_ready", lambda backend, data_dir: (True, "")
    )

    def screen_evaluate(config: RunConfig, limit: int | None) -> TrialMetrics:
        del limit
        served.append((config.gpu_memory_utilization, config.max_model_len))
        return TrialMetrics(quality=0.5, latency_s=1.0)

    def tune_finalist(base: RunConfig, resolution: ResolvedModel, cell_dir: Path):
        del cell_dir
        name = resolution["name"]
        return FinalistTuneResult(
            name=name,
            backend="vllm",
            source=resolution["chosen_source"] or name,
            study_name=f"j-{name}",
            overrides_by_pick={"best_quality": {}},
            finals={
                "best_quality": {
                    "rows": [{"model": name, "quality": 0.5}],
                    "metrics": {"objective_score": 0.5},
                    "manifest": {"split": FINAL_SPLIT},
                    "table": "ok",
                    "retrieval": {},
                    "paths": {},
                    "telemetry": None,
                    "run_timestamp": "t",
                }
            },
        )

    result = run_joint_search(
        RunConfig(data_dir=tmp_path, gpu_memory_utilization=HOST_UTILIZATION),
        specs,
        n_trials=1,
        run_id="serving",
        screen_limit=2,
        min_finalists=2,
        screen_evaluate=screen_evaluate,
        tune_finalist=tune_finalist,
        isolate=False,
        max_model_len=4096,
    )
    assert served == [(HOST_UTILIZATION, 4096), (HOST_UTILIZATION, 4096)]
    manifest = json.loads((result.run_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["serving"] == {
        "gpu_memory_utilization": HOST_UTILIZATION,
        "max_model_len": 4096,
    }


def test_a_sampled_trial_never_launches_above_the_declared_ceiling():
    trial = _CeilingTrial(dict(BASE_TRIAL_VALUES))
    overrides = suggest_overrides(
        trial, backend="vllm", gpu_memory_utilization_ceiling=HOST_UTILIZATION
    )
    low, _high = GPU_MEMORY_UTILIZATION_RANGE
    assert trial.float_ranges["gpu_memory_utilization"] == (low, HOST_UTILIZATION)
    assert overrides["gpu_memory_utilization"] <= HOST_UTILIZATION


def test_a_ceiling_at_or_below_the_range_floor_pins_the_utilization_instead_of_sampling():
    """Nothing left to search is not a reason to sample outside what the host was validated for."""
    trial = _CeilingTrial(dict(BASE_TRIAL_VALUES))
    assert suggest_gpu_memory_utilization(trial, 0.55) == 0.55
    assert "gpu_memory_utilization" not in trial.float_ranges


def test_no_ceiling_keeps_the_full_serving_range():
    trial = _CeilingTrial(dict(BASE_TRIAL_VALUES))
    suggest_gpu_memory_utilization(trial, None)
    assert trial.float_ranges["gpu_memory_utilization"] == GPU_MEMORY_UTILIZATION_RANGE
