"""M4.3: the flashinfer sampler preflight + verdict-gated launch_env + run-eval CLI knobs.

All driven by injected probes / config overrides -- no CUDA, no vLLM, no YAML.
"""

import typer

import pytest

from llb.backends.preflight import (
    SAMPLER_FLASHINFER,
    SAMPLER_NATIVE,
    flashinfer_sampler_ok,
    load_verdict,
    probe_sampler,
    run_preflight,
    save_verdict,
    verdict_path,
)


def test_probe_flashinfer_ok_yields_flashinfer_verdict():
    verdict = probe_sampler(probe=lambda: True)
    assert verdict["sampler"] == SAMPLER_FLASHINFER
    assert "built and ran" in verdict["detail"] and verdict["checked_at"]


def test_probe_flashinfer_unavailable_yields_native_verdict():
    verdict = probe_sampler(probe=lambda: False)
    assert verdict["sampler"] == SAMPLER_NATIVE


def test_probe_swallows_a_broken_build_into_native_verdict():
    def boom() -> bool:
        raise RuntimeError("cub::BlockAdjacentDifference::FlagHeads not found")

    verdict = probe_sampler(probe=boom)
    assert verdict["sampler"] == SAMPLER_NATIVE
    assert "FlagHeads" in verdict["detail"]  # the failure is captured for diagnosis


def test_run_preflight_persists_and_round_trips(tmp_path):
    verdict = run_preflight(probe=lambda: True, data_dir=tmp_path)
    assert verdict_path(tmp_path).exists()
    assert load_verdict(tmp_path) == verdict
    assert flashinfer_sampler_ok(tmp_path) is True


def test_flashinfer_sampler_ok_false_for_native_and_missing(tmp_path):
    assert flashinfer_sampler_ok(tmp_path) is False  # no verdict written yet
    save_verdict(probe_sampler(probe=lambda: False), tmp_path)
    assert flashinfer_sampler_ok(tmp_path) is False  # native verdict


def test_load_verdict_tolerates_corrupt_file(tmp_path):
    path = verdict_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert load_verdict(tmp_path) is None


# --- run-eval serving knobs settable without YAML (Task 1) ---------------------------------


def test_run_eval_knobs_apply_without_yaml():
    from llb.cli.helpers import load_config as _load_config

    cfg = _load_config(
        None, model="org/m", backend="vllm", max_model_len=4096, gpu_memory_utilization=0.7
    )
    assert cfg.max_model_len == 4096 and cfg.gpu_memory_utilization == 0.7


def test_run_eval_omitted_knobs_keep_config_defaults():
    from llb.cli.helpers import load_config as _load_config

    cfg = _load_config(None, max_model_len=None, gpu_memory_utilization=None)
    assert cfg.max_model_len is None and cfg.gpu_memory_utilization == 0.85


def test_run_eval_knob_is_revalidated_by_runconfig():
    from llb.cli.helpers import load_config as _load_config

    with pytest.raises(typer.Exit):  # > 1.0 is out of range -> RunConfig rejects -> Exit(2)
        _load_config(None, gpu_memory_utilization=1.5)
