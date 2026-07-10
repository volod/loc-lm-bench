"""vLLM serving preflight: the flashinfer sampler preflight + verdict-gated launch_env + run-eval CLI knobs.

All driven by injected probes / config overrides -- no CUDA, no vLLM, no YAML.
"""

import typer

import pytest

from llb.backends.preflight import (
    SAMPLER_FLASHINFER,
    SAMPLER_NATIVE,
    auto_pin_flashinfer,
    flashinfer_sampler_ok,
    load_verdict,
    probe_sampler,
    run_preflight,
    save_verdict,
    verdict_is_current,
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


# --- vLLM serving preflight auto-pin a host-compatible flashinfer ---------------------------------------------


def test_auto_pin_installs_first_working_candidate():
    installed: list[str] = []

    def installer(version: str) -> bool:
        installed.append(version)
        return True

    probes = iter([False, True])  # the first candidate installs but still fails; the second works
    pinned, ok = auto_pin_flashinfer(
        ("0.9.9", "0.2.5"), probe=lambda: next(probes), installer=installer
    )
    assert ok and pinned == "0.2.5" and installed == ["0.9.9", "0.2.5"]


def test_auto_pin_gives_up_when_no_candidate_works():
    pinned, ok = auto_pin_flashinfer(("a", "b"), probe=lambda: False, installer=lambda _v: True)
    assert pinned is None and not ok


def test_probe_sampler_auto_pins_when_bundled_fails():
    probes = iter([False, True])  # bundled flashinfer fails, the pinned candidate works
    verdict = probe_sampler(
        probe=lambda: next(probes),
        candidates=("0.2.5",),
        installer=lambda _v: True,
        driver="595.71",
    )
    assert verdict["sampler"] == SAMPLER_FLASHINFER
    assert verdict["auto_pinned"] and verdict["pinned_version"] == "0.2.5"
    assert verdict["driver"] == "595.71" and "auto-pinning" in verdict["detail"]


# --- vLLM serving preflight re-run the preflight on a driver change (no full rebuild) -------------------------


def test_verdict_is_current_only_for_matching_driver():
    v = probe_sampler(probe=lambda: True, driver="500.1", candidates=())
    assert verdict_is_current(v, "500.1") is True
    assert verdict_is_current(v, "600.2") is False  # a driver change invalidates the cache
    assert verdict_is_current(None, "500.1") is False


def test_run_preflight_skips_when_verdict_current(tmp_path):
    run_preflight(probe=lambda: True, data_dir=tmp_path, driver="500", candidates=())
    # same driver -> the cached flashinfer verdict is reused even though the probe would now fail
    again = run_preflight(probe=lambda: False, data_dir=tmp_path, driver="500", candidates=())
    assert again["sampler"] == SAMPLER_FLASHINFER


def test_run_preflight_reprobes_on_driver_change(tmp_path):
    run_preflight(probe=lambda: True, data_dir=tmp_path, driver="500", candidates=())
    changed = run_preflight(probe=lambda: False, data_dir=tmp_path, driver="600", candidates=())
    assert changed["sampler"] == SAMPLER_NATIVE and changed["driver"] == "600"


def test_run_preflight_force_reprobes_same_driver(tmp_path):
    run_preflight(probe=lambda: True, data_dir=tmp_path, driver="500", candidates=())
    forced = run_preflight(
        probe=lambda: False, data_dir=tmp_path, driver="500", candidates=(), force=True
    )
    assert forced["sampler"] == SAMPLER_NATIVE


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
