"""Hard-isolation sweep executor (M3.3), driven entirely by fakes (no GPU / subprocess)."""

import pytest

from llb.config import RunConfig
from llb.contracts import GpuSample
from llb.executor.isolation import (
    cell_key,
    cool_down,
    isolate_cell,
    parse_smi_samples,
    run_sweep,
    sample_gpu,
)
from llb.executor.vram import VERDICT_BASELINE_SHIFT, VERDICT_RECLAIMED, VramNotReclaimed


def cfg(tmp_path, model="a:1", backend="ollama", run_name="r"):
    return RunConfig(model=model, backend=backend, run_name=run_name, data_dir=tmp_path)


def gpu(temp=40) -> list[GpuSample]:
    return [
        {"index": 0, "temp_c": temp, "power_w": 120.0, "sm_clock_mhz": 2100, "mem_clock_mhz": 9000}
    ]


def test_cell_key_stable_ignores_run_name(tmp_path):
    a = cfg(tmp_path, run_name="x")
    b = cfg(tmp_path, run_name="y")
    assert cell_key(a) == cell_key(b)  # relabel resumes
    assert cell_key(cfg(tmp_path, model="b:2")) != cell_key(a)  # different model -> new cell


def test_parse_smi_samples():
    rows = parse_smi_samples("0, 61, 130.5, 2100, 9000\n1, [N/A], x, 1500, 5001")
    assert rows[0] == {
        "index": 0,
        "temp_c": 61,
        "power_w": 130.5,
        "sm_clock_mhz": 2100,
        "mem_clock_mhz": 9000,
    }
    assert rows[1]["temp_c"] is None and rows[1]["power_w"] is None  # [N/A] tolerated


def test_cool_down_returns_when_below_threshold():
    report = cool_down(threshold_c=55, sampler=lambda: gpu(40), sleep=lambda _s: None)
    assert report["capped"] is False and report["final_temp_c"] == 40


def test_cool_down_caps_when_stays_hot():
    ticks = iter([0.0, 1.0, 2.0, 999.0])
    report = cool_down(
        threshold_c=55,
        max_wait_s=10.0,
        sampler=lambda: gpu(80),  # never cools
        sleep=lambda _s: None,
        clock=lambda: next(ticks),
    )
    assert report["capped"] is True and report["final_temp_c"] == 80


def test_sample_gpu_no_driver(monkeypatch):
    import llb.executor.isolation as iso

    monkeypatch.setattr(
        iso.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError())
    )
    assert sample_gpu() == []


def test_run_sweep_runs_each_cell_and_writes_markers(tmp_path):
    calls = []

    def runner(config, split):
        calls.append((config.model, split))
        return str(tmp_path / "run-eval" / config.model)

    report = run_sweep(
        [cfg(tmp_path, model="a:1"), cfg(tmp_path, model="b:2")],
        sweep_id="s1",
        split="final",
        data_dir=tmp_path,
        cell_runner=runner,
        vram_reader=lambda: 1000,  # constant -> always reclaimed
        gpu_sampler=lambda: gpu(40),
        sleep=lambda _s: None,
    )
    assert report["completed"] == 2 and report["failed"] == 0
    assert calls == [("a:1", "final"), ("b:2", "final")]
    markers = list((tmp_path / "sweep" / "s1" / "cells").glob("*.json"))
    assert len(markers) == 2


def test_run_sweep_persists_thermal_flag_into_run_bundle(tmp_path):
    import json

    def runner(config, split):
        bundle = tmp_path / "run-eval" / config.model.replace(":", "_")
        bundle.mkdir(parents=True)
        (bundle / "manifest.json").write_text("{}", encoding="utf-8")
        return str(bundle)

    run_sweep(
        [cfg(tmp_path, model="a:1")],
        sweep_id="th",
        data_dir=tmp_path,
        cell_runner=runner,
        vram_reader=lambda: 1000,
        gpu_sampler=lambda: gpu(80),  # stays hot
        cooldown_temp_c=55,
        cooldown_max_s=0.0,  # cap immediately -> elevated
        sleep=lambda _s: None,
    )
    thermal = json.loads((tmp_path / "run-eval" / "a_1" / "thermal.json").read_text())
    assert thermal["cooldown_capped"] is True and thermal["final_temp_c"] == 80


def test_run_sweep_resumes_completed_cells(tmp_path):
    runner_calls = {"n": 0}

    def runner(config, split):
        runner_calls["n"] += 1
        return "dir"

    args = dict(
        sweep_id="s2",
        data_dir=tmp_path,
        cell_runner=runner,
        vram_reader=lambda: 0,
        gpu_sampler=lambda: gpu(40),
        sleep=lambda _s: None,
    )
    configs = [cfg(tmp_path, model="a:1")]
    run_sweep(configs, **args)
    second = run_sweep(configs, **args)
    assert runner_calls["n"] == 1  # not re-run
    assert second["skipped"] == 1 and second["completed"] == 0


def test_run_sweep_reruns_cell_with_truncated_marker(tmp_path):
    config = cfg(tmp_path, model="a:1")
    marker_dir = tmp_path / "sweep" / "s2-corrupt" / "cells"
    marker_dir.mkdir(parents=True)
    (marker_dir / f"{cell_key(config)}.json").write_text("{", encoding="utf-8")
    calls = {"n": 0}

    def runner(_config, _split):
        calls["n"] += 1
        return "dir"

    report = run_sweep(
        [config],
        sweep_id="s2-corrupt",
        data_dir=tmp_path,
        cell_runner=runner,
        gpu_sampler=lambda: gpu(40),
        sleep=lambda _s: None,
    )
    assert calls["n"] == 1 and report["completed"] == 1


def test_run_sweep_empty_input_has_no_artifact_side_effect(tmp_path):
    report = run_sweep([], sweep_id="empty", data_dir=tmp_path)
    assert report["n_cells"] == 0
    assert not (tmp_path / "sweep").exists()


def test_run_sweep_aborts_on_vram_not_reclaimed_for_owning_backend(tmp_path):
    reads = iter([1000] + [9000] * 100)  # baseline low, then never returns

    with pytest.raises(VramNotReclaimed):
        run_sweep(
            [cfg(tmp_path, backend="vllm")],  # vLLM owns its VRAM -> gated
            sweep_id="s3",
            data_dir=tmp_path,
            cell_runner=lambda c, s: "dir",
            vram_reader=lambda: next(reads),
            gpu_sampler=lambda: gpu(40),
            sleep=lambda _s: None,
        )


def test_run_sweep_does_not_gate_ollama_keepalive(tmp_path):
    # Ollama keeps weights warm by design, so a high residual must NOT abort the sweep.
    report = run_sweep(
        [cfg(tmp_path, backend="ollama")],
        sweep_id="s3b",
        data_dir=tmp_path,
        cell_runner=lambda c, s: "dir",
        vram_reader=lambda: 9000,  # would trip the gate if it applied
        gpu_sampler=lambda: gpu(40),
        sleep=lambda _s: None,
    )
    assert report["completed"] == 1
    assert report["results"][0]["vram_residual_mb"] is None  # gate skipped for Ollama


def test_run_sweep_records_cell_failure_and_continues(tmp_path):
    def runner(config, split):
        if config.model == "bad:1":
            raise RuntimeError("boom")
        return "dir"

    report = run_sweep(
        [cfg(tmp_path, model="bad:1"), cfg(tmp_path, model="ok:1")],
        sweep_id="s4",
        data_dir=tmp_path,
        cell_runner=runner,
        vram_reader=lambda: 0,
        gpu_sampler=lambda: gpu(40),
        sleep=lambda _s: None,
    )
    assert report["failed"] == 1 and report["completed"] == 1
    bad = next(r for r in report["results"] if r["model"] == "bad:1")
    assert bad["status"] == "failed" and "boom" in bad["detail"]


# --- M3.3 isolate_cell + live PID attribution ---------------------------------------------


def test_isolate_cell_reclaimed_runs_work_and_cools_down():
    out, iso = isolate_cell(
        lambda: "result",
        backend="vllm",
        vram_reader=lambda: 1000,  # constant -> reclaimed within tolerance
        pid_usage_reader=lambda: {100: 500},
        gpu_sampler=lambda: gpu(40),
        sleep=lambda _s: None,
    )
    assert out == "result"
    assert iso["vram_verdict"] == VERDICT_RECLAIMED and iso["vram_residual_mb"] == 0


def test_isolate_cell_aborts_on_attributed_leak():
    reads = iter([1000] + [9000] * 100)  # baseline 1000, then stuck high -> residual 8000
    usage = iter([{100: 500}, {100: 500, 200: 3000}])  # a NEW pid (200) still holds VRAM
    with pytest.raises(VramNotReclaimed, match="leaked"):
        isolate_cell(
            lambda: "x",
            backend="vllm",
            vram_reader=lambda: next(reads),
            pid_usage_reader=lambda: next(usage),
            gpu_sampler=lambda: gpu(40),
            sleep=lambda _s: None,
        )


def test_isolate_cell_tolerates_baseline_shift():
    reads = iter([1000] + [9000] * 100)  # residual 8000
    usage = iter([{100: 500}, {100: 8500}])  # only the PRE-EXISTING pid 100 grew (no new pid)
    out, iso = isolate_cell(
        lambda: "done",
        backend="vllm",
        vram_reader=lambda: next(reads),
        pid_usage_reader=lambda: next(usage),
        gpu_sampler=lambda: gpu(40),
        sleep=lambda _s: None,
    )
    assert out == "done"
    assert iso["vram_verdict"] == VERDICT_BASELINE_SHIFT and iso["vram_residual_mb"] == 8000


def test_run_sweep_aborts_on_attributed_leak(tmp_path):
    reads = iter([1000] + [9000] * 200)
    usage = iter([{100: 500}, {100: 500, 200: 4000}])  # leaked launched pid 200
    with pytest.raises(VramNotReclaimed):
        run_sweep(
            [cfg(tmp_path, backend="vllm")],
            sweep_id="leak",
            data_dir=tmp_path,
            cell_runner=lambda c, s: "dir",
            vram_reader=lambda: next(reads),
            pid_usage_reader=lambda: next(usage),
            gpu_sampler=lambda: gpu(40),
            sleep=lambda _s: None,
        )


def test_run_sweep_completes_through_baseline_shift(tmp_path):
    reads = iter([1000] + [9000] * 200)
    usage = iter([{100: 500}, {100: 8500}])  # unrelated process grew -> not a leak
    report = run_sweep(
        [cfg(tmp_path, backend="vllm")],
        sweep_id="bshift",
        data_dir=tmp_path,
        cell_runner=lambda c, s: str(tmp_path / "d"),
        vram_reader=lambda: next(reads),
        pid_usage_reader=lambda: next(usage),
        gpu_sampler=lambda: gpu(40),
        sleep=lambda _s: None,
    )
    assert report["completed"] == 1 and report["results"][0]["vram_residual_mb"] == 8000
