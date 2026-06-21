"""Hard-isolation sweep executor (M3.3), driven entirely by fakes (no GPU / subprocess)."""

import pytest

from llb.config import RunConfig
from llb.contracts import GpuSample
from llb.executor.isolation import (
    cell_key,
    cool_down,
    parse_smi_samples,
    run_sweep,
    sample_gpu,
)
from llb.executor.vram import VramNotReclaimed


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
