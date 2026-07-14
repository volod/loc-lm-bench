"""Tests for isolation failures."""

from llb.executor.isolation import (
    run_sweep,
)
from test_isolation import cfg, gpu


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
