import json

import pytest

from llb.tracking import manifest as manifest_module
from llb.tracking import mlflow as mlflow_module
from llb.tracking.manifest import RunManifest, persist_run
from llb.tracking.server import build_mlflow_command


def make_manifest():
    return RunManifest(
        run_id="abc123",
        run_name="t",
        config={"model": "m"},
        metrics={"objective_score": 0.5, "reliability": 1.0, "tokens_per_s": 10.0},
        n_cases=2,
    )


def test_manifest_written_before_mirror(tmp_path):
    out_dir = tmp_path / "run"
    seen = {}

    def mirror(manifest, out_dir):
        # The canonical record must already be on disk when the mirror runs.
        seen["manifest_exists"] = (out_dir / "manifest.json").exists()
        seen["scores_exists"] = any(out_dir.glob("scores.*"))

    paths = persist_run(make_manifest(), [{"item_id": "x"}], out_dir, mirror=mirror)
    assert seen["manifest_exists"] is True
    assert seen["scores_exists"] is True
    assert paths["mirror"] == "ok"

    configured = make_manifest().model_copy(
        update={"config": {"model": "m", "data_dir": str(tmp_path / "data")}}
    )
    assert mlflow_module._mlflow_root(configured, out_dir) == tmp_path / "data" / "mlflow"

    command = build_mlflow_command(
        tmp_path / ".venv" / "bin" / "mlflow",
        tmp_path / "data" / "mlflow" / "mlflow.db",
        tmp_path / "data" / "mlflow" / "artifacts",
        "127.0.0.1",
        5000,
    )
    assert command[-4:] == ["--host", "127.0.0.1", "--port", "5000"]
    assert command[command.index("--backend-store-uri") + 1].startswith("sqlite:////")

    rich_manifest = RunManifest(
        run_id="run-42",
        run_name="eval",
        split="final",
        config={"model": "model-uk", "backend": "ollama"},
        metrics={"objective_score": 0.75, "reliability": 1.0, "tokens_per_s": 20.0},
        retrieval={"n": 4, "k": 5, "recall_at_k": 1.0, "mrr": 0.8},
        judge={"calibration_rho": None, "threshold": 0.6, "trusted": False},
        telemetry={
            "steady_tokens_per_s": 20.0,
            "mean_completion_tokens": 10.0,
            "tokens_per_char": 0.25,
            "max_new_tokens": 128,
            "n_warmup": 1,
            "n_measured": 3,
            "n_failed": 0,
            "load_time_s": None,
            "peak_vram_mb": 4000,
            "requested_context": None,
            "served_context": None,
            "backend": "ollama",
            "gpu_memory_utilization": None,
            "n_gpu_layers": None,
            "gpus": [{"name": "GPU", "total_mb": 16000, "driver": "1.0"}],
        },
        n_cases=4,
    )
    metrics = mlflow_module._mlflow_metrics(rich_manifest)
    assert metrics["quality.objective_score"] == 0.75
    assert "objective_score" not in metrics
    assert metrics["retrieval.mrr"] == 0.8
    assert metrics["telemetry.peak_vram_mb"] == 4000
    assert metrics["hardware.gpu_total_mb"] == 16000
    assert mlflow_module._mlflow_tags(rich_manifest)["llb.canonical_run_id"] == "run-42"
    assert mlflow_module._mlflow_tags(rich_manifest)["llb.split"] == "final"
    assert mlflow_module._mlflow_run_name(rich_manifest) == "model-uk | ollama | run-42"


def test_mirror_failure_does_not_lose_run(tmp_path):
    out_dir = tmp_path / "run"

    def boom(manifest, out_dir):
        raise RuntimeError("mlflow down")

    paths = persist_run(make_manifest(), [{"item_id": "x"}], out_dir, mirror=boom)
    assert (out_dir / "manifest.json").exists()
    assert paths["mirror"].startswith("failed")
    data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert data["run_id"] == "abc123"


def test_existing_canonical_run_is_not_overwritten(tmp_path):
    out_dir = tmp_path / "run"
    persist_run(make_manifest(), [{"item_id": "x"}], out_dir, mirror=lambda *args: None)
    with pytest.raises(FileExistsError, match="already exist"):
        persist_run(make_manifest(), [{"item_id": "y"}], out_dir, mirror=lambda *args: None)


def test_canonical_bundle_is_not_published_when_score_write_fails(tmp_path, monkeypatch):
    out_dir = tmp_path / "run"
    staging_dir = tmp_path / ".run.tmp"
    (staging_dir / "vllm").mkdir(parents=True)
    (staging_dir / "vllm" / "server.log").write_text("log", encoding="utf-8")

    def fail_scores(rows, path):
        raise OSError("disk full")

    monkeypatch.setattr(manifest_module, "write_scores", fail_scores)
    with pytest.raises(OSError, match="disk full"):
        persist_run(
            make_manifest(),
            [{"item_id": "x"}],
            out_dir,
            mirror=lambda *args: None,
            staging_dir=staging_dir,
        )

    assert not out_dir.exists()
    assert not staging_dir.exists()


def test_staged_backend_logs_publish_with_canonical_bundle(tmp_path):
    out_dir = tmp_path / "run"
    staging_dir = tmp_path / ".run.tmp"
    (staging_dir / "vllm").mkdir(parents=True)
    (staging_dir / "vllm" / "server.log").write_text("log", encoding="utf-8")

    persist_run(
        make_manifest(),
        [{"item_id": "x"}],
        out_dir,
        mirror=lambda *args: None,
        staging_dir=staging_dir,
    )

    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "vllm" / "server.log").read_text(encoding="utf-8") == "log"
    assert not staging_dir.exists()
