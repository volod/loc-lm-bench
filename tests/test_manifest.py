import json

import pytest

from llb.tracking import manifest as manifest_module
from llb.tracking.manifest import RunManifest, persist_run, write_scores


def make_manifest():
    return RunManifest(
        run_id="abc123",
        run_name="t",
        config={"model": "m"},
        metrics={"objective_score": 0.5, "reliability": 1.0, "tokens_per_s": 10.0},
        n_cases=2,
    )


def test_write_scores_format_follows_pyarrow_availability(tmp_path):
    # Parquet when pyarrow (the [track] extra) is present; JSONL fallback otherwise.
    out = write_scores([{"item_id": "x", "score": 1.0}], tmp_path / "scores")
    try:
        import pyarrow  # noqa: F401

        assert out.suffix == ".parquet"
    except ImportError:
        assert out.suffix == ".jsonl"
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert rows == [{"item_id": "x", "score": 1.0}]


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
