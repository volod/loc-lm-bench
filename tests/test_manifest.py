import json

import pytest

from llb.tracking.manifest import RunManifest, persist_run, write_scores


def make_manifest():
    return RunManifest(run_id="abc123", run_name="t", config={"model": "m"},
                       metrics={"objective_score": 0.5}, n_cases=2)


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
    seen = {}

    def mirror(manifest, out_dir):
        # The canonical record must already be on disk when the mirror runs.
        seen["manifest_exists"] = (out_dir / "manifest.json").exists()
        seen["scores_exists"] = any(out_dir.glob("scores.*"))

    paths = persist_run(make_manifest(), [{"item_id": "x"}], tmp_path, mirror=mirror)
    assert seen["manifest_exists"] is True
    assert seen["scores_exists"] is True
    assert paths["mirror"] == "ok"


def test_mirror_failure_does_not_lose_run(tmp_path):
    def boom(manifest, out_dir):
        raise RuntimeError("mlflow down")

    paths = persist_run(make_manifest(), [{"item_id": "x"}], tmp_path, mirror=boom)
    assert (tmp_path / "manifest.json").exists()
    assert paths["mirror"].startswith("failed")
    data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert data["run_id"] == "abc123"


def test_existing_canonical_run_is_not_overwritten(tmp_path):
    persist_run(make_manifest(), [{"item_id": "x"}], tmp_path, mirror=lambda *args: None)
    with pytest.raises(FileExistsError, match="already exist"):
        persist_run(make_manifest(), [{"item_id": "y"}], tmp_path, mirror=lambda *args: None)
