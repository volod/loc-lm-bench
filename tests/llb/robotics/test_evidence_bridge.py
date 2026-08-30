import json
import shutil
import sys
from pathlib import Path

import pytest

from llb.core.paths import PROJECT_ROOT
from llb.robotics.evidence_bridge import run_evidence_bridge
from llb.robotics.evidence_models import HflowProjectionRow
from llb.robotics.hflow_fixture import write_fixture_manifest
from llb.robotics.hflow_manifest import (
    PROJECTION_MANIFEST_NAME,
    load_projection_manifest,
    write_projection_manifest,
)
from llb.robotics.mcap_validation import inspect_mcap_channels

FIXTURE_ROOT = PROJECT_ROOT / "samples" / "robotics" / "hflow"


def _copy_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "hflow"
    shutil.copytree(FIXTURE_ROOT, target)
    return target


def _rewrite_rows(root: Path, rows: tuple[HflowProjectionRow, ...]) -> None:
    manifest = root / PROJECTION_MANIFEST_NAME
    manifest.unlink()
    write_projection_manifest(manifest, rows)
    write_fixture_manifest(root)


def test_committed_fixture_replays_without_importing_hflow(tmp_path: Path) -> None:
    sys.modules.pop("hflow", None)
    output, report = run_evidence_bridge(FIXTURE_ROOT, output_dir=tmp_path / "run")

    assert "hflow" not in sys.modules
    assert report["verdict"] == "pass"
    assert report["projection_count"] == 5
    assert report["episode_count"] == 2
    assert report["admission_counts"] == {
        "accepted": 2,
        "draft": 1,
        "quarantined": 1,
        "unverified": 1,
    }
    assert report["corpus_documents"] == 2
    corpus_documents = sorted((output / "corpus" / "robotics").rglob("*.md"))
    assert [path.name for path in corpus_documents] == [
        "clean-human-label.md",
        "clean-model-procedure.md",
    ]
    assert all(path.parent.name for path in corpus_documents)

    ledger = [
        json.loads(line)
        for line in (output / "evidence-ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(ledger) == 5
    assert all(item["generation"]["check_versions"] for item in ledger)
    assert all(item["generation"]["enrichment_versions"] for item in ledger)
    excluded = [item for item in ledger if item["admission"] != "accepted"]
    assert all(item["source_span"] is None for item in excluded)


def test_fixture_canonical_files_open_with_stock_mcap_reader() -> None:
    rows = load_projection_manifest(FIXTURE_ROOT / PROJECTION_MANIFEST_NAME)
    by_uri = {row.mcap_uri: row.channels for row in rows}
    windows = [
        inspect_mcap_channels(FIXTURE_ROOT / uri, channels) for uri, channels in by_uri.items()
    ]
    assert len(windows) == 2
    assert all(window.message_count == 20 for window in windows)


def test_fixture_refuses_stale_projection_bytes(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    path = root / "projections" / "clean-human-label.md"
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="stale HFlow fixture file"):
        run_evidence_bridge(root, output_dir=tmp_path / "run")


def test_bridge_refuses_mixed_pipeline_generation(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    rows = list(load_projection_manifest(root / PROJECTION_MANIFEST_NAME))
    rows[0] = rows[0].model_copy(update={"pipeline_version": "mixed-version"})
    _rewrite_rows(root, tuple(rows))
    with pytest.raises(ValueError, match="mixed HFlow generation"):
        run_evidence_bridge(root, output_dir=tmp_path / "run")


def test_model_projection_needs_accepted_exact_span(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    rows = list(load_projection_manifest(root / PROJECTION_MANIFEST_NAME))
    index = next(
        index for index, row in enumerate(rows) if row.authored_by == "model" and row.verified
    )
    rows[index] = rows[index].model_copy(update={"verification_ref": None})
    _rewrite_rows(root, tuple(rows))
    with pytest.raises(ValueError, match="needs verification_ref"):
        run_evidence_bridge(root, output_dir=tmp_path / "run")


def test_bridge_refuses_interval_without_channel_message(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    rows = list(load_projection_manifest(root / PROJECTION_MANIFEST_NAME))
    row = rows[0]
    rows[0] = row.model_copy(update={"start_ns": row.start_ns + 1, "end_ns": row.start_ns + 2})
    _rewrite_rows(root, tuple(rows))
    with pytest.raises(ValueError, match="interval contains no messages"):
        run_evidence_bridge(root, output_dir=tmp_path / "run")
