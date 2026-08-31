"""Run the pinned HFlow dev loop and export its portable bridge fixture."""

import importlib.metadata
import json
import logging
from pathlib import Path
from typing import Any

from llb.core.fsutil import atomic_write_text
from llb.robotics.digests import file_digest
from llb.robotics.evidence_bridge import run_evidence_bridge
from llb.robotics.hflow_fixture import (
    CHECK_NAME,
    CHECK_VERSION,
    CURATION_SQL,
    ENRICHMENT_NAME,
    ENRICHMENT_VERSION,
    build_hflow_fixture,
)
from llb.robotics.upstreams import HFLOW_REVISION, HFLOW_VERSION

LOGGER = logging.getLogger(__name__)
MAX_JOINT_STEP_RAD = 0.4
SYNTHETIC_DURATION_S = 1.0
SYNTHETIC_JOINT_HZ = 20.0
SYNTHETIC_JUMP_AT_S = 0.5


def _require_exact_hflow(hflow: Any) -> None:
    observed_version = importlib.metadata.version("hflow")
    if observed_version != HFLOW_VERSION or hflow.__version__ != HFLOW_VERSION:
        raise ValueError(
            f"HFlow integration requires version {HFLOW_VERSION}, got {observed_version}"
        )
    direct_url_text = importlib.metadata.distribution("hflow").read_text("direct_url.json")
    if direct_url_text is None:
        raise ValueError("HFlow integration requires an exact direct-VCS installation")
    direct_url = json.loads(direct_url_text)
    observed_revision = direct_url.get("vcs_info", {}).get("commit_id")
    if observed_revision != HFLOW_REVISION:
        raise ValueError(
            f"HFlow integration requires revision {HFLOW_REVISION}, got {observed_revision}"
        )


def _build_app(hflow: Any, data_root: Path) -> Any:
    app = hflow.App("llb-robotics-evidence", data_root=data_root, default_checks=())

    @app.check(version=CHECK_VERSION, name=CHECK_NAME, critical=True)
    def bridge_quality(ep: Any) -> Any:
        import numpy as np

        joints = ep.channel("/joint_states").to_numpy()
        max_step = float(np.abs(np.diff(joints, axis=0)).max())
        return hflow.CheckResult(
            measurements={"max_joint_step_rad": max_step},
            verdict=max_step <= MAX_JOINT_STEP_RAD,
        )

    @app.enrich(version=ENRICHMENT_VERSION, name=ENRICHMENT_NAME)
    def bridge_projection(ep: Any) -> Any:
        artifact = ep.workdir / "projection.txt"
        artifact.write_text("HFlow enrichment artifact.\n", encoding="utf-8")
        return hflow.EnrichmentResult(
            labels={"caption": "Smooth synthetic joint trajectory."},
            artifacts={"projection": artifact},
        )

    return app


def _synthesize_sources(root: Path) -> tuple[Path, Path]:
    from hflow.testing import SyntheticEpisodeSpec, synthesize_episode

    source_root = root / "sources"
    clean = synthesize_episode(
        source_root / "clean.mcap",
        SyntheticEpisodeSpec(
            duration_s=SYNTHETIC_DURATION_S,
            cameras=(),
            joint_hz=SYNTHETIC_JOINT_HZ,
            joint_jump_at_s=None,
            black_segment=None,
            timestamp_offset_segment=None,
            seed=11,
        ),
    )
    bad = synthesize_episode(
        source_root / "quarantined.mcap",
        SyntheticEpisodeSpec(
            duration_s=SYNTHETIC_DURATION_S,
            cameras=(),
            joint_hz=SYNTHETIC_JOINT_HZ,
            joint_jump_at_s=SYNTHETIC_JUMP_AT_S,
            black_segment=None,
            timestamp_offset_segment=None,
            seed=17,
        ),
    )
    return clean, bad


def _validate_curation(path: Path, reports: tuple[Any, Any]) -> None:
    import duckdb

    rows = duckdb.sql(
        "SELECT episode_id, schema_version, pipeline_version, status "
        "FROM read_parquet(?) ORDER BY episode_id",
        params=[str(path)],
    ).fetchall()
    expected = sorted(
        (
            str(report.catalog_entry.episode_id),
            report.stamps.schema_version,
            report.stamps.pipeline_version,
            "quarantined" if report.quarantined else "ok",
        )
        for report in reports
    )
    if rows != expected:
        raise ValueError("HFlow curated Parquet rows do not match app.test catalog outputs")


def run_hflow_integration(run_root: Path, export_fixture: Path | None = None) -> dict[str, object]:
    """Exercise pinned ``app.test()``/``curate`` and replay the projected corpus."""
    try:
        import hflow
    except ImportError as exc:
        raise RuntimeError("install the exact pinned HFlow package before integration") from exc

    _require_exact_hflow(hflow)
    run_root.mkdir(parents=True, exist_ok=False)
    clean_source, bad_source = _synthesize_sources(run_root)
    app = _build_app(hflow, run_root / "hflow-data")
    clean_report = app.test(clean_source, verbose=False, record=True)
    bad_report = app.test(bad_source, verbose=False, record=True)
    if clean_report.has_errors or clean_report.quarantined:
        raise ValueError("pinned HFlow clean episode did not pass the integration pipeline")
    if bad_report.has_errors or not bad_report.quarantined:
        raise ValueError("pinned HFlow defect episode was not quarantined")
    if clean_report.stamps.pipeline_version != bad_report.stamps.pipeline_version:
        raise ValueError("HFlow app.test outputs use mixed pipeline versions")

    curated_path = run_root / "hflow-curated.parquet"
    curate_report = hflow.curate(app.workspace.catalog_root, CURATION_SQL, output=curated_path)
    if curate_report.row_count != 2:
        raise ValueError(f"HFlow curation expected 2 episodes, got {curate_report.row_count}")
    _validate_curation(curated_path, (clean_report, bad_report))
    fixture_root = run_root / "portable-fixture"
    fixture = build_hflow_fixture(fixture_root, clean_report, bad_report)
    _output, bridge_report = run_evidence_bridge(
        fixture_root,
        output_dir=run_root / "bridge",
    )
    if export_fixture is not None:
        import shutil

        if export_fixture.exists():
            raise ValueError(f"refusing to replace an existing fixture: {export_fixture}")
        shutil.copytree(fixture_root, export_fixture)

    report: dict[str, object] = {
        "schema_version": 1,
        "verdict": "pass",
        "hflow_version": HFLOW_VERSION,
        "hflow_revision": HFLOW_REVISION,
        "pipeline_version": clean_report.stamps.pipeline_version,
        "hflow_schema_version": clean_report.stamps.schema_version,
        "app_test_episodes": 2,
        "clean_status": "accepted",
        "defect_status": "quarantined",
        "curated_rows": curate_report.row_count,
        "curated_manifest_sha256": file_digest(curated_path),
        "fixture_files": len(fixture.files),
        "bridge": bridge_report,
    }
    atomic_write_text(
        run_root / "integration-report.json",
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )
    LOGGER.info("pinned HFlow integration report written to %s", run_root)
    return report
