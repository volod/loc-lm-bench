"""Shared MLflow mirror and canonical-run reconciliation."""

import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, TypeGuard
from urllib.parse import unquote, urlparse

from llb.core.paths import resolve_data_dir, resolve_project_path
from llb.tracking.manifest import RunManifest

_LOG = logging.getLogger(__name__)

MLFLOW_EXPERIMENT = "loc-lm-bench"
MLFLOW_STORE_DIR = "mlflow"
MLFLOW_MIRROR_SCHEMA = "3"
MLFLOW_RUN_ID_TAG = "llb.canonical_run_id"
MLFLOW_SCHEMA_TAG = "llb.mirror_schema"


def mirror_run(manifest: RunManifest, out_dir: Path) -> None:
    """Mirror one canonical run into the shared local MLflow SQLite store."""
    client, experiment_id = _mlflow_client(_mlflow_root(manifest, out_dir))
    _log_mlflow_run(client, experiment_id, manifest, out_dir)


def _mlflow_client(tracking_root: Path) -> tuple[Any, str]:
    import mlflow
    from mlflow.tracking import MlflowClient

    artifact_root = tracking_root / "artifacts"
    tracking_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{tracking_root / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)

    client: Any = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(
            MLFLOW_EXPERIMENT,
            artifact_location=artifact_root.resolve().as_uri(),
        )
    )
    return client, experiment_id


def _log_mlflow_run(
    client: Any,
    experiment_id: str,
    manifest: RunManifest,
    out_dir: Path,
    existing_run_id: str | None = None,
) -> str:
    display_name = _mlflow_run_name(manifest)
    tags = _mlflow_tags(manifest)
    created = existing_run_id is None
    run_id = existing_run_id
    if created:
        run = client.create_run(
            experiment_id,
            start_time=_timestamp_ms(manifest.created_at),
            tags=tags,
            run_name=display_name,
        )
        run_id = run.info.run_id
    assert run_id is not None
    try:
        client.set_tag(run_id, "mlflow.runName", display_name)
        for tag_key, tag_value in tags.items():
            client.set_tag(run_id, tag_key, tag_value)
        for param_key, param_value in manifest.config.items():
            if _scalar(param_value):
                client.log_param(run_id, param_key, param_value)
        for metric_key, metric_value in _mlflow_metrics(manifest).items():
            client.log_metric(run_id, metric_key, metric_value)
        _log_canonical_artifacts(client, run_id, out_dir)
        client.set_tag(run_id, MLFLOW_SCHEMA_TAG, MLFLOW_MIRROR_SCHEMA)
        client.set_terminated(
            run_id,
            status="FINISHED",
            end_time=int((out_dir / "manifest.json").stat().st_mtime * 1000),
        )
    except Exception:
        if created:
            client.set_terminated(run_id, status="FAILED")
        raise
    return run_id


def _mlflow_run_name(manifest: RunManifest) -> str:
    model = str(manifest.config.get("model", "unknown-model"))
    backend = str(manifest.config.get("backend", "unknown-backend"))
    return f"{model} | {backend} | {manifest.run_id}"


def _mlflow_tags(manifest: RunManifest) -> dict[str, str]:
    tags = {
        MLFLOW_RUN_ID_TAG: manifest.run_id,
        "llb.run_name": manifest.run_name,
        "llb.created_at": manifest.created_at,
    }
    if manifest.split is not None:
        tags["llb.split"] = manifest.split
    for key in ("model", "backend"):
        value = manifest.config.get(key)
        if isinstance(value, str):
            tags[f"llb.{key}"] = value
    gpus = manifest.telemetry["gpus"] if manifest.telemetry is not None else []
    if gpus:
        tags["llb.gpus"] = ", ".join(str(gpu["name"]) for gpu in gpus)
        tags["llb.gpu_drivers"] = ", ".join(str(gpu["driver"]) for gpu in gpus)
    return tags


def _mlflow_metrics(manifest: RunManifest) -> dict[str, float]:
    values: dict[str, float] = {"cases.n": float(manifest.n_cases)}

    def add(prefix: str, record: Mapping[str, object] | None) -> None:
        for key, value in (record or {}).items():
            if _numeric(value):
                values[f"{prefix}.{key}"] = float(value)

    add("quality", manifest.metrics)
    add("retrieval", manifest.retrieval)
    add("telemetry", manifest.telemetry)
    add("judge", manifest.judge)
    if manifest.judge is not None:
        values["judge.trusted"] = float(manifest.judge["trusted"])
    gpus = manifest.telemetry["gpus"] if manifest.telemetry is not None else []
    values["hardware.gpu_count"] = float(len(gpus))
    values["hardware.gpu_total_mb"] = float(sum(gpu["total_mb"] for gpu in gpus))
    return values


def _log_canonical_artifacts(client: Any, run_id: str, out_dir: Path) -> None:
    for path in (
        out_dir / "manifest.json",
        *sorted(out_dir.glob("scores.*")),
        *sorted(out_dir.glob("report.*")),
    ):
        if path.is_file():
            client.log_artifact(run_id, str(path), artifact_path="canonical")
    backend_logs = out_dir / "vllm"
    if backend_logs.is_dir():
        client.log_artifacts(run_id, str(backend_logs), artifact_path="canonical/vllm")


def _timestamp_ms(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def _canonical_run_id(client: Any, run: Any) -> str | None:
    tagged = run.data.tags.get(MLFLOW_RUN_ID_TAG)
    if tagged:
        return str(tagged)
    for artifact in ("canonical/manifest.json", "manifest.json"):
        try:
            parsed = urlparse(run.info.artifact_uri)
            path = (
                Path(unquote(parsed.path)) / artifact
                if parsed.scheme == "file"
                else Path(client.download_artifacts(run.info.run_id, artifact))
            )
            manifest = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        client.set_tag(run.info.run_id, MLFLOW_RUN_ID_TAG, manifest.run_id)
        return manifest.run_id
    return None


def sync_mlflow_runs(data_dir: Path | None = None) -> dict[str, int]:
    """Reconcile canonical run artifacts into the shared MLflow mirror."""
    data_root = resolve_data_dir(data_dir)
    client, experiment_id = _mlflow_client(data_root / MLFLOW_STORE_DIR)
    existing: dict[str, Any] = {}
    for run in client.search_runs([experiment_id]):
        canonical_id = _canonical_run_id(client, run)
        if canonical_id is not None:
            existing[canonical_id] = run

    report = {"created": 0, "updated": 0, "current": 0, "failed": 0}
    for manifest_path in sorted((data_root / "run-eval").glob("*/manifest.json")):
        try:
            manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            previous = existing.get(manifest.run_id)
            if (
                previous is not None
                and previous.data.tags.get(MLFLOW_SCHEMA_TAG) == MLFLOW_MIRROR_SCHEMA
            ):
                report["current"] += 1
                continue
            _log_mlflow_run(
                client,
                experiment_id,
                manifest,
                manifest_path.parent,
                existing_run_id=previous.info.run_id if previous is not None else None,
            )
            report["updated" if previous is not None else "created"] += 1
        except Exception as exc:
            report["failed"] += 1
            _LOG.warning("[mlflow] failed to synchronize %s: %s", manifest_path, exc)
    return report


def _mlflow_root(manifest: RunManifest, out_dir: Path) -> Path:
    configured = manifest.config.get("data_dir")
    data_dir = resolve_project_path(configured) if isinstance(configured, str) else out_dir.parent
    return data_dir / MLFLOW_STORE_DIR


def _scalar(value: object) -> TypeGuard[str | int | float | bool]:
    return isinstance(value, (str, int, float, bool))


def _numeric(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
