"""Read and write the linkage bundle under `$DATA_DIR/<method>/<run>/linkage/`.

The split is deliberate: `settings.json` is what was asked for, `blocking_counts.json` is what it
cost, `match_parameters.json` is what was learned, `model.json` is what a replay re-scores from,
and the two JSONL files are the decision itself -- so a later reader can diff any one of them
alone. `accuracy.json` joins them only where a reviewer-labelled set exists.
"""

import json
from pathlib import Path
from typing import Any

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.serialization import stated_sections
from llb.core.contracts.common import JsonObject
from llb.core.contracts.data_prep.linkage import (
    LinkageSettings,
    LinkageSpecRecord,
    LinkageSummaryRecord,
)
from llb.core.fsutil import atomic_write_text
from llb.linkage.constants import (
    ACCURACY_FILE,
    BLOCKING_COUNTS_FILE,
    CLUSTERS_FILE,
    LINKAGE_SETTINGS_SCHEMA_ID,
    LINKAGE_SETTINGS_SCHEMA_VERSION,
    LINKAGE_SUBDIR,
    MATCH_PARAMETERS_FILE,
    MODEL_FILE,
    PAIRS_FILE,
    SETTINGS_FILE,
)
from llb.linkage.model import LinkageResult
from llb.linkage.spec import LinkageSpec


def linkage_dir(bundle_dir: Path) -> Path:
    """The `linkage/` subdirectory of a run bundle -- the seam's home in any run."""
    return Path(bundle_dir) / LINKAGE_SUBDIR


def _point(point: Any) -> JsonObject | None:
    return point.payload() if point is not None else None


def _write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Any) -> None:
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, body)


def write_linkage_artifacts(
    result: LinkageResult, bundle_dir: Path, metadata: JsonObject | None = None
) -> dict[str, str]:
    """Write the whole bundle and return the artifact name -> path map for a run manifest."""
    out_dir = linkage_dir(bundle_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = LinkageSettings(
        schema_id=LINKAGE_SETTINGS_SCHEMA_ID,
        schema_version=LINKAGE_SETTINGS_SCHEMA_VERSION,
        specification=LinkageSpecRecord.model_validate(result.spec.payload()),
        summary=LinkageSummaryRecord.model_validate(result.summary()),
        metadata=dict(metadata) if metadata else None,
    )
    _write_json(out_dir / SETTINGS_FILE, stated_sections(settings))
    _write_json(
        out_dir / BLOCKING_COUNTS_FILE,
        {"rules": [count.payload() for count in result.blocking_counts]},
    )
    _write_json(
        out_dir / MATCH_PARAMETERS_FILE,
        {"levels": [parameter.payload() for parameter in result.match_parameters]},
    )
    _write_json(out_dir / MODEL_FILE, result.trained_model)
    _write_jsonl(out_dir / PAIRS_FILE, (pair.payload() for pair in result.pairs))
    _write_jsonl(out_dir / CLUSTERS_FILE, (cluster.payload() for cluster in result.clusters))
    paths = {
        "settings": str(out_dir / SETTINGS_FILE),
        "blocking_counts": str(out_dir / BLOCKING_COUNTS_FILE),
        "match_parameters": str(out_dir / MATCH_PARAMETERS_FILE),
        "model": str(out_dir / MODEL_FILE),
        "pairs": str(out_dir / PAIRS_FILE),
        "clusters": str(out_dir / CLUSTERS_FILE),
    }
    if result.accuracy:
        _write_json(
            out_dir / ACCURACY_FILE,
            {
                "curve": [point.payload() for point in result.accuracy],
                "pair_operating_point": _point(result.pair_operating_point),
                "cluster_operating_point": _point(result.cluster_operating_point),
            },
        )
        paths["accuracy"] = str(out_dir / ACCURACY_FILE)
    return paths


def read_settings(bundle_dir: Path) -> LinkageSettings:
    """The bundle settings at the current contract, migrating an older bundle on the way.

    A pre-contract bundle carries no identity and left its tuning knobs to whatever build read it
    back; the registry stamps the version it was written at and the migration states those knobs,
    so a replay re-scores from the run's settings rather than from today's defaults.
    """
    path = linkage_dir(bundle_dir) / SETTINGS_FILE
    record = json.loads(path.read_text(encoding="utf-8"))
    settings = DEFAULT_REGISTRY.read_as(LINKAGE_SETTINGS_SCHEMA_ID, record, source=str(path))
    if not isinstance(settings, LinkageSettings):
        raise TypeError(f"{path}: linkage settings did not resolve to the current contract")
    return settings


def read_saved_spec(bundle_dir: Path) -> LinkageSpec:
    """Recover the specification a bundle was produced under (no Splink import needed)."""
    spec = LinkageSpec.from_payload(read_settings(bundle_dir).specification.model_dump())
    spec.validate()
    return spec


def read_saved_summary(bundle_dir: Path) -> JsonObject:
    """The run summary a bundle recorded -- counts, threshold, and how the model was fitted."""
    return read_settings(bundle_dir).summary.model_dump()


def read_saved_model(bundle_dir: Path) -> JsonObject:
    """Recover the trained model a bundle re-scores from."""
    model: JsonObject = json.loads(
        (linkage_dir(bundle_dir) / MODEL_FILE).read_text(encoding="utf-8")
    )
    return model


def read_pairs(bundle_dir: Path) -> list[JsonObject]:
    """The scored pairs of a written bundle, in the order they were written."""
    path = linkage_dir(bundle_dir) / PAIRS_FILE
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
