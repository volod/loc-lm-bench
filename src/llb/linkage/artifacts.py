"""Read and write the linkage bundle under `$DATA_DIR/<method>/<run>/linkage/`.

The split is deliberate: `settings.json` is what was asked for, `blocking_counts.json` is what it
cost, `match_parameters.json` is what was learned, `model.json` is what a replay re-scores from,
and the two JSONL files are the decision itself -- so a later reader can diff any one of them
alone. `accuracy.json` joins them only where a reviewer-labelled set exists.
"""

import json
from pathlib import Path
from typing import Any

from llb.core.contracts.common import JsonObject
from llb.core.fsutil import atomic_write_text
from llb.linkage.constants import (
    ACCURACY_FILE,
    BLOCKING_COUNTS_FILE,
    CLUSTERS_FILE,
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
    settings: JsonObject = {
        "specification": result.spec.payload(),
        "summary": result.summary(),
    }
    if metadata:
        settings["metadata"] = dict(metadata)
    _write_json(out_dir / SETTINGS_FILE, settings)
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


def read_saved_spec(bundle_dir: Path) -> LinkageSpec:
    """Recover the specification a bundle was produced under (no Splink import needed)."""
    payload = json.loads((linkage_dir(bundle_dir) / SETTINGS_FILE).read_text(encoding="utf-8"))
    spec = LinkageSpec.from_payload(payload["specification"])
    spec.validate()
    return spec


def read_saved_summary(bundle_dir: Path) -> JsonObject:
    """The run summary a bundle recorded -- counts, threshold, and how the model was fitted."""
    payload = json.loads((linkage_dir(bundle_dir) / SETTINGS_FILE).read_text(encoding="utf-8"))
    summary: JsonObject = payload.get("summary", {})
    return summary


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
