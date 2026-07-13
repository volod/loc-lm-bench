"""Read a finalized run bundle for miss analysis: manifest + score rows + per-item retrieval
records, plus the optional draft-bundle provenance sidecar.

Pure and file-driven -- no endpoint, GPU, or store -- so the classifier stays unit-testable over a
synthetic bundle. Legacy bundles predating `retrieval.jsonl` yield an empty retrieval map and the
classifier falls back to the scored `retrieval_hit` signal.
"""

import json
import logging
from pathlib import Path

from llb.board.miss_analysis.model import ITEM_PROVENANCE_FILENAME, RETRIEVAL_FILENAME
from llb.core.contracts import JsonObject

_LOG = logging.getLogger(__name__)


def load_scored_bundle(
    run_dir: Path | str,
) -> tuple[JsonObject, list[JsonObject], dict[str, JsonObject]]:
    """Read a finalized run bundle: (manifest, score rows, retrieval records by item id).

    The retrieval map is empty for legacy bundles that predate `retrieval.jsonl`; the
    classifier then falls back to the scored `retrieval_hit` signal.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    scores_path = run_dir / "scores.jsonl"
    if not manifest_path.is_file() or not scores_path.is_file():
        raise SystemExit(
            f"[analyze-misses] {run_dir} is not a finalized run bundle "
            "(manifest.json + scores.jsonl required)"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(scores_path)
    retrieval_path = run_dir / RETRIEVAL_FILENAME
    retrieval: dict[str, JsonObject] = {}
    if retrieval_path.is_file():
        retrieval = {str(rec["item_id"]): rec for rec in _read_jsonl(retrieval_path)}
    else:
        _LOG.warning(
            "[analyze-misses] %s has no %s (older bundle); span-overlap classification "
            "falls back to the scored retrieval_hit signal",
            run_dir,
            RETRIEVAL_FILENAME,
        )
    return manifest, rows, retrieval


def _read_jsonl(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_item_provenance(goldset_path: Path | str) -> dict[str, JsonObject]:
    """Draft-bundle sidecar labels (`item_provenance.jsonl` beside the goldset), keyed by item
    id. Soft input: absent for plain goldsets, then heuristics label question type and topic."""
    sidecar = Path(goldset_path).parent / ITEM_PROVENANCE_FILENAME
    if not sidecar.is_file():
        return {}
    return {str(row.get("id")): row for row in _read_jsonl(sidecar)}
