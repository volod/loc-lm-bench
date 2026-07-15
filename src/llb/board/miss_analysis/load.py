"""Read finalized score and retrieval artifacts for miss analysis."""

import json
from pathlib import Path

from llb.board.miss_analysis.model import ITEM_PROVENANCE_FILENAME, RETRIEVAL_FILENAME
from llb.core.contracts.common import JsonObject


def load_scored_bundle(
    run_dir: Path | str,
) -> tuple[JsonObject, list[JsonObject], dict[str, JsonObject]]:
    """Read a finalized run bundle: (manifest, score rows, retrieval records by item id)."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    scores_path = run_dir / "scores.jsonl"
    retrieval_path = run_dir / RETRIEVAL_FILENAME
    if not manifest_path.is_file() or not scores_path.is_file() or not retrieval_path.is_file():
        raise SystemExit(
            f"[analyze-misses] {run_dir} is not a finalized run bundle "
            f"(manifest.json + scores.jsonl + {RETRIEVAL_FILENAME} required)"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(scores_path)
    retrieval = {str(rec["item_id"]): rec for rec in _read_jsonl(retrieval_path)}
    missing = [str(row.get("item_id")) for row in rows if str(row.get("item_id")) not in retrieval]
    if missing:
        raise SystemExit(
            f"[analyze-misses] {RETRIEVAL_FILENAME} is missing {len(missing)} scored item(s): "
            + ", ".join(missing[:5])
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
