"""Focused dataset io implementation."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from llb.core.contracts.common import JsonObject
from llb.core.fsutil import atomic_write_text

DATASET_MANIFEST = "dataset_manifest.json"

SFT_FILENAME = "sft.jsonl"

DPO_FILENAME = "dpo.jsonl"


def dataset_digest(sft_records: list[JsonObject], dpo_records: list[JsonObject]) -> str:
    payload = {"sft": sft_records, "dpo": dpo_records}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_dataset_manifest(dataset_dir: Path | str) -> JsonObject:
    return _read_json(Path(dataset_dir) / DATASET_MANIFEST, label="dataset manifest")


def _read_misses(path: Path | str | None) -> dict[str, JsonObject]:
    if path is None:
        return {}
    rows = _read_jsonl_if_exists(Path(path))
    return {str(row.get("item_id")): row for row in rows if row.get("item_id")}


def _miss_weights(misses: dict[str, JsonObject]) -> dict[str, float]:
    if not misses:
        return {}
    clusters = Counter(
        (
            str(row.get("miss_class", "?")),
            str(row.get("topic") or row.get("source_doc_id") or "?"),
        )
        for row in misses.values()
    )
    weights: dict[str, float] = {}
    for item_id, row in misses.items():
        key = (
            str(row.get("miss_class", "?")),
            str(row.get("topic") or row.get("source_doc_id") or "?"),
        )
        weights[item_id] = float(1 + clusters[key])
    return weights


def _read_json(path: Path, *, label: str) -> JsonObject:
    if not path.is_file():
        raise ValueError(f"{label} not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return data


def _read_jsonl(path: Path) -> list[JsonObject]:
    if not path.is_file():
        raise ValueError(f"JSONL file not found: {path}")
    return _read_jsonl_if_exists(path)


def _read_jsonl_if_exists(path: Path) -> list[JsonObject]:
    if not path.is_file():
        return []
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[JsonObject]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, content)
