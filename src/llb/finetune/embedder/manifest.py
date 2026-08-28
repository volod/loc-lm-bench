"""Manifests and digests for the embedder fine-tune lane.

Two artifacts are recorded, and each answers a different question a reader will have later:

  - `pairs_manifest.json` says WHAT WAS TRAINED ON -- the gold item ids, their split counts, the
    corpus fingerprint, and the chunking the pairs were cut with. It is what the split guard
    re-checks against the gold set (`llb.finetune.guard.assert_tuning_only`).
  - `embedder_manifest.json`, written INTO the tuned model directory, says WHICH ENCODER THIS IS --
    the base model whose convention it inherits, and a digest over (base, dataset, seed,
    hyperparameters). The digest is the identity a store records, so training again into the same
    directory is visibly a different encoder rather than silently the same one
    (`llb.rag.encoders.tuned`).
"""

import hashlib
import json
from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.core.fsutil import atomic_write_text
from llb.rag.encoders.tuned import MANIFEST_KIND, TUNED_EMBEDDER_MANIFEST

PAIRS_MANIFEST = "pairs_manifest.json"
PAIRS_KIND = "llb.finetune.embedder.pairs"

# Re-exported so the training lane never has to know where the reader's constants live.
TUNED_MANIFEST = TUNED_EMBEDDER_MANIFEST
TUNED_KIND = MANIFEST_KIND


def pairs_digest(records: list[JsonObject]) -> str:
    """Content digest over the exported rows: the same pairs digest the same on any host."""
    blob = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def tuned_digest(
    base_model: str, dataset_digest: str, seed: int, hyperparameters: JsonObject
) -> str:
    """Identity of one training: same base, data, seed, and configuration -- same encoder."""
    payload = {
        "base_model": base_model,
        "dataset_digest": dataset_digest,
        "seed": seed,
        "hyperparameters": hyperparameters,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_json_manifest(path: Path, manifest: JsonObject) -> None:
    """Write one manifest atomically, sorted, ASCII-safe line endings."""
    atomic_write_text(path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def load_pairs_manifest(pairs_dir: Path | str) -> JsonObject:
    """Read a pair-export manifest, refusing anything that is not one."""
    path = Path(pairs_dir) / PAIRS_MANIFEST
    if not path.is_file():
        raise ValueError(f"embedder pairs manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("kind") != PAIRS_KIND:
        raise ValueError(f"not an embedder pairs manifest: {path}")
    return data


def tuned_manifest(
    *,
    base_model: str,
    convention_family: str,
    pairs: JsonObject,
    pairs_dir: Path | str,
    seed: int,
    hyperparameters: JsonObject,
    trainer: str,
    loss_curve: list[float],
) -> JsonObject:
    """The record written into the tuned model directory (see the module docstring)."""
    dataset_digest = str(pairs["dataset_digest"])
    return {
        "kind": TUNED_KIND,
        "base_model": base_model,
        "convention_family": convention_family,
        "tuned_digest": tuned_digest(base_model, dataset_digest, seed, hyperparameters),
        "dataset_digest": dataset_digest,
        "pairs_manifest": str(Path(pairs_dir) / PAIRS_MANIFEST),
        "item_ids": list(pairs.get("item_ids") or []),
        "split_counts": dict(pairs.get("split_counts") or {}),
        "n_pairs": int(pairs.get("n_pairs") or 0),
        "negatives_per_pair": int(pairs.get("negatives_per_pair") or 0),
        "corpus_fingerprint": pairs.get("corpus_fingerprint"),
        "seed": seed,
        "trainer": trainer,
        "hyperparameters": hyperparameters,
        "loss_curve": [round(float(value), 6) for value in loss_curve],
    }


def write_tuned_manifest(out_dir: Path | str, manifest: JsonObject) -> Path:
    """Write `embedder_manifest.json` into the tuned model directory and return its path."""
    path = Path(out_dir) / TUNED_MANIFEST
    write_json_manifest(path, manifest)
    return path
