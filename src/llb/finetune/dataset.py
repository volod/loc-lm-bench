"""Deterministic fine-tuning dataset export from scored RAG bundles.

Exports only tuning-split, verified gold items. The SFT prompt is rendered through
`eval.graph.build_messages`, so the training records use the same chat shape as `run-eval`.
When miss analysis is present, missed tuning cases become the targeted export and carry cluster
weights; otherwise every scored tuning case is exported with weight 1.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from llb.core.contracts import ChunkRecord, JsonObject
from llb.core.fsutil import atomic_write_text
from llb.eval import common as eval_common
from llb.eval.graph import build_messages
from llb.goldset.schema import GoldItem, load_goldset

TUNING_SPLIT = "tuning"
DATASET_MANIFEST = "dataset_manifest.json"
SFT_FILENAME = "sft.jsonl"
DPO_FILENAME = "dpo.jsonl"
MISS_THRESHOLD = 0.5


def export_finetune_set(
    *,
    run_dir: Path | str,
    goldset_path: Path | str,
    out_dir: Path | str,
    misses_path: Path | str | None = None,
) -> JsonObject:
    """Export SFT and optional DPO records from one finalized tuning run bundle."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    manifest = _read_json(run_dir / "manifest.json", label="run manifest")
    rows = _read_jsonl(run_dir / "scores.jsonl")
    retrieval = {
        str(row["item_id"]): row for row in _read_jsonl_if_exists(run_dir / "retrieval.jsonl")
    }
    gold_by_id = {item.id: item for item in load_goldset(goldset_path)}
    misses = _read_misses(misses_path)
    miss_ids = set(misses)
    scored_tuning = [
        row
        for row in rows
        if (item := gold_by_id.get(str(row.get("item_id")))) is not None
        and item.verified
        and item.split == TUNING_SPLIT
    ]
    if not scored_tuning:
        raise ValueError("fine-tune export found no scored, verified tuning-split items")
    selected = [row for row in scored_tuning if str(row.get("item_id")) in miss_ids]
    if not selected:
        selected = scored_tuning

    weights = _miss_weights(misses)
    sft_records: list[JsonObject] = []
    dpo_records: list[JsonObject] = []
    for row in sorted(selected, key=lambda rec: str(rec.get("item_id"))):
        item = gold_by_id[str(row["item_id"])]
        context = _context_for_item(item, retrieval.get(item.id))
        messages = build_messages(
            item.question,
            context,
            cited=bool((manifest.get("config") or {}).get("cited_answers", False)),
        )
        weight = weights.get(item.id, 1.0)
        sft_records.append(
            {
                "item_id": item.id,
                "split": item.split,
                "weight": weight,
                "messages": messages,
                "response": item.reference_answer,
                "reference_answer": item.reference_answer,
                "source_run": str(run_dir),
                "prompt_template": "eval.rag.chat",
            }
        )
        rejected = str(row.get("answer_preview") or "").strip()
        if rejected and float(row.get("objective_score", 0.0)) < MISS_THRESHOLD:
            dpo_records.append(
                {
                    "item_id": item.id,
                    "split": item.split,
                    "weight": weight,
                    "prompt": messages,
                    "chosen": item.reference_answer,
                    "rejected": rejected,
                    "source_run": str(run_dir),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / SFT_FILENAME, sft_records)
    _write_jsonl(out_dir / DPO_FILENAME, dpo_records)
    digest = dataset_digest(sft_records, dpo_records)
    manifest_payload: JsonObject = {
        "kind": "llb.finetune.dataset",
        "dataset_digest": digest,
        "source_run": str(run_dir),
        "source_run_id": manifest.get("run_id"),
        "goldset_path": str(goldset_path),
        "misses_path": str(misses_path) if misses_path else None,
        "item_ids": [str(record["item_id"]) for record in sft_records],
        "split_counts": dict(Counter(str(record["split"]) for record in sft_records)),
        "n_sft": len(sft_records),
        "n_dpo": len(dpo_records),
        "prompt_template": "eval.rag.chat",
    }
    atomic_write_text(
        out_dir / DATASET_MANIFEST,
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest_payload


def subset_dataset(
    *,
    dataset_dir: Path | str,
    out_dir: Path | str,
    item_ids: list[str] | tuple[str, ...],
    role: str,
) -> JsonObject:
    """Materialize a real dataset directory holding only `item_ids`, with its own digest.

    A hyperparameter trial trains on a sub-slice of the tuning split, so it needs a dataset of its
    own rather than a filtered view: `adapter_digest` is derived from `dataset_digest`, so a subset
    that inherited its parent's digest would let two adapters trained on different data collide on
    one registry id.
    """
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    parent = load_dataset_manifest(dataset_dir)
    keep = {str(item_id) for item_id in item_ids}
    sft = [
        row
        for row in _read_jsonl_if_exists(dataset_dir / SFT_FILENAME)
        if str(row.get("item_id")) in keep
    ]
    dpo = [
        row
        for row in _read_jsonl_if_exists(dataset_dir / DPO_FILENAME)
        if str(row.get("item_id")) in keep
    ]
    if not sft:
        raise ValueError(f"dataset subset '{role}' selected no SFT records from {dataset_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / SFT_FILENAME, sft)
    _write_jsonl(out_dir / DPO_FILENAME, dpo)
    manifest_payload: JsonObject = {
        "kind": "llb.finetune.dataset",
        "dataset_digest": dataset_digest(sft, dpo),
        "parent_dataset_digest": parent.get("dataset_digest"),
        "subset_role": role,
        "source_run": parent.get("source_run"),
        "source_run_id": parent.get("source_run_id"),
        "goldset_path": parent.get("goldset_path"),
        "misses_path": parent.get("misses_path"),
        "item_ids": [str(record["item_id"]) for record in sft],
        "split_counts": dict(Counter(str(record["split"]) for record in sft)),
        "n_sft": len(sft),
        "n_dpo": len(dpo),
        "prompt_template": parent.get("prompt_template", "eval.rag.chat"),
    }
    atomic_write_text(
        out_dir / DATASET_MANIFEST,
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest_payload


def dataset_digest(sft_records: list[JsonObject], dpo_records: list[JsonObject]) -> str:
    payload = {"sft": sft_records, "dpo": dpo_records}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_dataset_manifest(dataset_dir: Path | str) -> JsonObject:
    return _read_json(Path(dataset_dir) / DATASET_MANIFEST, label="dataset manifest")


def _context_for_item(item: GoldItem, retrieval_record: JsonObject | None) -> str:
    chunks: list[ChunkRecord] = []
    if retrieval_record:
        for retrieved in retrieval_record.get("retrieved") or []:
            chunks.append(
                {
                    "doc_id": str(retrieved.get("doc_id", "?")),
                    "char_start": int(retrieved.get("char_start", 0)),
                    "char_end": int(retrieved.get("char_end", 0)),
                    "text": str(retrieved.get("text_preview") or ""),
                }
            )
    if not chunks:
        chunks = [
            {
                "doc_id": span.doc_id,
                "char_start": span.char_start,
                "char_end": span.char_end,
                "text": span.text,
            }
            for span in item.source_spans
        ]
    return eval_common.format_context(chunks)


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
