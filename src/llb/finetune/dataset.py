"""Deterministic fine-tuning dataset export from scored RAG bundles.

Exports only tuning-split, verified gold items. The SFT prompt is rendered through
`eval.graph.build_messages`, so the training records use the same chat shape as `run-eval`.
When miss analysis is present, missed tuning cases become the targeted export and carry cluster
weights; otherwise every scored tuning case is exported with weight 1.
"""

import json
from collections import Counter
from pathlib import Path

from llb.core.contracts.rag import ChunkRecord
from llb.core.contracts.common import JsonObject
from llb.core.fsutil import atomic_write_text
from llb.eval import common as eval_common
from llb.eval.graph import build_messages
from llb.goldset.schema import GoldItem, load_goldset
from llb.finetune.dataset_io import (
    DATASET_MANIFEST,
    DPO_FILENAME,
    SFT_FILENAME,
    _miss_weights,
    _read_json,
    _read_jsonl,
    _read_jsonl_if_exists,
    _read_misses,
    _write_jsonl,
    dataset_digest,
    load_dataset_manifest,
)

TUNING_SPLIT = "tuning"
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
    cited = bool((manifest.get("config") or {}).get("cited_answers", False))
    sft_records: list[JsonObject] = []
    dpo_records: list[JsonObject] = []
    for row in sorted(selected, key=lambda rec: str(rec.get("item_id"))):
        item = gold_by_id[str(row["item_id"])]
        context = _context_for_item(item, retrieval.get(item.id))
        messages = build_messages(item.question, context, cited=cited)
        weight = weights.get(item.id, 1.0)
        sft_records.append(_sft_record(item, messages, weight, run_dir))
        dpo = _dpo_record(item, row, messages, weight, run_dir)
        if dpo is not None:
            dpo_records.append(dpo)

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


def _sft_record(item: GoldItem, messages: object, weight: float, run_dir: Path) -> JsonObject:
    """One supervised fine-tuning record for a scored tuning item."""
    return {
        "item_id": item.id,
        "split": item.split,
        "weight": weight,
        "messages": messages,
        "response": item.reference_answer,
        "reference_answer": item.reference_answer,
        "source_run": str(run_dir),
        "prompt_template": "eval.rag.chat",
    }


def _dpo_record(
    item: GoldItem, row: JsonObject, messages: object, weight: float, run_dir: Path
) -> JsonObject | None:
    """A DPO pair when the row has a below-threshold rejected answer preview, else None."""
    rejected = str(row.get("answer_preview") or "").strip()
    if not rejected or float(row.get("objective_score", 0.0)) >= MISS_THRESHOLD:
        return None
    return {
        "item_id": item.id,
        "split": item.split,
        "weight": weight,
        "prompt": messages,
        "chosen": item.reference_answer,
        "rejected": rejected,
        "source_run": str(run_dir),
    }


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
