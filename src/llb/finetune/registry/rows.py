"""Adapter-registry rows for CLI and board presentation."""

from llb.core.contracts.common import JsonObject
from llb.finetune.registry.model import AdapterEntry
from llb.finetune.registry.staleness import staleness
from llb.finetune.adapter_manifest import ADAPTER_DIGEST_SHORT_CHARS


def adapter_rows(entries: dict[str, AdapterEntry]) -> list[JsonObject]:
    """Build newest-first adapter rows with their current staleness verdict."""
    ordered = sorted(entries.values(), key=lambda entry: entry.recency, reverse=True)
    rows: list[JsonObject] = []
    for entry in ordered:
        report = staleness(entry)
        rows.append(
            {
                "adapter_id": entry.short_id,
                "base_model": entry.base_model,
                "staleness": report.verdict,
                "reasons": list(report.reasons),
                "dataset_digest": entry.dataset_digest[:ADAPTER_DIGEST_SHORT_CHARS],
                "objective_score": (entry.eval_summary or {}).get("objective_score"),
                "delta": (entry.eval_summary or {}).get("delta"),
                "source_run": entry.source_run,
                "adapter_dir": str(entry.adapter_dir),
                "merges": [str(merge.get("backend")) for merge in entry.merges],
                "created_at": entry.created_at,
            }
        )
    return rows
