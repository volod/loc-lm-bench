"""Drift report: re-run retrieval validation on the gold set after a refresh.

Scores the old and new stores' recall@k / MRR over the same gold items, reports the deltas, and
recommends a re-tune when either absolute delta crosses the configured threshold. The report is
advisory only -- re-tuning stays an operator (or orchestrator) decision, and the gold set is
never regenerated here: items whose gold spans point into changed documents scoring lower is
exactly the drift signal the report exists to surface.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.core.contracts.common import JsonObject
from llb.rag.refresh.diff import ManifestDiff
from llb.rag.retrieval import evaluate_retrieval

DEFAULT_RETUNE_THRESHOLD = 0.05

DRIFT_JSON = "drift.json"
DRIFT_REPORT_MD = "report.md"


@dataclass(frozen=True)
class RetrievalDrift:
    """Old-vs-new gold-set retrieval metrics and the re-tune recommendation."""

    n_items: int
    k: int
    old_recall: float
    old_mrr: float
    new_recall: float
    new_mrr: float
    threshold: float

    @property
    def delta_recall(self) -> float:
        return self.new_recall - self.old_recall

    @property
    def delta_mrr(self) -> float:
        return self.new_mrr - self.old_mrr

    @property
    def retune_recommended(self) -> bool:
        return abs(self.delta_recall) >= self.threshold or abs(self.delta_mrr) >= self.threshold


def _store_metrics(store: Any, items: list[Any], k: int) -> tuple[float, float]:
    from llb.executor.cases import spans_as_dicts

    pairs = [(store.retrieve(item.question, k), spans_as_dicts(item)) for item in items]
    metrics = evaluate_retrieval(pairs, k)
    return float(metrics["recall_at_k"]), float(metrics["mrr"])


def measure_drift(
    old_store: Any,
    new_store: Any,
    items: list[Any],
    *,
    k: int = 10,
    threshold: float = DEFAULT_RETUNE_THRESHOLD,
) -> RetrievalDrift:
    """Re-run retrieval validation over `items` against the old and new stores."""
    old_recall, old_mrr = _store_metrics(old_store, items, k)
    new_recall, new_mrr = _store_metrics(new_store, items, k)
    return RetrievalDrift(
        n_items=len(items),
        k=k,
        old_recall=old_recall,
        old_mrr=old_mrr,
        new_recall=new_recall,
        new_mrr=new_mrr,
        threshold=threshold,
    )


def drift_payload(
    diff: ManifestDiff, drift: RetrievalDrift | None, generation_dir: Path | str | None
) -> JsonObject:
    """The JSON-serializable drift report (`drift=None` == validation skipped, diff only)."""
    payload: JsonObject = {
        "generation_dir": str(generation_dir) if generation_dir is not None else None,
        "diff": diff.counts(),
        "added": diff.added,
        "modified": diff.modified,
        "deleted": diff.deleted,
    }
    if drift is not None:
        payload["retrieval"] = {
            "n_items": drift.n_items,
            "k": drift.k,
            "old": {"recall_at_k": drift.old_recall, "mrr": drift.old_mrr},
            "new": {"recall_at_k": drift.new_recall, "mrr": drift.new_mrr},
            "delta": {"recall_at_k": drift.delta_recall, "mrr": drift.delta_mrr},
            "retune_threshold": drift.threshold,
            "retune_recommended": drift.retune_recommended,
        }
    return payload


def render_drift_report(
    diff: ManifestDiff, drift: RetrievalDrift | None, generation_dir: Path | str | None
) -> str:
    """ASCII Markdown rendering of the drift report."""
    counts = diff.counts()
    lines = [
        "# Corpus refresh drift report",
        "",
        f"- generation: {generation_dir if generation_dir is not None else '(none)'}",
        "- documents: "
        + ", ".join(
            f"{counts[key]} {key}" for key in ("added", "modified", "deleted", "unchanged")
        ),
        "",
    ]
    if drift is None:
        lines += ["Retrieval validation skipped: no gold set was available for this refresh.", ""]
        return "\n".join(lines)
    lines += [
        f"Retrieval validation over {drift.n_items} gold items:",
        "",
        "| metric | old | new | delta |",
        "| --- | --- | --- | --- |",
        f"| recall@{drift.k} | {drift.old_recall:.3f} | {drift.new_recall:.3f} "
        f"| {drift.delta_recall:+.3f} |",
        f"| MRR | {drift.old_mrr:.3f} | {drift.new_mrr:.3f} | {drift.delta_mrr:+.3f} |",
        "",
    ]
    if drift.retune_recommended:
        lines += [
            "RE-TUNE RECOMMENDED: the recall/MRR delta crosses the threshold "
            f"({drift.threshold}). Re-run the tuner over the refreshed store "
            "(make tune-rag / joint-search).",
            "",
        ]
    else:
        lines += [
            f"No re-tune needed: retrieval deltas stay under the threshold ({drift.threshold}).",
            "",
        ]
    return "\n".join(lines)


def write_drift_report(
    out_dir: Path | str,
    diff: ManifestDiff,
    drift: RetrievalDrift | None,
    generation_dir: Path | str | None,
) -> tuple[Path, Path]:
    """Persist `drift.json` + `report.md` under `out_dir`; returns their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / DRIFT_JSON
    md_path = out_dir / DRIFT_REPORT_MD
    json_path.write_text(
        json.dumps(drift_payload(diff, drift, generation_dir), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(render_drift_report(diff, drift, generation_dir), encoding="utf-8")
    return json_path, md_path
