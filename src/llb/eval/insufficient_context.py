"""Insufficient-context probe: does the model abstain when its gold evidence is removed?

`groundedness-citation-metrics`. A grounded RAG system should DECLINE when the retrieved context
does not contain the answer, not fabricate one. This probe measures that directly: for a seeded
sample of gold items, it re-runs the question with every chunk overlapping the item's gold spans
EXCLUDED from retrieval (through the shipped chunk-metadata filter seam), so the supporting evidence
is gone. Correct behavior is an explicit abstention (`llb.eval.common.is_abstention`) and the score
is abstention accuracy -- the share of probes on which the model abstained.

Probe cases are scored on their OWN axis and NEVER enter the plain correctness aggregates (they live
in `probes.jsonl`, not `scores.jsonl`), so an abstention is never mistaken for a wrong answer or a
refusal on a scoreable case.

Pure core: `sample_probe_items` / `gold_excluding_filter` / `run_insufficient_context_probe` take an
injected store and a `chat` callable, so the whole probe is unit-testable with fakes -- no backend.
"""

import json
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llb.backends.base import ERR_BACKEND, ERR_TIMEOUT
from llb.core.contracts import ChatMessage, ChunkRecord, JsonObject, SourceSpanRecord
from llb.eval import common as eval_common
from llb.eval.graph import build_messages
from llb.executor.cases import spans_as_dicts
from llb.goldset.schema import GoldItem
from llb.rag.filters import ChunkFilter
from llb.rag.retrieval import chunk_hits_any
from llb.scoring.leaderboard import bootstrap_mean_ci

_LOG = logging.getLogger(__name__)

PROBE_METHOD = "insufficient-context"
_TRANSPORT_ERRORS = (ERR_TIMEOUT, ERR_BACKEND)

# (messages) -> (answer text, transport error or None); production binds a BackendLauncher.
ProbeChat = Callable[[list[ChatMessage]], tuple[str, str | None]]


@dataclass(slots=True)
class InsufficientContextReport:
    """Abstention-accuracy summary over a sampled, gold-excluded probe set."""

    model: str
    backend: str
    k: int
    n_probes: int
    n_abstained: int
    n_errors: int
    abstention_accuracy: float
    ci: tuple[float, float] | None
    rows: list[JsonObject] = field(default_factory=list)


def gold_excluding_filter(spans: list[SourceSpanRecord]) -> ChunkFilter:
    """A chunk filter that REJECTS any chunk overlapping a gold span (the evidence to remove)."""

    def accept(chunk: ChunkRecord) -> bool:
        return not chunk_hits_any(chunk, spans)

    return accept


def sample_probe_items(items: list[GoldItem], n: int, seed: int) -> list[GoldItem]:
    """A deterministic seeded sample of `n` items (id-sorted for stable row order)."""
    ordered = sorted(items, key=lambda it: it.id)
    if n >= len(ordered):
        return ordered
    chosen = random.Random(seed).sample(ordered, n)
    return sorted(chosen, key=lambda it: it.id)


def run_insufficient_context_probe(
    items: list[GoldItem],
    store: Any,
    chat: ProbeChat,
    *,
    model: str,
    backend: str,
    k: int,
    n: int,
    seed: int = 0,
    cited: bool = False,
    context_order: str = eval_common.ORDER_RANK,
) -> InsufficientContextReport:
    """Run the probe over a seeded sample and score abstention accuracy.

    Each probe retrieves `k` chunks with the item's gold spans excluded, asks the question, and
    records whether the model abstained. Transport errors are excluded from the accuracy denominator
    (they measure nothing), not counted as failed abstentions.
    """
    probes = sample_probe_items(items, n, seed)
    rows: list[JsonObject] = []
    for item in probes:
        spans = spans_as_dicts(item)
        chunks = store.retrieve(item.question, k, chunk_filter=gold_excluding_filter(spans))
        answer, error = chat(
            build_messages(
                item.question, eval_common.format_context(chunks, context_order), cited=cited
            )
        )
        status = eval_common.classify_response(answer, error)
        abstained = error is None and eval_common.is_abstention(answer or "")
        rows.append(
            {
                "item_id": item.id,
                "probe": True,
                "n_context": len(chunks),
                "status": status,
                "abstained": abstained,
                "answer_preview": (answer or "")[:280],
            }
        )
    return _summarize(rows, model=model, backend=backend, k=k)


def _summarize(
    rows: list[JsonObject], *, model: str, backend: str, k: int
) -> InsufficientContextReport:
    scored = [r for r in rows if r["status"] not in _TRANSPORT_ERRORS]
    outcomes = [1.0 if r["abstained"] else 0.0 for r in scored]
    n_abstained = sum(1 for r in scored if r["abstained"])
    accuracy = (n_abstained / len(scored)) if scored else 0.0
    return InsufficientContextReport(
        model=model,
        backend=backend,
        k=k,
        n_probes=len(rows),
        n_abstained=n_abstained,
        n_errors=len(rows) - len(scored),
        abstention_accuracy=accuracy,
        ci=bootstrap_mean_ci(outcomes),
        rows=rows,
    )


def render_report(report: InsufficientContextReport) -> str:
    """ASCII Markdown report (AGENTS.md: no box-drawing, no emojis)."""
    ci = f"[{report.ci[0]:.3f}, {report.ci[1]:.3f}]" if report.ci else "n/a"
    errors = f", {report.n_errors} transport errors excluded" if report.n_errors else ""
    return "\n".join(
        [
            "# Insufficient-context probe (abstention accuracy)",
            "",
            f"- model: `{report.model}` (backend: {report.backend})",
            f"- gold evidence excluded from retrieval; k={report.k}",
            f"- probes: {report.n_probes}{errors}",
            f"- abstained: {report.n_abstained}",
            f"- abstention accuracy: **{report.abstention_accuracy:.3f}** (95% CI {ci})",
            "",
            "Correct behavior on a probe is an explicit abstention (refusal or"
            " insufficient-context signal); probe cases never enter the correctness aggregates.",
            "",
        ]
    )


def write_probe(report: InsufficientContextReport, out_dir: Path) -> dict[str, str]:
    """Persist `insufficient_context_report.md` + `probes.jsonl`; returns their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "insufficient_context_report.md"
    probes_path = out_dir / "probes.jsonl"
    report_path.write_text(render_report(report), encoding="utf-8")
    with probes_path.open("w", encoding="utf-8") as fh:
        for row in report.rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"insufficient_context_report": str(report_path), "probes": str(probes_path)}
