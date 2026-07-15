"""Lost-in-the-middle probe: gold-chunk position sensitivity at fixed k (rerank-context-order).

`llb probe-context-position` measures how much a model's RAG answer quality depends on WHERE
the gold evidence sits in the prompt: for every gold item whose gold chunk is retrievable, the
probe builds a fixed-k context of REAL retrieved distractors and lays the gold chunk at the
head, middle, and tail, asking the same question three times through the standard RAG chat
prompt. Per-position mean objective score with bootstrap CIs then names the context-order
recommendation for the probed model (`rank` when the head wins, `reverse_rank` when the tail
wins) -- measured evidence for `RunConfig.context_order` instead of a lore default.

Pure core: `build_probe_cases` / `assemble_context_chunks` / `summarize` take injected stores
and a `chat` callable, so the whole probe is unit-testable with fakes -- no backend, no GPU.
Artifacts land under `$DATA_DIR/context-position/<timestamp>/{report.md,cases.jsonl}`.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llb.core.contracts.common import ChatMessage, JsonObject
from llb.core.contracts.rag import ChunkRecord
from llb.eval import common as eval_common
from llb.eval.graph import build_messages
from llb.executor.cases import spans_as_dicts
from llb.goldset.schema import GoldItem
from llb.rag.retrieval import chunk_hits_any
from llb.scoring import correctness
from llb.scoring.leaderboard import bootstrap_mean_ci

_LOG = logging.getLogger(__name__)

PROBE_METHOD = "context-position"

POSITION_HEAD = "head"
POSITION_MIDDLE = "middle"
POSITION_TAIL = "tail"
POSITIONS = (POSITION_HEAD, POSITION_MIDDLE, POSITION_TAIL)

# Candidate depth scanned for the gold chunk + distractors (per item, one retrieval).
DEFAULT_CANDIDATE_DEPTH = 50
# Minimum probe k: below three chunks head/middle/tail collapse into the same index.
MIN_PROBE_K = 3

# (messages) -> (answer text, transport error or None); production binds a BackendLauncher.
ProbeChat = Callable[[list[ChatMessage]], tuple[str, str | None]]

SKIP_GOLD_NOT_RETRIEVED = "gold_not_retrieved"
SKIP_TOO_FEW_DISTRACTORS = "too_few_distractors"


@dataclass(slots=True)
class ProbeCase:
    """One probe-ready gold item: its gold chunk plus k-1 real distractors (rank order)."""

    item: GoldItem
    gold_chunk: ChunkRecord
    distractors: list[ChunkRecord]


@dataclass(slots=True)
class PositionSummary:
    """Per-position aggregate over the shared probe item set."""

    position: str
    n: int
    mean_score: float
    ci: tuple[float, float] | None


@dataclass(slots=True)
class ProbeReport:
    model: str
    backend: str
    k: int
    n_items: int
    skipped: dict[str, int]
    positions: list[PositionSummary]
    recommendation: str
    recommendation_note: str
    rows: list[JsonObject] = field(default_factory=list)


def position_index(position: str, k: int) -> int:
    """0-based prompt slot of the gold chunk for a position label at context size k."""
    if position == POSITION_HEAD:
        return 0
    if position == POSITION_MIDDLE:
        return (k - 1) // 2
    if position == POSITION_TAIL:
        return k - 1
    raise ValueError(f"unknown position: {position!r}; choose from {POSITIONS}")


def build_probe_cases(
    items: list[GoldItem],
    store: Any,
    k: int,
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
) -> tuple[list[ProbeCase], dict[str, int]]:
    """Select probe-ready items: gold chunk retrievable + k-1 distractors available.

    One retrieval of `candidate_depth` per item supplies both the gold chunk (the first
    candidate overlapping a gold span) and the distractors (the best-ranked non-gold
    candidates), so every probe context is built from chunks the retriever actually returns
    for that question -- real distractors, not synthetic filler. Items without a retrievable
    gold chunk or with too few distractors are counted per skip reason, never invented.
    """
    if k < MIN_PROBE_K:
        raise ValueError(f"probe k must be >= {MIN_PROBE_K} (head/middle/tail must differ)")
    cases: list[ProbeCase] = []
    skipped = {SKIP_GOLD_NOT_RETRIEVED: 0, SKIP_TOO_FEW_DISTRACTORS: 0}
    depth = max(candidate_depth, k)
    for item in items:
        spans = spans_as_dicts(item)
        candidates = store.retrieve(item.question, depth)
        gold = next((c for c in candidates if chunk_hits_any(c, spans)), None)
        if gold is None:
            skipped[SKIP_GOLD_NOT_RETRIEVED] += 1
            continue
        distractors = [c for c in candidates if not chunk_hits_any(c, spans)][: k - 1]
        if len(distractors) < k - 1:
            skipped[SKIP_TOO_FEW_DISTRACTORS] += 1
            continue
        cases.append(ProbeCase(item=item, gold_chunk=gold, distractors=distractors))
    return cases, skipped


def assemble_context_chunks(case: ProbeCase, position: str, k: int) -> list[ChunkRecord]:
    """The k-chunk context with the gold chunk at the requested slot (distractors keep rank
    order around it). Pure list surgery; chunk contents are never altered."""
    chunks = list(case.distractors)
    chunks.insert(position_index(position, k), case.gold_chunk)
    return chunks


def _score_answer(item: GoldItem, answer: str, error: str | None) -> tuple[str, float]:
    status = eval_common.classify_response(answer, error)
    score = correctness.answer_correctness(answer or "", item.reference_answer)["score"]
    return status, float(score)


def summarize(rows: list[JsonObject]) -> list[PositionSummary]:
    """Per-position mean objective score + bootstrap CI over the shared probe item set."""
    out: list[PositionSummary] = []
    for position in POSITIONS:
        scores = [float(r["objective_score"]) for r in rows if r["position"] == position]
        out.append(
            PositionSummary(
                position=position,
                n=len(scores),
                mean_score=round(sum(scores) / len(scores), 4) if scores else 0.0,
                ci=bootstrap_mean_ci(scores),
            )
        )
    return out


def recommend_order(positions: list[PositionSummary]) -> tuple[str, str]:
    """Name the context-order policy the measured position sensitivity supports.

    Head-at-least-tail keeps the best-first default (`rank`); a tail win recommends
    `reverse_rank`. Overlapping head/tail CIs are flagged as an unresolved preference (the
    recommendation still names the higher mean, honestly qualified).
    """
    by_pos = {p.position: p for p in positions}
    head, tail = by_pos[POSITION_HEAD], by_pos[POSITION_TAIL]
    order = (
        eval_common.ORDER_RANK
        if head.mean_score >= tail.mean_score
        else (eval_common.ORDER_REVERSE_RANK)
    )
    note = (
        f"head {head.mean_score:.3f} vs tail {tail.mean_score:.3f}"
        f" (middle {by_pos[POSITION_MIDDLE].mean_score:.3f})"
    )
    if head.ci and tail.ci and head.ci[0] <= tail.ci[1] and tail.ci[0] <= head.ci[1]:
        note += "; head/tail CIs overlap -- position sensitivity not resolved at this n"
    return order, note


def run_probe(
    items: list[GoldItem],
    store: Any,
    chat: ProbeChat,
    *,
    model: str,
    backend: str,
    k: int,
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
) -> ProbeReport:
    """Execute the full probe: build cases, ask each question at every gold position, score,
    aggregate per position, and derive the ordering recommendation."""
    cases, skipped = build_probe_cases(items, store, k, candidate_depth)
    rows: list[JsonObject] = []
    for case in cases:
        for position in POSITIONS:
            chunks = assemble_context_chunks(case, position, k)
            answer, error = chat(
                build_messages(case.item.question, eval_common.format_context(chunks))
            )
            status, score = _score_answer(case.item, answer, error)
            rows.append(
                {
                    "item_id": case.item.id,
                    "position": position,
                    "gold_index": position_index(position, k),
                    "k": k,
                    "status": status,
                    "objective_score": round(score, 4),
                    "answer_preview": (answer or "")[:280],
                }
            )
    positions = summarize(rows)
    recommendation, note = recommend_order(positions)
    return ProbeReport(
        model=model,
        backend=backend,
        k=k,
        n_items=len(cases),
        skipped={reason: n for reason, n in skipped.items() if n},
        positions=positions,
        recommendation=recommendation,
        recommendation_note=note,
        rows=rows,
    )
