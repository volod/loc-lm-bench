"""Confidence-aware reading of the final board: per-case bootstrap, paired deltas, Pareto.

A scoreboard row is a mean over the held-out split. Ranking two models by that mean is exactly the
small-sample rank reversal this run exists to prevent, so every row is re-read from the per-case
`scores.jsonl` its final-split evaluation wrote:

- a percentile-bootstrap interval on quality AND on latency, drawn over shared item index sets so
  every row is resampled together (common random numbers);
- the paired delta against the declared incumbent row, carrying the win/loss/tie ledger, the
  calibrated sign-flip reading, and the borderline qualifier every other lane reports;
- the quality-versus-latency Pareto frontier over the same rows, because a run whose objectives
  were `quality,latency` owes an operator the tradeoff and not just the argmax.

Rows are paired on the item ids the bundles share, and a row whose bundle is unreadable is dropped
with its reason rather than compared against a different item set.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.artifacts.runs.bundle import read_case_rows
from llb.core.contracts.runs import EvalResult
from llb.eval.paired_cases import rows_by_item
from llb.rag.fusion_evidence.paired import PairedComparison, paired_comparison
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    Interval,
    bootstrap_index_sets,
    bootstrap_interval,
)

_LOG = logging.getLogger(__name__)

# The per-case columns a generator row is read on: the scored objective and the wall latency the
# quality/latency tradeoff is priced in.
QUALITY_COLUMN = "objective_score"
LATENCY_COLUMN = "latency_s"


@dataclass(frozen=True)
class BoardRow:
    """One (model, pick) row with its per-case vectors -- the unit everything below compares."""

    model: str
    pick: str
    backend: str
    quality: list[float]
    latency: list[float]

    @property
    def key(self) -> str:
        return f"{self.model}::{self.pick}"


def read_board_rows(
    entries: Sequence[Mapping[str, Any]],
    finals: Mapping[str, Mapping[str, EvalResult]],
) -> tuple[list[BoardRow], list[dict[str, str]]]:
    """Re-read every scoreboard entry's per-case rows; report the ones that cannot be read."""
    loaded: list[tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]] = []
    unreadable: list[dict[str, str]] = []
    for entry in entries:
        result = finals.get(str(entry.get("model")), {}).get(str(entry.get("pick")))
        scores = (result or {}).get("paths", {}).get("scores") if result else None
        if not scores or not Path(str(scores)).is_file():
            unreadable.append(
                {"row": f"{entry.get('model')}::{entry.get('pick')}", "reason": "no scores.jsonl"}
            )
            continue
        try:
            loaded.append((entry, rows_by_item(read_case_rows(Path(str(scores))))))
        except (OSError, ValueError) as exc:
            unreadable.append(
                {"row": f"{entry.get('model')}::{entry.get('pick')}", "reason": str(exc)}
            )
    if not loaded:
        return [], unreadable
    shared = sorted(set.intersection(*(set(by_item) for _entry, by_item in loaded)))
    rows = [
        BoardRow(
            model=str(entry.get("model")),
            pick=str(entry.get("pick")),
            backend=str(entry.get("backend") or ""),
            quality=[float(by_item[item].get(QUALITY_COLUMN, 0.0) or 0.0) for item in shared],
            latency=[float(by_item[item].get(LATENCY_COLUMN, 0.0) or 0.0) for item in shared],
        )
        for entry, by_item in loaded
    ]
    return rows, unreadable


def paired_deltas(rows: Sequence[BoardRow], candidate: str, baseline: str) -> list[float]:
    """Per-case candidate-minus-baseline quality, for the realized half of the power contract."""
    by_key = {row.key: row for row in rows}
    if candidate not in by_key or baseline not in by_key:
        return []
    return [
        value - other for value, other in zip(by_key[candidate].quality, by_key[baseline].quality)
    ]


def strongest_challenger(uncertainty: "BoardUncertainty") -> str | None:
    """The non-incumbent row with the largest paired delta -- the one the run could resolve on."""
    if not uncertainty.paired:
        return None
    return max(sorted(uncertainty.paired), key=lambda row: uncertainty.paired[row]["delta"]["mean"])


def baseline_row(rows: Sequence[BoardRow], incumbent: str | None) -> BoardRow | None:
    """The incumbent's best-quality row -- the one every candidate delta is measured against."""
    owned = [row for row in rows if row.model == incumbent]
    if not owned:
        return None
    return max(owned, key=lambda row: (sum(row.quality) / max(len(row.quality), 1), row.pick))


def pareto_frontier(rows: Sequence[BoardRow]) -> list[str]:
    """Row keys no other row beats on quality AND latency at once (higher / lower is better)."""
    means = {
        row.key: (
            sum(row.quality) / max(len(row.quality), 1),
            sum(row.latency) / max(len(row.latency), 1),
        )
        for row in rows
    }
    frontier: list[str] = []
    for row in rows:
        quality, latency = means[row.key]
        dominated = any(
            other.key != row.key
            and means[other.key][0] >= quality
            and means[other.key][1] <= latency
            and (means[other.key][0] > quality or means[other.key][1] < latency)
            for other in rows
        )
        if not dominated:
            frontier.append(row.key)
    return frontier


@dataclass(frozen=True)
class BoardUncertainty:
    """The confidence-aware board: intervals per row, paired deltas, and the frontier."""

    n_items: int
    confidence: float
    resamples: int
    baseline: str | None
    quality: dict[str, Interval]
    latency: dict[str, Interval]
    paired: dict[str, PairedComparison]
    frontier: list[str]
    unreadable: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_items": self.n_items,
            "confidence": self.confidence,
            "resamples": self.resamples,
            "baseline": self.baseline,
            "quality": {key: dict(value) for key, value in self.quality.items()},
            "latency": {key: dict(value) for key, value in self.latency.items()},
            "paired_vs_baseline": {key: dict(value) for key, value in self.paired.items()},
            "pareto_frontier": list(self.frontier),
            "unreadable_rows": list(self.unreadable),
        }


def read_uncertainty(
    rows: Sequence[BoardRow],
    *,
    incumbent: str | None,
    unreadable: Sequence[dict[str, str]] = (),
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> BoardUncertainty:
    """Bootstrap every row and pair the non-incumbent rows against the incumbent's best row."""
    n_items = len(rows[0].quality) if rows else 0
    index_sets = bootstrap_index_sets(n_items, resamples, seed)
    base = baseline_row(rows, incumbent)
    if incumbent and base is None:
        _LOG.warning("[joint-search] long-run incumbent %s was not scored on this board", incumbent)
    return BoardUncertainty(
        n_items=n_items,
        confidence=confidence,
        resamples=resamples,
        baseline=base.key if base else None,
        quality={row.key: bootstrap_interval(row.quality, index_sets, confidence) for row in rows},
        latency={row.key: bootstrap_interval(row.latency, index_sets, confidence) for row in rows},
        paired=(
            {
                row.key: paired_comparison(row.quality, base.quality, index_sets, confidence)
                for row in rows
                if base is not None and row.key != base.key
            }
            if base is not None
            else {}
        ),
        frontier=pareto_frontier(rows),
        unreadable=list(unreadable),
    )


__all__ = [
    "LATENCY_COLUMN",
    "QUALITY_COLUMN",
    "BoardRow",
    "BoardUncertainty",
    "baseline_row",
    "paired_deltas",
    "pareto_frontier",
    "read_board_rows",
    "read_uncertainty",
    "strongest_challenger",
]
