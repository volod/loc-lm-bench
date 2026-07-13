"""Transparent monthly evidence and control diagnostics for cutoff fitting."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from llb.bench.knowledge_cutoff.data import CutoffEvent

DEFAULT_THRESHOLD = 0.5
MCQ_CHANCE_FLOOR = 0.25
SUSTAINED_MONTHS = 2


@dataclass(frozen=True, slots=True)
class MonthlyPoint:
    month: str
    n: int
    correct: int
    incorrect: int
    abstain: int
    accuracy: float
    chance_adjusted_accuracy: float

    def to_dict(self) -> dict[str, object]:
        return {
            "month": self.month,
            "n": self.n,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "abstain": self.abstain,
            "accuracy": self.accuracy,
            "chance_adjusted_accuracy": self.chance_adjusted_accuracy,
        }


@dataclass(frozen=True, slots=True)
class CutoffSummary:
    curve: list[MonthlyPoint]
    threshold: float
    last_above: str | None
    first_sustained_below: str | None
    eligible_accuracy: float
    parse_rate: float
    controls: dict[str, float | int | None]

    def to_dict(self) -> dict[str, object]:
        return {
            "curve": [point.to_dict() for point in self.curve],
            "threshold": self.threshold,
            "last_above": self.last_above,
            "first_sustained_below": self.first_sustained_below,
            "eligible_accuracy": self.eligible_accuracy,
            "parse_rate": self.parse_rate,
            "controls": self.controls,
        }


def case_row(
    event: CutoffEvent,
    *,
    response: str,
    selected: str | None,
    expected: str,
    choice_order: tuple[str, ...],
) -> dict[str, Any]:
    if selected is None:
        label = "abstain"
    else:
        label = "correct" if selected == expected else "incorrect"
    return {
        "item_id": event.id,
        "month": event.month,
        "category": event.category,
        "predictability": event.predictability,
        "region": event.region,
        "counts_for_curve": event.counts_for_curve,
        "label": label,
        "selected": selected,
        "expected": expected,
        "choice_order": list(choice_order),
        "objective_score": 1.0 if label == "correct" else 0.0,
        "response": response,
    }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(rows: list[dict[str, Any]], threshold: float = DEFAULT_THRESHOLD) -> CutoffSummary:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    monthly: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "incorrect": 0, "abstain": 0}
    )
    parsed = 0
    eligible = 0
    eligible_correct = 0
    alive_n = alive_correct = fake_n = fake_rejected = fake_confirmed = 0
    for row in rows:
        label = str(row["label"])
        if label != "abstain":
            parsed += 1
        category = row["category"]
        if category == "control_alive":
            alive_n += 1
            alive_correct += int(label == "correct")
        elif category == "fake_event":
            fake_n += 1
            fake_rejected += int(label == "correct")
            fake_confirmed += int(label == "incorrect")
        elif row["counts_for_curve"]:
            bucket = monthly[str(row["month"])]
            bucket[label] += 1
            eligible += 1
            eligible_correct += int(label == "correct")

    curve: list[MonthlyPoint] = []
    for month, counts in sorted(monthly.items()):
        n = sum(counts.values())
        accuracy = counts["correct"] / n
        adjusted = max(0.0, min(1.0, (accuracy - MCQ_CHANCE_FLOOR) / (1 - MCQ_CHANCE_FLOOR)))
        curve.append(
            MonthlyPoint(month, n, **counts, accuracy=accuracy, chance_adjusted_accuracy=adjusted)
        )

    above = [point.month for point in curve if point.accuracy >= threshold]
    sustained = next(
        (
            curve[index].month
            for index in range(len(curve) - SUSTAINED_MONTHS + 1)
            if all(point.accuracy < threshold for point in curve[index:])
        ),
        None,
    )
    controls: dict[str, float | int | None] = {
        "living_person_n": alive_n,
        "living_person_accuracy": _safe_rate(alive_correct, alive_n),
        "fake_event_n": fake_n,
        "fake_event_rejection_rate": _safe_rate(fake_rejected, fake_n),
        "fake_event_confabulation_rate": _safe_rate(fake_confirmed, fake_n),
    }
    return CutoffSummary(
        curve=curve,
        threshold=threshold,
        last_above=above[-1] if above else None,
        first_sustained_below=sustained,
        eligible_accuracy=eligible_correct / eligible if eligible else 0.0,
        parse_rate=parsed / len(rows) if rows else 0.0,
        controls=controls,
    )
