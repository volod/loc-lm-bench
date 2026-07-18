"""Multi-objective vocabulary: parse goals, normalize metrics, Pareto picks."""

from dataclasses import asdict, dataclass
from typing import Any, Sequence

OBJECTIVE_QUALITY = "quality"
OBJECTIVE_LATENCY = "latency"
OBJECTIVE_COST = "cost"
KNOWN_OBJECTIVES = (OBJECTIVE_QUALITY, OBJECTIVE_LATENCY, OBJECTIVE_COST)

# Accuracy floor for the "cheapest within floor" pick: fraction of the front's best quality.
DEFAULT_ACCURACY_FLOOR_RATIO = 0.9


@dataclass(frozen=True)
class TrialMetrics:
    """Per-trial measurements returned by a multi-objective evaluate hook."""

    quality: float
    latency_s: float = 0.0
    cost_usd: float = 0.0
    throughput: float = 0.0


@dataclass(frozen=True)
class ParetoPoint:
    """One non-dominated trial on the Pareto front."""

    number: int
    quality: float
    latency_s: float
    cost_usd: float
    throughput: float
    overrides: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoalPick:
    """A named operator-facing selection from the Pareto front."""

    goal: str
    point: ParetoPoint

    def to_dict(self) -> dict[str, Any]:
        return {"goal": self.goal, "point": self.point.to_dict()}


def parse_objectives(raw: str | Sequence[str]) -> tuple[str, ...]:
    """Parse `quality,latency[,cost]` into a validated ordered tuple (at least two goals)."""
    parts = (
        [p.strip().lower() for p in raw.split(",")]
        if isinstance(raw, str)
        else [str(p).strip().lower() for p in raw]
    )
    parts = [p for p in parts if p]
    if len(parts) < 2:
        raise ValueError("multi-objective mode needs at least two goals (e.g. quality,latency)")
    unknown = [p for p in parts if p not in KNOWN_OBJECTIVES]
    if unknown:
        raise ValueError(f"unknown objectives {unknown}; expected subset of {KNOWN_OBJECTIVES}")
    if len(set(parts)) != len(parts):
        raise ValueError(f"duplicate objectives in {parts}")
    if OBJECTIVE_QUALITY not in parts:
        raise ValueError("objectives must include 'quality'")
    return tuple(parts)


def study_directions(objectives: Sequence[str]) -> list[str]:
    """Optuna directions: maximize quality, minimize latency and cost."""
    mapping = {
        OBJECTIVE_QUALITY: "maximize",
        OBJECTIVE_LATENCY: "minimize",
        OBJECTIVE_COST: "minimize",
    }
    return [mapping[name] for name in objectives]


def metrics_vector(metrics: TrialMetrics, objectives: Sequence[str]) -> tuple[float, ...]:
    """Map TrialMetrics onto the study objective vector in declaration order."""
    values = {
        OBJECTIVE_QUALITY: metrics.quality,
        OBJECTIVE_LATENCY: metrics.latency_s,
        OBJECTIVE_COST: metrics.cost_usd,
    }
    return tuple(values[name] for name in objectives)


def normalize_outcome(outcome: Any, *, wall_latency_s: float = 0.0) -> TrialMetrics:
    """Accept TrialMetrics, quality float, (quality, throughput), or (quality, latency, cost)."""
    if isinstance(outcome, TrialMetrics):
        return outcome
    if isinstance(outcome, tuple):
        if len(outcome) == 2:
            return TrialMetrics(
                quality=float(outcome[0]),
                throughput=float(outcome[1]),
                latency_s=wall_latency_s,
            )
        if len(outcome) >= 3:
            return TrialMetrics(
                quality=float(outcome[0]),
                latency_s=float(outcome[1]),
                cost_usd=float(outcome[2]),
                throughput=float(outcome[3]) if len(outcome) > 3 else 0.0,
            )
    return TrialMetrics(quality=float(outcome), latency_s=wall_latency_s)


def extract_pareto_front(trials: Sequence[Any], objectives: Sequence[str]) -> list[ParetoPoint]:
    """Build ParetoPoints from COMPLETE Optuna trials (values already non-dominated by study)."""
    points: list[ParetoPoint] = []
    for trial in trials:
        values = trial.values
        if values is None:
            continue
        by_name = dict(zip(objectives, values))
        points.append(
            ParetoPoint(
                number=int(trial.number),
                quality=float(by_name.get(OBJECTIVE_QUALITY, 0.0)),
                latency_s=float(
                    by_name.get(OBJECTIVE_LATENCY, trial.user_attrs.get("latency_s", 0.0))
                ),
                cost_usd=float(by_name.get(OBJECTIVE_COST, trial.user_attrs.get("cost_usd", 0.0))),
                throughput=float(trial.user_attrs.get("throughput", 0.0)),
                overrides=dict(trial.user_attrs.get("overrides") or {}),
            )
        )
    return points


def select_goal_picks(
    front: Sequence[ParetoPoint],
    *,
    accuracy_floor: float | None = None,
    accuracy_floor_ratio: float = DEFAULT_ACCURACY_FLOOR_RATIO,
    include_cost: bool = False,
) -> list[GoalPick]:
    """Named picks: best quality, best quality-per-second, optional cheapest within floor."""
    if not front:
        return []
    picks = [
        GoalPick("best_quality", max(front, key=lambda p: (p.quality, -p.latency_s))),
        GoalPick(
            "best_quality_per_second",
            max(front, key=lambda p: (p.quality / max(p.latency_s, 1e-9), p.quality)),
        ),
    ]
    if include_cost:
        best_q = max(p.quality for p in front)
        floor = accuracy_floor if accuracy_floor is not None else best_q * accuracy_floor_ratio
        eligible = [p for p in front if p.quality >= floor]
        if eligible:
            picks.append(GoalPick("cheapest_within_floor", min(eligible, key=lambda p: p.cost_usd)))
    return picks


def nondominated_trials(trials: Sequence[Any], objectives: Sequence[str]) -> list[Any]:
    """Return trials whose value vectors are non-dominated under the study directions."""
    directions = study_directions(objectives)
    kept: list[Any] = []
    for trial in trials:
        if trial.values is None:
            continue
        dominated = False
        for other in trials:
            if other is trial or other.values is None:
                continue
            if _dominates(other.values, trial.values, directions):
                dominated = True
                break
        if not dominated:
            kept.append(trial)
    return kept


def _dominates(left: Sequence[float], right: Sequence[float], directions: Sequence[str]) -> bool:
    """True when ``left`` is at least as good on every objective and better on one."""
    not_worse = True
    strictly_better = False
    for lval, rval, direction in zip(left, right, directions):
        if direction == "maximize":
            if lval < rval:
                not_worse = False
                break
            if lval > rval:
                strictly_better = True
        else:
            if lval > rval:
                not_worse = False
                break
            if lval < rval:
                strictly_better = True
    return not_worse and strictly_better
