"""Optuna fit for a monotone chance-floor knowledge-decay curve."""

import math
from dataclasses import dataclass
from typing import Any

from llb.bench.knowledge_cutoff.score import MCQ_CHANCE_FLOOR, CutoffSummary

DEFAULT_TRIALS = 200
DEFAULT_SEED = 42
MIN_FIT_MONTHS = 4
MIN_FIT_EVENTS = 12


@dataclass(frozen=True, slots=True)
class DecayFit:
    status: str
    effective_cutoff: str | None
    cutoff_ordinal: float | None
    scale_months: float | None
    ceiling: float | None
    chance_floor: float
    negative_log_likelihood: float | None
    trials: int
    seed: int
    fitted_curve: list[dict[str, float | str]]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "effective_cutoff": self.effective_cutoff,
            "cutoff_ordinal": self.cutoff_ordinal,
            "scale_months": self.scale_months,
            "ceiling": self.ceiling,
            "chance_floor": self.chance_floor,
            "negative_log_likelihood": self.negative_log_likelihood,
            "trials": self.trials,
            "seed": self.seed,
            "fitted_curve": self.fitted_curve,
        }


def month_ordinal(month: str) -> int:
    year_text, month_text = month.split("-", 1)
    return int(year_text) * 12 + int(month_text) - 1


def ordinal_month(ordinal: float) -> str:
    rounded = round(ordinal)
    year, month_index = divmod(rounded, 12)
    return f"{year:04d}-{month_index + 1:02d}"


def _probability(x: float, cutoff: float, scale: float, ceiling: float) -> float:
    exponent = max(-60.0, min(60.0, (x - cutoff) / scale))
    return MCQ_CHANCE_FLOOR + (ceiling - MCQ_CHANCE_FLOOR) / (1.0 + math.exp(exponent))


def _insufficient(summary: CutoffSummary, trials: int, seed: int) -> DecayFit:
    curve: list[dict[str, float | str]] = [
        {"month": point.month, "predicted_accuracy": point.accuracy} for point in summary.curve
    ]
    return DecayFit(
        "insufficient_data",
        None,
        None,
        None,
        None,
        MCQ_CHANCE_FLOOR,
        None,
        trials,
        seed,
        curve,
    )


def fit_decay(
    summary: CutoffSummary,
    *,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
) -> DecayFit:
    """Fit ceiling / midpoint / scale by minimizing grouped binomial log loss."""
    if trials < 1:
        raise ValueError("trials must be at least 1")
    n_events = sum(point.n for point in summary.curve)
    if len(summary.curve) < MIN_FIT_MONTHS or n_events < MIN_FIT_EVENTS:
        return _insufficient(summary, trials, seed)
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("knowledge-cutoff fitting needs Optuna; install `.[cutoff]`") from exc

    ordinals = [month_ordinal(point.month) for point in summary.curve]
    start = min(ordinals)
    xs = [value - start for value in ordinals]
    span = max(xs) - min(xs)
    padding = max(3.0, span * 0.25)

    def objective(trial: Any) -> float:
        cutoff = trial.suggest_float("cutoff", -padding, span + padding)
        scale = trial.suggest_float("scale_months", 0.25, max(1.0, span), log=True)
        ceiling = trial.suggest_float("ceiling", MCQ_CHANCE_FLOOR + 0.01, 1.0)
        nll = 0.0
        for x, point in zip(xs, summary.curve, strict=True):
            probability = min(1.0 - 1e-9, max(1e-9, _probability(x, cutoff, scale, ceiling)))
            nll -= point.correct * math.log(probability)
            nll -= (point.n - point.correct) * math.log(1.0 - probability)
        return nll / n_events

    sampler = optuna.samplers.TPESampler(seed=seed)
    previous_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
    finally:
        optuna.logging.set_verbosity(previous_verbosity)
    params = study.best_params
    cutoff_ordinal = start + float(params["cutoff"])
    cutoff_label = ordinal_month(cutoff_ordinal)
    first_month = summary.curve[0].month
    last_month = summary.curve[-1].month
    if cutoff_ordinal < month_ordinal(first_month) - 0.5:
        cutoff_label = f"<{first_month}"
    elif cutoff_ordinal > month_ordinal(last_month) + 0.5:
        cutoff_label = f">={last_month}"
    fitted: list[dict[str, float | str]] = [
        {
            "month": point.month,
            "predicted_accuracy": _probability(
                x,
                float(params["cutoff"]),
                float(params["scale_months"]),
                float(params["ceiling"]),
            ),
        }
        for x, point in zip(xs, summary.curve, strict=True)
    ]
    return DecayFit(
        status="ok",
        effective_cutoff=cutoff_label,
        cutoff_ordinal=cutoff_ordinal,
        scale_months=float(params["scale_months"]),
        ceiling=float(params["ceiling"]),
        chance_floor=MCQ_CHANCE_FLOOR,
        negative_log_likelihood=float(study.best_value),
        trials=trials,
        seed=seed,
        fitted_curve=fitted,
    )
