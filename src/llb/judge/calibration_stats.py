"""Focused calibration stats implementation."""

import random
from llb.core.contracts import CalibrationResult

DEFAULT_THRESHOLD = 0.6


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den_a = sum((x - mean_a) ** 2 for x in a) ** 0.5
    den_b = sum((y - mean_b) ** 2 for y in b) ** 0.5
    if den_a == 0 or den_b == 0:
        return 0.0
    return float(num / (den_a * den_b))


def spearman_rho(human: list[float], judge: list[float]) -> float:
    if len(human) != len(judge):
        raise ValueError("human and judge ratings must be the same length")
    if len(human) < 2:
        raise ValueError("need >= 2 paired ratings")
    return _pearson(_average_ranks(human), _average_ranks(judge))


def bootstrap_ci(
    human: list[float],
    judge: list[float],
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = random.Random(seed)
    m = len(human)
    rhos: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(m) for _ in range(m)]
        sample_h = [human[i] for i in idx]
        sample_j = [judge[i] for i in idx]
        try:
            rhos.append(spearman_rho(sample_h, sample_j))
        except ValueError:
            continue
    if not rhos:
        return (0.0, 0.0)
    rhos.sort()
    lo = rhos[int((alpha / 2) * len(rhos))]
    hi = rhos[min(len(rhos) - 1, int((1 - alpha / 2) * len(rhos)))]
    return (lo, hi)


def calibrate(
    human: list[float], judge: list[float], threshold: float = DEFAULT_THRESHOLD
) -> CalibrationResult:
    rho = spearman_rho(human, judge)
    lo, hi = bootstrap_ci(human, judge)
    return {
        "rho": rho,
        "ci_low": lo,
        "ci_high": hi,
        "n": len(human),
        "threshold": threshold,
        "trusted": rho >= threshold,
    }
