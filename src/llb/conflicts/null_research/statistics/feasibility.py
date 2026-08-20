"""How many independent control units an operator-affordable semantic tail would require.

A null-calibrated threshold is only useful at the tail an operator can actually adjudicate: a
candidate cap of `N` rows over a corpus with `P` comparable pairs is the per-pair tail
`alpha = N / P`. Estimating that tail needs enough INDEPENDENT control observations for the tail
itself to be populated, which is what turns the requested operating point into a control-bank size.
This lane computes that requirement before any candidate is fitted, so a lane that cannot possibly
resolve its own operating point is reported as infeasible rather than as a fitted number.
"""

import math

from llb.conflicts.null_research.evaluation import MIN_TAIL_OBSERVATIONS
from llb.conflicts.null_research.geometry import CorpusGeometry
from llb.core.contracts.common import JsonObject


def _dataset_feasibility(
    corpus: CorpusGeometry,
    available_units: int,
    *,
    candidate_cap: int,
    nominal_fpr: float,
) -> JsonObject:
    pairs = len(corpus.observed_similarities)
    if pairs <= 0:
        raise ValueError(f"{corpus.name} has no comparable cross-document pairs")
    if candidate_cap < 1:
        raise ValueError("the candidate cap must be positive")
    operating_alpha = min(1.0, candidate_cap / pairs)
    required = math.ceil(MIN_TAIL_OBSERVATIONS / operating_alpha)
    return {
        "comparable_chunk_pairs": pairs,
        "candidate_cap": candidate_cap,
        "operating_tail_alpha": float(f"{operating_alpha:.6g}"),
        "required_independent_units": required,
        "available_independent_units": available_units,
        "unit_deficit_factor": round(required / available_units, 3) if available_units else None,
        "rows_selected_at_nominal_fpr": round(nominal_fpr * pairs, 1),
        "nominal_fpr_overshoot_factor": round(nominal_fpr * pairs / candidate_cap, 1),
        "feasible": available_units >= required,
    }


def operating_point_feasibility(
    corpora: dict[str, CorpusGeometry],
    available_units: dict[str, int],
    *,
    candidate_cap: int,
    nominal_fpr: float,
) -> JsonObject:
    """Report the control-bank size each corpus's affordable candidate tail would demand."""
    datasets = {
        name: _dataset_feasibility(
            corpus,
            available_units[name],
            candidate_cap=candidate_cap,
            nominal_fpr=nominal_fpr,
        )
        for name, corpus in corpora.items()
    }
    return {
        "min_tail_observations": MIN_TAIL_OBSERVATIONS,
        "candidate_cap": candidate_cap,
        "nominal_fpr": nominal_fpr,
        "datasets": datasets,
        "feasible": all(bool(payload["feasible"]) for payload in datasets.values()),
        "largest_required_units": max(
            int(payload["required_independent_units"]) for payload in datasets.values()
        ),
    }
