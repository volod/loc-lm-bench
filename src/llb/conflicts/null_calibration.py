"""Decide the semantic tier's operating cosine from the corpus's own null distribution.

The measurement lives in `null_distribution` (the record) and `null_sampling` (how it is
obtained); this module owns only the precedence between the operator's knobs.
"""

import logging

from llb.conflicts.null_distribution import NullDistribution

_LOG = logging.getLogger(__name__)


def resolve_cos_threshold(
    *,
    explicit: float | None,
    quantile: float | None,
    default: float,
    distribution: NullDistribution | None,
    max_candidate_pairs: int | None = None,
) -> tuple[float, str, float | None]:
    """Pick the operating threshold, say where it came from, and report the quantile used.

    Precedence is explicit > cos_quantile > max_candidate_pairs > default: an operator who names a
    cosine has usually swept for it on this corpus and must not have it silently overridden by an
    estimate, and one who names a raw quantile is deliberately reaching past the budget knob.
    """
    if explicit is not None:
        if quantile is not None or max_candidate_pairs is not None:
            _LOG.warning(
                "[conflicts] --cos-threshold %.3f given alongside a calibration knob; "
                "the explicit threshold wins and the calibration is recorded only",
                explicit,
            )
        return explicit, "explicit", None
    if distribution is not None:
        resolved_quantile = quantile
        if resolved_quantile is None and max_candidate_pairs is not None:
            resolved_quantile = distribution.quantile_for_top_n(max_candidate_pairs)
        elif resolved_quantile is not None and max_candidate_pairs is not None:
            _LOG.warning(
                "[conflicts] --cos-quantile %.6f overrides --max-candidate-pairs %d",
                resolved_quantile,
                max_candidate_pairs,
            )
        if resolved_quantile is not None:
            if (
                not distribution.exhaustive
                and resolved_quantile > distribution.resolvable_quantile()
            ):
                _LOG.warning(
                    "[conflicts] quantile %.8f is finer than the %d-pair sample can resolve "
                    "(floor %.8f); the threshold pins to the sample maximum and UNDERSTATES the "
                    "tail -- raise --null-sample-pairs for a trustworthy calibration",
                    resolved_quantile,
                    distribution.n_pairs,
                    distribution.resolvable_quantile(),
                )
            calibrated = distribution.quantile(resolved_quantile)
            _LOG.info(
                "[conflicts] quantile %.6f over %d %s pairs (%d comparable) -> "
                "cos_threshold %.4f, rank cutoff ~%.0f",
                resolved_quantile,
                distribution.n_pairs,
                "exhaustive" if distribution.exhaustive else "sampled",
                distribution.total_pairs,
                calibrated,
                (1.0 - resolved_quantile) * distribution.total_pairs,
            )
            return calibrated, "calibrated", resolved_quantile
    return default, "default", None
