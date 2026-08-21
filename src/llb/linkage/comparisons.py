"""Translate the project's typed comparison vocabulary into Splink comparison creators.

Splink is lazy-imported here and nowhere else in the seam's hot path, so `llb.linkage.spec` and
the artifact readers stay importable in the base install (no `[linkage]` extra required).
"""

import hashlib
import json
from typing import Any

from llb.linkage.constants import (
    KIND_ARRAY_INTERSECT,
    KIND_COSINE,
    KIND_DATE_DIFFERENCE,
    KIND_EXACT,
    KIND_JACCARD,
    KIND_JARO_WINKLER,
    KIND_LEVENSHTEIN,
    SPLINK_MISSING,
)
from llb.linkage.spec import BlockingRule, ComparisonSpec, LinkageSpec


def require_splink() -> Any:
    """Import Splink or fail with the one-line install hint the rest of the CLI uses."""
    try:
        import splink
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SystemExit(SPLINK_MISSING) from exc
    return splink


def build_comparison(comparison: ComparisonSpec) -> Any:
    """One `ComparisonSpec` -> the Splink comparison creator that scores it."""
    require_splink()
    import splink.comparison_library as cl

    column, kind = comparison.column, comparison.kind
    if kind == KIND_EXACT:
        return cl.ExactMatch(column)
    if kind == KIND_LEVENSHTEIN:
        return cl.LevenshteinAtThresholds(column, [int(t) for t in comparison.thresholds])
    if kind == KIND_JARO_WINKLER:
        return cl.JaroWinklerAtThresholds(column, list(comparison.thresholds))
    if kind == KIND_ARRAY_INTERSECT:
        return cl.ArrayIntersectAtSizes(column, [int(t) for t in comparison.thresholds])
    if kind == KIND_JACCARD:
        return cl.JaccardAtThresholds(column, list(comparison.thresholds))
    if kind == KIND_COSINE:
        return cl.CosineSimilarityAtThresholds(column, list(comparison.thresholds))
    if kind == KIND_DATE_DIFFERENCE:
        # `date_metric` is validated against DATE_METRICS in ComparisonSpec.validate; Splink types
        # it as a Literal, which a spec loaded from JSON cannot be.
        metric: Any = comparison.date_metric
        return cl.AbsoluteDateDifferenceAtThresholds(
            column,
            input_is_string=True,
            datetime_format=comparison.date_format,
            metrics=[metric] * len(comparison.thresholds),
            thresholds=[int(t) for t in comparison.thresholds],
            term_frequency_adjustments=comparison.term_frequency,
        )
    raise ValueError(f"no Splink builder for comparison kind {kind!r}")  # pragma: no cover


def build_blocking_rule(rule: BlockingRule) -> Any:
    """One `BlockingRule` -> the Splink blocking-rule creator that generates its candidates."""
    require_splink()
    from splink import block_on

    return block_on(*rule.expressions)


def settings_uid(spec: LinkageSpec) -> str:
    """A stable model id derived from the specification.

    Splink's default is a random uid, which would make two byte-identical fits produce two
    different `model.json` files and turn every re-run into a spurious artifact diff.
    """
    payload = json.dumps(spec.payload(), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_settings(spec: LinkageSpec) -> Any:
    """The whole spec -> a Splink `SettingsCreator` for a dedupe-only linkage."""
    require_splink()
    from splink import SettingsCreator

    return SettingsCreator(
        link_type="dedupe_only",
        comparisons=[build_comparison(c) for c in spec.comparisons],
        blocking_rules_to_generate_predictions=[
            build_blocking_rule(r) for r in spec.blocking_rules
        ],
        probability_two_random_records_match=spec.random_match_probability,
        em_convergence=spec.em_convergence,
        max_iterations=spec.em_max_iterations,
        unique_id_column_name=spec.unique_id_column,
        retain_matching_columns=True,
        retain_intermediate_calculation_columns=False,
        additional_columns_to_retain=list(spec.retain_columns),
        linker_uid=settings_uid(spec),
    )
