"""Focused coverage report implementation."""

from collections import defaultdict
from llb.prep.ontology.models import DraftSeed


def _dimension_stats(
    pool: list[DraftSeed], chosen: list[DraftSeed], dim: str, *, coverage_target: int | None
) -> dict[str, int]:
    """Per-stratum-dimension draft coverage: buckets, buckets drafted / at target, seeds remaining."""
    pool_by_value: dict[str, int] = defaultdict(int)
    chosen_by_value: dict[str, int] = defaultdict(int)
    for s in pool:
        if dim in s.strata:
            pool_by_value[s.strata[dim]] += 1
    for s in chosen:
        if dim in s.strata:
            chosen_by_value[s.strata[dim]] += 1
    drafted = sum(chosen_by_value.values())
    available = sum(pool_by_value.values())
    stats = {
        "buckets": len(pool_by_value),
        "buckets_drafted": sum(1 for v in pool_by_value if chosen_by_value.get(v, 0) > 0),
        "drafted_seeds": drafted,
        "seeds_remaining": available - drafted,
    }
    if coverage_target is not None:
        stats["buckets_at_target"] = sum(
            1
            for value, n_avail in pool_by_value.items()
            if chosen_by_value.get(value, 0) >= min(coverage_target, n_avail)
        )
    return stats


def coverage_report(
    pool: list[DraftSeed],
    chosen: list[DraftSeed],
    *,
    coverage_target: int | None,
    max_items: int,
) -> dict[str, object]:
    """A "seeds remaining vs drafted" exhaustion report over every stratum dimension + semantic kind.

    Records, per stratum key (doc / section / difficulty / relation / entity_type / claim / event),
    how many buckets exist, how many were drafted (and, in coverage-target mode, how many reached
    the target), and how many candidate seeds remain undrafted -- so an operator sees whether a
    draft exhausted the corpus's breadth or was cut short by the item ceiling.
    """
    dims = sorted({key for s in pool for key in s.strata})
    kinds = sorted({s.kind for s in pool})
    by_kind: dict[str, dict[str, int]] = {}
    for kind in kinds:
        pool_n = sum(1 for s in pool if s.kind == kind)
        chosen_n = sum(1 for s in chosen if s.kind == kind)
        by_kind[kind] = {"pool": pool_n, "drafted": chosen_n, "remaining": pool_n - chosen_n}
    return {
        "mode": "coverage-target" if coverage_target is not None else "flat-cap",
        "coverage_target": coverage_target,
        "max_items": max_items,
        "pool_seeds": len(pool),
        "drafted_seeds": len(chosen),
        "seeds_remaining": len(pool) - len(chosen),
        "exhausted": len(chosen) >= len(pool),
        "by_semantic_kind": by_kind,
        "strata": {
            dim: _dimension_stats(pool, chosen, dim, coverage_target=coverage_target)
            for dim in dims
        },
    }
