"""Stage 4 -- sample seeds with coverage across document, semantic type, section, difficulty.

Each extracted fact, entity, claim, and event becomes a candidate `DraftSeed` tagged with the
coverage buckets it fills. Selection is a deterministic, seeded greedy: shuffle once, then take
seeds that introduce a not-yet-covered bucket first (so the drafted set spans documents,
relations / entity types / claims / events, sections, and difficulty), then fill any remaining
budget in shuffled order. Deterministic for a given (extractions, seed, max_items).
"""

import logging
import random
from collections import defaultdict

from llb.prep.ontology.constants import (
    DEFAULT_COVERAGE_TARGET,
    DEFAULT_MAX_ITEMS,
    DIFFICULTY_EASY_MAX_CHARS,
    DIFFICULTY_HARD_MIN_CHARS,
    RARE_RELATION_MAX_COUNT,
)
from llb.prep.ontology.inventory import section_at
from llb.prep.ontology.models import DocExtraction, DocRecord, DraftSeed

_LOG = logging.getLogger(__name__)


def classify_difficulty(evidence_len: int, *, rare: bool) -> str:
    """Short, common evidence is easy; long or rare evidence is hard."""
    if rare or evidence_len >= DIFFICULTY_HARD_MIN_CHARS:
        return "hard"
    if evidence_len <= DIFFICULTY_EASY_MAX_CHARS:
        return "easy"
    return "medium"


def _base_strata(doc_id: str, section: str, difficulty: str) -> dict[str, str]:
    """Coverage buckets shared by every seed kind."""
    return {
        "doc": doc_id,
        "section": section,
        "difficulty": difficulty,
    }


def build_seeds(docs: list[DocRecord], extractions: list[DocExtraction]) -> list[DraftSeed]:
    """Build the candidate seed pool, each item tagged with its coverage strata."""
    by_id = {doc.doc_id: doc for doc in docs}
    relation_counts: dict[str, int] = defaultdict(int)
    for extraction in extractions:
        for fact in extraction.facts:
            relation_counts[fact.relation] += 1

    seeds: list[DraftSeed] = []
    for extraction in extractions:
        doc = by_id.get(extraction.doc_id)
        if doc is None:
            continue
        for fact in extraction.facts:
            section = section_at(doc.sections, fact.evidence.char_start)
            rare = relation_counts[fact.relation] <= RARE_RELATION_MAX_COUNT
            difficulty = classify_difficulty(len(fact.evidence.text), rare=rare)
            seeds.append(
                DraftSeed(
                    doc_id=doc.doc_id,
                    kind="fact",
                    section_title=section,
                    difficulty=difficulty,
                    strata={
                        **_base_strata(doc.doc_id, section, difficulty),
                        "relation": fact.relation,
                    },
                    evidence=fact.evidence,
                    fact=fact,
                )
            )
        for entity in extraction.entities:
            mention = entity.mentions[0]
            section = section_at(doc.sections, mention.char_start)
            difficulty = classify_difficulty(len(mention.text), rare=False)
            seeds.append(
                DraftSeed(
                    doc_id=doc.doc_id,
                    kind="entity",
                    section_title=section,
                    difficulty=difficulty,
                    strata={
                        **_base_strata(doc.doc_id, section, difficulty),
                        "entity_type": entity.type,
                    },
                    evidence=mention,
                    entity=entity,
                )
            )
        for claim in extraction.claims:
            section = section_at(doc.sections, claim.evidence.char_start)
            difficulty = classify_difficulty(len(claim.evidence.text), rare=False)
            seeds.append(
                DraftSeed(
                    doc_id=doc.doc_id,
                    kind="claim",
                    section_title=section,
                    difficulty=difficulty,
                    strata={
                        **_base_strata(doc.doc_id, section, difficulty),
                        "claim": claim.text,
                    },
                    evidence=claim.evidence,
                    claim=claim,
                )
            )
        for event in extraction.events:
            section = section_at(doc.sections, event.evidence.char_start)
            difficulty = classify_difficulty(len(event.evidence.text), rare=False)
            seeds.append(
                DraftSeed(
                    doc_id=doc.doc_id,
                    kind="event",
                    section_title=section,
                    difficulty=difficulty,
                    strata={
                        **_base_strata(doc.doc_id, section, difficulty),
                        "event": event.description,
                    },
                    evidence=event.evidence,
                    event=event,
                )
            )
    return seeds


def _buckets(seed: DraftSeed) -> list[str]:
    return [f"{key}={value}" for key, value in seed.strata.items()]


def _select_flat_cap(shuffled: list[DraftSeed], max_items: int) -> list[DraftSeed]:
    """Coverage-first then fill: each pass-1 pick introduces a fresh bucket, pass 2 fills budget."""
    covered: set[str] = set()
    chosen: list[DraftSeed] = []
    chosen_ids: set[int] = set()
    for i, candidate in enumerate(shuffled):
        if len(chosen) >= max_items:
            break
        new_buckets = [b for b in _buckets(candidate) if b not in covered]
        if new_buckets:
            chosen.append(candidate)
            chosen_ids.add(i)
            covered.update(_buckets(candidate))
    for i, candidate in enumerate(shuffled):
        if len(chosen) >= max_items:
            break
        if i not in chosen_ids:
            chosen.append(candidate)
    return chosen


def _select_coverage_target(
    shuffled: list[DraftSeed], *, coverage_target: int, max_items: int
) -> list[DraftSeed]:
    """Draft up to `coverage_target` seeds per stratum bucket (relation / entity type / section /
    semantic kind), so yield tracks corpus breadth instead of a flat item cap. A candidate is kept
    while ANY of its buckets is still below the target; `max_items` stays a safety ceiling."""
    counts: dict[str, int] = defaultdict(int)
    chosen: list[DraftSeed] = []
    for candidate in shuffled:
        if len(chosen) >= max_items:
            break
        buckets = _buckets(candidate)
        if any(counts[b] < coverage_target for b in buckets):
            chosen.append(candidate)
            for b in buckets:
                counts[b] += 1
    return chosen


def select_seeds(
    pool: list[DraftSeed],
    *,
    max_items: int = DEFAULT_MAX_ITEMS,
    seed: int = 13,
    coverage_target: int | None = DEFAULT_COVERAGE_TARGET,
) -> list[DraftSeed]:
    """Select seeds from a prebuilt pool, deterministically for a given (pool, seed, knobs).

    `coverage_target` (yield-max): when set, draft up to that many seeds per stratum bucket instead
    of stopping at the flat `max_items` cap; when None, the flat cap applies (coverage-first then
    fill). `max_items` is always the hard ceiling on the returned count.
    """
    if not pool:
        return []
    rng = random.Random(seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    if coverage_target is not None and coverage_target >= 1:
        chosen = _select_coverage_target(
            shuffled, coverage_target=coverage_target, max_items=max_items
        )
        mode = f"coverage-target={coverage_target}"
    else:
        chosen = _select_flat_cap(shuffled, max_items)
        mode = "flat-cap"
    _LOG.info("[ontology] stage 4: %d seeds chosen (of %d) [%s]", len(chosen), len(pool), mode)
    return chosen


def sample_seeds(
    docs: list[DocRecord],
    extractions: list[DocExtraction],
    *,
    max_items: int = DEFAULT_MAX_ITEMS,
    seed: int = 13,
    coverage_target: int | None = DEFAULT_COVERAGE_TARGET,
) -> list[DraftSeed]:
    """Build the candidate pool and select seeds (see `select_seeds`)."""
    return select_seeds(
        build_seeds(docs, extractions),
        max_items=max_items,
        seed=seed,
        coverage_target=coverage_target,
    )


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
