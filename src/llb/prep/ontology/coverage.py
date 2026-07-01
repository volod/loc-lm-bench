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


def sample_seeds(
    docs: list[DocRecord],
    extractions: list[DocExtraction],
    *,
    max_items: int = DEFAULT_MAX_ITEMS,
    seed: int = 13,
) -> list[DraftSeed]:
    """Pick up to `max_items` seeds with maximal stratum coverage, deterministically."""
    pool = build_seeds(docs, extractions)
    if not pool:
        return []
    rng = random.Random(seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)

    covered: set[str] = set()
    chosen: list[DraftSeed] = []
    chosen_ids: set[int] = set()
    # pass 1: coverage-first -- each new pick must introduce a fresh bucket
    for i, candidate in enumerate(shuffled):
        if len(chosen) >= max_items:
            break
        new_buckets = [b for b in _buckets(candidate) if b not in covered]
        if new_buckets:
            chosen.append(candidate)
            chosen_ids.add(i)
            covered.update(_buckets(candidate))
    # pass 2: fill remaining budget in deterministic shuffled order
    for i, candidate in enumerate(shuffled):
        if len(chosen) >= max_items:
            break
        if i not in chosen_ids:
            chosen.append(candidate)
    _LOG.info(
        "[ontology] stage 4: %d seeds chosen (of %d) covering %d strata buckets",
        len(chosen),
        len(pool),
        len(covered),
    )
    return chosen
