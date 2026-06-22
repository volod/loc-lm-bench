"""Stage 3 -- induce a constrained candidate ontology from the extractions.

Pure, deterministic aggregation (no LLM call): entity types come from the extracted entities,
relations from the SRO facts. Each type carries a support `count`, a `confidence`
(count / max-count in its group), and a few example surface forms with their evidence. The
candidate is CONSTRAINED -- hapax types below `MIN_TYPE_COUNT` are dropped and each group is
capped -- so the ontology stays small and reviewable. Examples preserve source spans, so every
induced type links back to exact evidence (M4.4 acceptance).
"""

import logging
from collections import defaultdict

from llb.prep.ontology.constants import (
    MAX_ENTITY_TYPES,
    MAX_RELATION_TYPES,
    MIN_TYPE_COUNT,
    N_TYPE_EXAMPLES,
)
from llb.prep.ontology.models import DocExtraction, OntologyCandidate, OntologyType

_LOG = logging.getLogger(__name__)


def _induce_group(
    counts: dict[str, int], examples: dict[str, list[str]], *, cap: int, min_count: int
) -> list[OntologyType]:
    if not counts:
        return []
    max_count = max(counts.values())
    types = [
        OntologyType(
            name=name,
            count=count,
            confidence=round(count / max_count, 4),
            examples=examples[name][:N_TYPE_EXAMPLES],
        )
        for name, count in counts.items()
        if count >= min_count
    ]
    # deterministic: most-supported first, ties broken by name
    types.sort(key=lambda t: (-t.count, t.name))
    return types[:cap]


def induce_ontology(extractions: list[DocExtraction]) -> OntologyCandidate:
    """Aggregate extracted entity types + relations into a constrained ontology candidate."""
    ent_counts: dict[str, int] = defaultdict(int)
    ent_examples: dict[str, list[str]] = defaultdict(list)
    rel_counts: dict[str, int] = defaultdict(int)
    rel_examples: dict[str, list[str]] = defaultdict(list)

    for extraction in extractions:
        for entity in extraction.entities:
            ent_counts[entity.type] += 1
            if entity.name not in ent_examples[entity.type]:
                ent_examples[entity.type].append(entity.name)
        for fact in extraction.facts:
            rel_counts[fact.relation] += 1
            surface = f"{fact.subject} -> {fact.object}"
            if surface not in rel_examples[fact.relation]:
                rel_examples[fact.relation].append(surface)

    candidate = OntologyCandidate(
        entity_types=_induce_group(
            ent_counts, ent_examples, cap=MAX_ENTITY_TYPES, min_count=MIN_TYPE_COUNT
        ),
        relation_types=_induce_group(
            rel_counts, rel_examples, cap=MAX_RELATION_TYPES, min_count=MIN_TYPE_COUNT
        ),
    )
    _LOG.info(
        "[ontology] stage 3: %d entity types, %d relations induced",
        len(candidate.entity_types),
        len(candidate.relation_types),
    )
    return candidate
