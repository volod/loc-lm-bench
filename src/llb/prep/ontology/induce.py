"""Stage 3 -- induce a constrained candidate ontology from the extractions.

Pure, deterministic aggregation (no LLM call): entity types come from the extracted entities,
relations from the SRO facts. Each type carries a support `count`, a `confidence`, and a few
example surface forms with their evidence. The candidate is CONSTRAINED -- hapax types below
`MIN_TYPE_COUNT` are dropped and each group is capped -- so the ontology stays small and
reviewable. Examples preserve source spans, so every induced type links back to exact evidence
(ontology-assisted drafting acceptance).

`confidence` is a richer signal than raw frequency (verified-data hardening): a type seen many times in ONE document
is less trustworthy than one of similar count SPREAD across documents, so confidence blends the
normalized count with the normalized DOCUMENT frequency. The high-confidence induced types are
carried into the drafting prompt as explicit constraints (`ontology_constraints`).
"""

import logging
from collections import defaultdict

from llb.prep.ontology.constants import (
    CONFIDENCE_COUNT_WEIGHT,
    CONFIDENCE_DOCFREQ_WEIGHT,
    MAX_ENTITY_TYPES,
    MAX_RELATION_TYPES,
    MIN_TYPE_COUNT,
    N_CONSTRAINT_TYPES,
    N_TYPE_EXAMPLES,
    ONTOLOGY_CONSTRAINT_MIN_CONFIDENCE,
)
from llb.prep.ontology.models import DocExtraction, OntologyCandidate, OntologyType


def _confidence(count: int, max_count: int, n_docs: int, total_docs: int) -> float:
    """Blend normalized count with normalized document frequency (verified-data hardening richer signal)."""
    count_norm = count / max_count if max_count else 0.0
    docfreq_norm = n_docs / total_docs if total_docs else 0.0
    return round(CONFIDENCE_COUNT_WEIGHT * count_norm + CONFIDENCE_DOCFREQ_WEIGHT * docfreq_norm, 4)


_LOG = logging.getLogger(__name__)


def _induce_group(
    counts: dict[str, int],
    examples: dict[str, list[str]],
    docs_by_type: dict[str, set[str]],
    total_docs: int,
    *,
    cap: int,
    min_count: int,
) -> list[OntologyType]:
    if not counts:
        return []
    max_count = max(counts.values())
    types = [
        OntologyType(
            name=name,
            count=count,
            confidence=_confidence(count, max_count, len(docs_by_type[name]), total_docs),
            examples=examples[name][:N_TYPE_EXAMPLES],
        )
        for name, count in counts.items()
        if count >= min_count
    ]
    # deterministic: highest confidence first, then most-supported, ties broken by name
    types.sort(key=lambda t: (-t.confidence, -t.count, t.name))
    return types[:cap]


def induce_ontology(extractions: list[DocExtraction]) -> OntologyCandidate:
    """Aggregate extracted entity types + relations into a constrained ontology candidate."""
    ent_counts: dict[str, int] = defaultdict(int)
    ent_examples: dict[str, list[str]] = defaultdict(list)
    ent_docs: dict[str, set[str]] = defaultdict(set)
    rel_counts: dict[str, int] = defaultdict(int)
    rel_examples: dict[str, list[str]] = defaultdict(list)
    rel_docs: dict[str, set[str]] = defaultdict(set)

    for extraction in extractions:
        for entity in extraction.entities:
            ent_counts[entity.type] += 1
            ent_docs[entity.type].add(extraction.doc_id)
            if entity.name not in ent_examples[entity.type]:
                ent_examples[entity.type].append(entity.name)
        for fact in extraction.facts:
            rel_counts[fact.relation] += 1
            rel_docs[fact.relation].add(extraction.doc_id)
            surface = f"{fact.subject} -> {fact.object}"
            if surface not in rel_examples[fact.relation]:
                rel_examples[fact.relation].append(surface)

    total_docs = len(extractions)
    candidate = OntologyCandidate(
        entity_types=_induce_group(
            ent_counts,
            ent_examples,
            ent_docs,
            total_docs,
            cap=MAX_ENTITY_TYPES,
            min_count=MIN_TYPE_COUNT,
        ),
        relation_types=_induce_group(
            rel_counts,
            rel_examples,
            rel_docs,
            total_docs,
            cap=MAX_RELATION_TYPES,
            min_count=MIN_TYPE_COUNT,
        ),
    )
    _LOG.info(
        "[ontology] stage 3: %d entity types, %d relations induced",
        len(candidate.entity_types),
        len(candidate.relation_types),
    )
    return candidate


def ontology_constraints(
    candidate: OntologyCandidate,
    *,
    min_confidence: float = ONTOLOGY_CONSTRAINT_MIN_CONFIDENCE,
    n_types: int = N_CONSTRAINT_TYPES,
) -> str:
    """A drafting-prompt hint listing the HIGH-CONFIDENCE induced types (entities + relations) as
    explicit constraints, so the drafter focuses questions on the corpus's reliable types (verified-data hardening).

    Returns an empty string when no induced type clears the confidence floor (no hint added)."""
    entities = [t for t in candidate.entity_types if t.confidence >= min_confidence][:n_types]
    relations = [t for t in candidate.relation_types if t.confidence >= min_confidence][:n_types]
    if not entities and not relations:
        return ""
    lines = []
    if entities:
        lines.append("типи сутностей: " + ", ".join(t.name for t in entities))
    if relations:
        lines.append("типи відношень: " + ", ".join(t.name for t in relations))
    return "Орієнтуйся на провідні типи корпусу -- " + "; ".join(lines) + "."
