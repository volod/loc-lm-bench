"""Turn one declared `AnswerEnvelope` into the extraction record the axiom checker already reads.

The whole point of the two-step gate is that step two adds no second checker: the axiom classes,
their evidence shapes, and their base-rate accounting all ship (`llb.prep.ontology.axioms`). What
was missing is the ADAPTER -- an answer expressed as the same `DocExtraction` a corpus document
produces, grounded in a synthetic document so every violation still names its evidence exactly.

Two decisions here are what keep the gate from refusing correct work:

  - a `MISC` endpoint type asserts NOTHING. `MISC` is the closed vocabulary's fallback
    (`normalize_entity_type` collapses anything out-of-vocabulary into it), so treating it as a
    type claim would make every unrecognized surface a `domain`/`range` violation. An endpoint
    typed `MISC` contributes no type assertion, which is exactly how the ledger checker already
    treats an untyped endpoint: unchecked, never failed.
  - endpoint surfaces fold through the corpus's own notion of identity before they are compared
    (`llb.eval.answer_validation.identity`), so a paraphrase the corpus already treats as one
    thing does not read as a second value. The fold is given the endpoint's DECLARED type, because
    two written forms of one QUANTITY are the same value while the same two strings read as a name
    are not.
"""

from collections.abc import Callable

from llb.eval.answer_envelope.models import AnswerEnvelope, EnvelopeClaim, EnvelopeTriple
from llb.eval.answer_validation.constants import ANSWER_DOC_ID
from llb.goldset.schema import SourceSpan
from llb.prep.ontology.extraction.entity_types import DEFAULT_ENTITY_TYPE
from llb.prep.ontology.models import DocExtraction, Entity, SROFact

# How an endpoint surface and its declared type are mapped onto the surface the corpus records.
Canonicalize = Callable[[str, str], str]


def declared_triples(envelope: AnswerEnvelope) -> list[tuple[EnvelopeClaim, EnvelopeTriple]]:
    """Every claim that declared a triple -- the population the gate can actually test."""
    return [(claim, claim.triple) for claim in envelope.claims if claim.triple is not None]


def answer_extraction(
    envelope: AnswerEnvelope, canonical: Canonicalize | None = None
) -> DocExtraction:
    """The envelope's declared triples as one `DocExtraction` under the answer's synthetic doc id.

    Each claim occupies its own span of a synthetic document whose text is the claim texts joined,
    so a violation the gate reports quotes the CLAIM that caused it, the same way a corpus
    violation quotes the sentence it came from.
    """
    fold: Canonicalize = canonical or (lambda name, _type: name)
    facts: list[SROFact] = []
    entities: list[Entity] = []
    offset = 0
    for claim, triple in declared_triples(envelope):
        text = claim.text or f"{triple.subject} {triple.relation} {triple.object}"
        span = SourceSpan(
            doc_id=ANSWER_DOC_ID,
            char_start=offset,
            char_end=offset + len(text),
            text=text,
        )
        offset = span.char_end + 1  # the joining separator, so no two claims share a span
        subject = fold(triple.subject, triple.subject_type)
        obj = fold(triple.object, triple.object_type)
        facts.append(SROFact(subject=subject, relation=triple.relation, object=obj, evidence=span))
        entities += _typed(subject, triple.subject_type, span)
        entities += _typed(obj, triple.object_type, span)
    return DocExtraction(doc_id=ANSWER_DOC_ID, entities=entities, facts=facts)


def _typed(name: str, entity_type: str, span: SourceSpan) -> list[Entity]:
    """The endpoint's type assertion, or none at all when the model fell back to `MISC`."""
    if entity_type == DEFAULT_ENTITY_TYPE:
        return []
    return [Entity(name=name, type=entity_type, aliases=[], mentions=[span])]
