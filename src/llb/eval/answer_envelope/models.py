"""The typed RAG answer contract (typed-rag-answer-envelope).

The free-text answer path recovers every answer-side signal AFTER the fact: a status from regex
markers, an abstention from first-person refusal stems, citations scraped out of prose, and
"claims" re-segmented by punctuation. Each of those is an estimate of a thing the model could
simply have DECLARED. This module is the declaration: one Pydantic contract the generation
boundary parses a completion into, so a status, an abstention, a claim, and its citations are
recorded fields rather than inferences.

What is REQUIRED is the part a heuristic cannot recover reliably:

  - `answer` -- the Ukrainian answer text, scored exactly as the free-text answer of the same
    string would be (the envelope changes where the string comes from, never how it is scored);
  - `abstained` -- an explicit flag, so "the context does not carry it" stops being a regex over
    apology stems;
  - `claims` -- each factual statement with the prompt-position `[i]` indices it rests on, so
    citation validity reads a declared list instead of `[i]` markers scraped from prose. An
    abstaining envelope declares an empty list, which is itself a statement.

`evidence` is optional: verbatim quotes are asked for in the prompt and are useful for a reviewer,
but the citations already carry what the answer-side metrics need, so a model that omits them has
not failed the contract. `triple` is optional per claim and is typed against the CLOSED entity
vocabulary (`llb.prep.ontology.extraction.entity_types`) -- an out-of-vocabulary type normalizes
to `MISC` rather than expanding the schema, exactly as the extraction path does.

Unknown extra keys are IGNORED, not rejected: a model that adds `"confidence"` has still emitted
the contract, and conformance should measure the declared fields, not decoration around them.

SEMANTIC checks on the contents (does the triple hold? does the claim follow from its chunk?) are
deliberately NOT here -- they are the next capability's job. This module answers one question:
did the model emit the requested shape?
"""

import json
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

from llb.prep.ontology.extraction.entity_types import (
    DEFAULT_ENTITY_TYPE,
    entity_types_prompt_block,
    normalize_entity_type,
)

# A raw type string folded into the closed vocabulary (synonyms canonicalized, unknowns -> MISC).
EntityType = Annotated[str, AfterValidator(normalize_entity_type)]


class EnvelopeTriple(BaseModel):
    """An optional subject/relation/object reading of one claim, typed against the ontology.

    The two type fields default to `MISC` so a model that emits a bare triple still satisfies the
    contract; what it may not do is invent a type, because normalization is what keeps the
    vocabulary closed.
    """

    subject: str
    relation: str
    object: str
    subject_type: EntityType = DEFAULT_ENTITY_TYPE
    object_type: EntityType = DEFAULT_ENTITY_TYPE


class EvidenceSpan(BaseModel):
    """A verbatim quote copied out of one numbered prompt chunk."""

    chunk: int
    quote: str


class EnvelopeClaim(BaseModel):
    """One factual statement plus the prompt positions (`[i]` numbering) it is drawn from."""

    text: str
    citations: list[int] = Field(default_factory=list)
    triple: EnvelopeTriple | None = None


class AnswerEnvelope(BaseModel):
    """The whole declared answer: text, abstention, per-claim citations, and evidence quotes."""

    answer: str
    abstained: bool
    claims: list[EnvelopeClaim]
    evidence: list[EvidenceSpan] = Field(default_factory=list)


# The one worked example shown to the model. It is a MODEL INSTANCE, not prompt text, so the
# schema block in the prompt cannot drift from the contract this module validates against: adding
# a field here changes what the model is asked for and what it is checked against together.
ENVELOPE_EXAMPLE = AnswerEnvelope(
    answer="Конвенцію підписано 1994 року.",
    abstained=False,
    claims=[
        EnvelopeClaim(
            text="Конвенцію підписано 1994 року.",
            citations=[1],
            triple=EnvelopeTriple(
                subject="Конвенція",
                relation="підписана",
                object="1994",
                subject_type="LAW",
                object_type="DATE",
            ),
        )
    ],
    evidence=[EvidenceSpan(chunk=1, quote="Конвенцію було підписано у 1994 році")],
)


def envelope_schema_block() -> str:
    """The worked JSON example injected into the envelope prompt."""
    return json.dumps(ENVELOPE_EXAMPLE.model_dump(), ensure_ascii=False, indent=2)


def envelope_prompt_values() -> dict[str, str]:
    """The template variables the envelope generation and repair prompts render from."""
    return {"schema": envelope_schema_block(), "entity_types": entity_types_prompt_block()}
