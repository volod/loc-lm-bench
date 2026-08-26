"""WHICH ledger facts the retrieved context put in front of the model -- the gate's per-case scope.

Only the facts and type assertions whose evidence span falls inside a chunk the prompt actually
carried. A fact the model never saw cannot be a contradiction it committed; scoring it as one
would refuse an answer for the corpus's content rather than for the answer's, which is exactly the
failure mode a validator has to avoid.

The other half of "does this answer contradict the corpus" -- WHICH surfaces are the same thing --
has its own module (`llb.eval.answer_validation.identity`): the corpus's recorded aliases, the node
overlay the entity-resolution lane proposes, and value equivalence for the types whose members are
values. This module owns only the scoping and hands identity to that one, so the two cannot drift.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

from llb.core.contracts.rag import ChunkRecord
from llb.eval.answer_validation.equivalence import Lemmatizer
from llb.eval.answer_validation.identity import SurfaceIdentity
from llb.goldset.schema import SourceSpan
from llb.prep.ontology.extraction.entity_types import DEFAULT_ENTITY_TYPE
from llb.prep.ontology.models import DocExtraction, Entity

# A (doc_id, char_start, char_end) window the prompt carried.
Window = tuple[str, int, int]


def chunk_windows(chunks: Sequence[ChunkRecord]) -> list[Window]:
    """The evidence windows the prompt laid in front of the model, as typed triples."""
    return [
        (
            str(chunk.get("doc_id", "")),
            int(chunk.get("char_start", 0)),
            int(chunk.get("char_end", 0)),
        )
        for chunk in chunks
    ]


def _visible(span: SourceSpan, windows: Iterable[Window]) -> bool:
    """Whether a span's evidence overlaps any retrieved window of the same document.

    OVERLAP, not containment: a chunk boundary that cuts an evidence span still put the fact in
    front of the model, and requiring containment would silently drop exactly the facts near a
    boundary -- an artifact of the chunker, not of what the model could read.
    """
    return any(
        doc_id == span.doc_id and span.char_start < end and start < span.char_end
        for doc_id, start, end in windows
    )


class CorpusLedger:
    """The whole extraction ledger, with the per-case scoping and identity fold the gate needs."""

    def __init__(
        self,
        extractions: Sequence[DocExtraction],
        *,
        overlay: Path | str | None = None,
        lemmatize: Lemmatizer | None = None,
    ) -> None:
        self._extractions = list(extractions)
        # Indexed by document because scoping is a per-CASE operation over a handful of retrieved
        # documents: scanning the whole ledger once per answer would make the gate's cost scale
        # with the corpus rather than with the context.
        self._by_doc = {extraction.doc_id: extraction for extraction in self._extractions}
        self.identity = SurfaceIdentity(self._extractions, overlay=overlay, lemmatize=lemmatize)

    @property
    def n_docs(self) -> int:
        return len(self._extractions)

    @property
    def n_facts(self) -> int:
        return sum(len(extraction.facts) for extraction in self._extractions)

    def canonical(self, name: str, entity_type: str = "") -> str:
        """The surface the corpus records this endpoint under, given the type it was declared as."""
        return self.identity.fold(name, entity_type)

    def scoped(self, chunks: Sequence[ChunkRecord]) -> list[DocExtraction]:
        """The ledger restricted to the evidence the retrieved chunks carried.

        An entity keeps only the mentions inside the window, so an entity typed on the strength of
        a mention the prompt never showed contributes no type assertion either.
        """
        windows = chunk_windows(chunks)
        scoped: list[DocExtraction] = []
        for doc_id in dict.fromkeys(doc for doc, _start, _end in windows):
            extraction = self._by_doc.get(doc_id)
            if extraction is None:
                continue
            facts = [fact for fact in extraction.facts if _visible(fact.evidence, windows)]
            entities = _scoped_entities(extraction.entities, windows)
            if facts or entities:
                scoped.append(
                    DocExtraction(doc_id=extraction.doc_id, entities=entities, facts=facts)
                )
        return scoped


def scoped_fact_count(extractions: Sequence[DocExtraction]) -> int:
    """How many corpus facts the gate had in front of it -- the population a verdict rests on."""
    return sum(len(extraction.facts) for extraction in extractions)


def _scoped_entities(entities: Sequence[Entity], windows: Sequence[Window]) -> list[Entity]:
    """The typed entities the window carried. A `MISC` type is dropped, on BOTH sides.

    `MISC` is what `normalize_entity_type` collapses anything out-of-vocabulary into, so it names
    the absence of a recognized type rather than a type. The answer side already drops it
    (`answer_ledger._typed`); dropping it here too is what keeps the two sides symmetric -- an
    extractor fallback must not be able to refuse an answer under a `domain`/`range` axiom any more
    than the model's own fallback can.
    """
    kept: list[Entity] = []
    for entity in entities:
        if entity.type == DEFAULT_ENTITY_TYPE:
            continue
        mentions = [mention for mention in entity.mentions if _visible(mention, windows)]
        if mentions:
            kept.append(
                Entity(
                    name=entity.name,
                    type=entity.type,
                    aliases=list(entity.aliases),
                    mentions=mentions,
                )
            )
    return kept
