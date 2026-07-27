"""The ordered query-prep pipeline that composes the individual steps into one pass."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from llb.rag.query_prep.base import (
    QUERY_PREP_STEPS,
    STEP_DECOMPOSE,
    STEP_GLOSSARY,
    STEP_HYDE,
    STEP_NORMALIZE,
    STEP_REWRITE,
    STEP_TYPOS,
    KnownWordProbe,
    LanguageGate,
    PlausibilityProbe,
    QueryEdit,
    QueryGenerator,
    QueryPrepResult,
    Rewriter,
)
from llb.rag.query_prep.decompose import apply_decompose
from llb.rag.query_prep.glossary import Glossary, apply_glossary
from llb.rag.query_prep.hyde import apply_hyde
from llb.rag.query_prep.normalize import apply_normalize, language_gate
from llb.rag.query_prep.restore import VocabularyContext, normalization_provenance
from llb.rag.query_prep.rewrite import apply_rewrite
from llb.rag.query_prep.typos import apply_typos


def _validate_steps(
    ordered: tuple[str, ...],
    *,
    vocabulary: frozenset[str] | None,
    glossary: Glossary | None,
    rewriter: Rewriter | None,
    hypothesizer: QueryGenerator | None,
    decomposer: QueryGenerator | None,
    known_word: KnownWordProbe | None,
    context: VocabularyContext | None,
    plausible: PlausibilityProbe | None,
) -> None:
    """Validate the ordered step contract and all dependency wiring."""
    unknown = [step for step in ordered if step not in QUERY_PREP_STEPS]
    if unknown:
        raise ValueError(
            f"unknown query-prep step(s): {unknown}; choose from {list(QUERY_PREP_STEPS)}"
        )
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"duplicate query-prep step(s): {ordered}")

    required_dependencies = (
        (STEP_TYPOS, vocabulary, "the 'typos' step needs a corpus vocabulary"),
        (STEP_GLOSSARY, glossary, "the 'glossary' step needs a query glossary"),
        (STEP_REWRITE, rewriter, "the 'rewrite' step needs a rewrite endpoint callable"),
        (STEP_HYDE, hypothesizer, "the 'hyde' step needs a hypothetical-answer endpoint callable"),
        (
            STEP_DECOMPOSE,
            decomposer,
            "the 'decompose' step needs a decomposition endpoint callable",
        ),
    )
    for step, dependency, message in required_dependencies:
        if step in ordered and dependency is None:
            raise ValueError(message)

    optional_dependencies = (
        (known_word, STEP_TYPOS, "the typo morphology guard needs the 'typos' step"),
        (context, STEP_TYPOS, "the query-context index needs the 'typos' step"),
        (plausible, STEP_NORMALIZE, "the normalize language gate needs the 'normalize' step"),
    )
    for optional_dependency, required_step, message in optional_dependencies:
        if optional_dependency is not None and required_step not in ordered:
            raise ValueError(message)


@dataclass
class QueryPrep:
    """An ordered pipeline of query-prep steps with their resolved dependencies.

    `process` runs the steps in order, threading the query through each and accumulating the edit
    log. An empty step list is an exact no-op (the processed query is byte-identical to the raw
    query), which is the off-by-default behavior the acceptance gate requires.
    """

    steps: tuple[str, ...] = ()
    vocabulary: "frozenset[str]" = field(default_factory=frozenset)
    glossary: Glossary | None = None
    rewriter: Rewriter | None = None
    hypothesizer: QueryGenerator | None = None
    decomposer: QueryGenerator | None = None
    known_word: KnownWordProbe | None = None
    context: VocabularyContext | None = None
    plausible: PlausibilityProbe | None = None

    @classmethod
    def build(
        cls,
        steps: Iterable[str],
        *,
        vocabulary: "frozenset[str] | None" = None,
        glossary: Glossary | None = None,
        rewriter: Rewriter | None = None,
        hypothesizer: QueryGenerator | None = None,
        decomposer: QueryGenerator | None = None,
        known_word: KnownWordProbe | None = None,
        context: VocabularyContext | None = None,
        plausible: PlausibilityProbe | None = None,
    ) -> "QueryPrep":
        """Validate step names and their required dependencies, then build the pipeline."""
        ordered = tuple(steps)
        _validate_steps(
            ordered,
            vocabulary=vocabulary,
            glossary=glossary,
            rewriter=rewriter,
            hypothesizer=hypothesizer,
            decomposer=decomposer,
            known_word=known_word,
            context=context,
            plausible=plausible,
        )
        return cls(
            steps=ordered,
            vocabulary=vocabulary if vocabulary is not None else frozenset(),
            glossary=glossary,
            rewriter=rewriter,
            hypothesizer=hypothesizer,
            decomposer=decomposer,
            known_word=known_word,
            context=context,
            plausible=plausible,
        )

    def process(self, query: str) -> QueryPrepResult:
        current = query
        edits: list[QueryEdit] = []
        rewrite_text: str | None = None
        hypothetical_answer: str | None = None
        decomposition: str | None = None
        subqueries: tuple[str, ...] = ()
        normalize_gate: LanguageGate | None = None
        for step in self.steps:
            if step == STEP_NORMALIZE:
                # With a plausibility probe wired in, decide transliteration for the whole query
                # so a foreign-language question is not mangled into unretrievable Cyrillic.
                normalize_gate = (
                    language_gate(current, self.plausible) if self.plausible is not None else None
                )
                current, step_edits = apply_normalize(current, gate=normalize_gate)
            elif step == STEP_TYPOS:
                # The edits accumulated so far carry each normalized token back to the form the
                # user typed, so candidate selection can refuse an incompatible restoration.
                current, step_edits = apply_typos(
                    current,
                    self.vocabulary,
                    known_word=self.known_word,
                    provenance=normalization_provenance(edits),
                    context=self.context,
                )
            elif step == STEP_GLOSSARY:
                assert self.glossary is not None  # guaranteed by build()
                current, step_edits = apply_glossary(current, self.glossary)
            elif step == STEP_REWRITE:
                assert self.rewriter is not None  # guaranteed by build()
                current, step_edits, rewrite_text = apply_rewrite(current, self.rewriter)
            elif step == STEP_HYDE:
                assert self.hypothesizer is not None  # guaranteed by build()
                hypothetical_answer, step_edits = apply_hyde(current, self.hypothesizer)
            else:  # STEP_DECOMPOSE
                assert self.decomposer is not None  # guaranteed by build()
                subqueries, step_edits, decomposition = apply_decompose(current, self.decomposer)
            edits.extend(step_edits)
        return QueryPrepResult(
            raw=query,
            processed=current,
            steps=self.steps,
            edits=tuple(edits),
            rewrite=rewrite_text,
            hypothetical_answer=hypothetical_answer,
            decomposition=decomposition,
            subqueries=subqueries,
            normalize_gate=normalize_gate,
        )
