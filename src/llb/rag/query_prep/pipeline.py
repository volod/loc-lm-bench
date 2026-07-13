"""The ordered query-prep pipeline that composes the individual steps into one pass."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from llb.rag.query_prep.base import (
    QUERY_PREP_STEPS,
    STEP_GLOSSARY,
    STEP_NORMALIZE,
    STEP_REWRITE,
    STEP_TYPOS,
    KnownWordProbe,
    QueryEdit,
    QueryPrepResult,
    Rewriter,
)
from llb.rag.query_prep.glossary import Glossary, apply_glossary
from llb.rag.query_prep.normalize import apply_normalize
from llb.rag.query_prep.rewrite import apply_rewrite
from llb.rag.query_prep.typos import apply_typos


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
    known_word: KnownWordProbe | None = None

    @classmethod
    def build(
        cls,
        steps: Iterable[str],
        *,
        vocabulary: "frozenset[str] | None" = None,
        glossary: Glossary | None = None,
        rewriter: Rewriter | None = None,
        known_word: KnownWordProbe | None = None,
    ) -> "QueryPrep":
        """Validate step names and their required dependencies, then build the pipeline."""
        ordered = tuple(steps)
        unknown = [step for step in ordered if step not in QUERY_PREP_STEPS]
        if unknown:
            raise ValueError(
                f"unknown query-prep step(s): {unknown}; choose from {list(QUERY_PREP_STEPS)}"
            )
        if len(set(ordered)) != len(ordered):
            raise ValueError(f"duplicate query-prep step(s): {ordered}")
        if STEP_TYPOS in ordered and vocabulary is None:
            raise ValueError("the 'typos' step needs a corpus vocabulary")
        if known_word is not None and STEP_TYPOS not in ordered:
            raise ValueError("the typo morphology guard needs the 'typos' step")
        if STEP_GLOSSARY in ordered and glossary is None:
            raise ValueError("the 'glossary' step needs a query glossary")
        if STEP_REWRITE in ordered and rewriter is None:
            raise ValueError("the 'rewrite' step needs a rewrite endpoint callable")
        return cls(
            steps=ordered,
            vocabulary=vocabulary if vocabulary is not None else frozenset(),
            glossary=glossary,
            rewriter=rewriter,
            known_word=known_word,
        )

    def process(self, query: str) -> QueryPrepResult:
        current = query
        edits: list[QueryEdit] = []
        rewrite_text: str | None = None
        for step in self.steps:
            if step == STEP_NORMALIZE:
                current, step_edits = apply_normalize(current)
            elif step == STEP_TYPOS:
                current, step_edits = apply_typos(
                    current, self.vocabulary, known_word=self.known_word
                )
            elif step == STEP_GLOSSARY:
                assert self.glossary is not None  # guaranteed by build()
                current, step_edits = apply_glossary(current, self.glossary)
            else:  # STEP_REWRITE
                assert self.rewriter is not None  # guaranteed by build()
                current, step_edits, rewrite_text = apply_rewrite(current, self.rewriter)
            edits.extend(step_edits)
        return QueryPrepResult(
            raw=query,
            processed=current,
            steps=self.steps,
            edits=tuple(edits),
            rewrite=rewrite_text,
        )
