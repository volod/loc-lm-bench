"""Answer-side gold-span coverage: did the ANSWER carry each gold span's fact?

The retrieval side distinguishes "carried one hop" from "carried both"
(`llb.rag.retrieval.span_coverage_at_k` / `all_spans_at_k`). The answer side had no such
distinction: `objective_score` is reference-answer token F1, so a two-hop answer that states one
fact fluently and omits the other scores roughly half -- the same value a vague answer touching
both facts earns. This module is the answer-side counterpart, one reading per gold span.

What a span REQUIRES is not its whole text. A gold span is a passage; the fact it contributes is
the part the reference answer restates and the part the question did not already supply:

    grounded(span)    = content of (span text) AND content of (reference answer)
    distinctive(span) = grounded(span) MINUS question content MINUS every OTHER span's content
    required(span)    = distinctive(span), or grounded(span) when that leaves nothing to judge

Each subtraction removes a way of scoring a fact the model never supplied. Without the reference
intersection, a correct one-line answer reads 0.05 against the registry paragraph that grounds it
-- a labeled span is routinely several times longer than the answer. Without the question
subtraction, naming a hop's subject carries the hop ("the trademark certificate - unknown"
reproduces most of that span's wording and none of its fact). Without the sibling-span
subtraction, vocabulary the two hops share -- units, dates, the shared subject -- lets one hop's
answer satisfy the other. The fallback is what keeps the subtractions safe on a literal ledger,
where the reference often restates the question verbatim and nothing distinctive is left: 42% of
the spans on the drafted goods ledger are judged on `grounded` for exactly that reason.

A span counts as CARRIED when the answer holds at least `SPAN_CARRIED_MIN_SHARE` of its required
terms AND every required numeral: an answer that restates a table fact with the wrong number has
not carried it, however much of the surrounding wording it reproduces.

Read the pair beside the objective, never instead of it. Coverage is a RECALL-side reading, so a
model that dumps its whole context scores 1.0 here by construction; `token_precision` /
`ranking_score` in the same row are what price that. The point of the pair is the case the
objective cannot express on its own -- both facts present, stated tersely or in the model's own
words, low token F1.
"""

from collections.abc import Callable, Sequence
from typing import NamedTuple

from llb.core.contracts.rag import AnswerSpanScores, SourceSpanRecord
from llb.scoring.correctness import normalize
from llb.scoring.function_words import FUNCTION_WORDS

# One surface token -> its lemma. Injectable so the metric is unit-testable without an analyzer;
# the default is the same Ukrainian lemmatizer the lexical index uses, so a form the retrieval
# side matches is a form the answer side matches too.
Lemmatizer = Callable[[str], str]

# Share of a span's required terms an answer must hold for the span to count as carried. The
# reading is bimodal on measured runs -- an answer either restates a hop or does not -- so the cut
# sits in the empty middle: over the 588 span readings of the recorded multi-hop bundles, 76% are
# exactly 0.0 or 1.0 and only 10% fall anywhere near the cut.
SPAN_CARRIED_MIN_SHARE = 0.5

# What an item with no labeled span (or no term-level requirement) scores. It mirrors the
# retrieval side, where an item labeling no span is vacuously covered: there was nothing to carry,
# so the reading is not evidence of a failure to carry it. `answer_spans_measured` is what tells a
# vacuous 1.0 from a carried one.
VACUOUS_COVERAGE = 1.0


class TermSets(NamedTuple):
    """One text as content lemmas plus numerals.

    Numerals are kept apart because they are matched literally -- a lemmatizer has nothing to say
    about `4001`, and a fact's numbers are the part a paraphrase may not vary.
    """

    lemmas: frozenset[str]
    numerals: frozenset[str]

    @property
    def all_terms(self) -> frozenset[str]:
        return self.lemmas | self.numerals


class SpanRequirement(NamedTuple):
    """What one gold span requires the answer to hold, and which of those terms are numerals.

    `distinctive` records which of the two definitions above produced it, so a run can be audited
    for how often the fallback fired without recomputing anything.
    """

    terms: frozenset[str]
    numerals: frozenset[str]
    distinctive: bool = True

    def __bool__(self) -> bool:
        return bool(self.terms)


def _is_numeral(token: str) -> bool:
    return any(character.isdigit() for character in token)


def term_sets(text: str, lemmatize: Lemmatizer) -> TermSets:
    """`text` as content lemmas and numerals over the SCORING tokenizer, so both sides tokenize
    alike.

    `llb.scoring.correctness.normalize` drops punctuation, which splits an apostrophe form
    (`пам'ятка`) into two pieces. That is applied to the answer and to the span identically, so the
    pieces still match each other; it costs a little lemma quality, never a comparison.
    """
    lemmas: set[str] = set()
    numerals: set[str] = set()
    for token in normalize(text).split():
        if _is_numeral(token):
            numerals.add(token)
        elif token not in FUNCTION_WORDS and (lemma := lemmatize(token)) not in FUNCTION_WORDS:
            lemmas.add(lemma)
    return TermSets(frozenset(lemmas), frozenset(numerals))


def span_requirements(
    spans: Sequence[SourceSpanRecord],
    reference: str,
    question: str = "",
    lemmatize: Lemmatizer | None = None,
) -> list[SpanRequirement]:
    """One requirement per labeled span -- item-level because each span's requirement subtracts
    what its SIBLINGS say, which is what stops one hop's answer from satisfying the other."""
    lemmatize = lemmatize or default_lemmatizer()
    reference_terms = term_sets(reference, lemmatize)
    given = term_sets(question, lemmatize).all_terms
    texts = [term_sets(str(span.get("text", "")), lemmatize) for span in spans]
    requirements = []
    for index, span in enumerate(texts):
        grounded = (span.lemmas & reference_terms.lemmas) | (
            span.numerals & reference_terms.numerals
        )
        elsewhere = frozenset().union(
            *(other.all_terms for position, other in enumerate(texts) if position != index),
        )
        distinctive = grounded - given - elsewhere
        required = distinctive or grounded
        requirements.append(
            SpanRequirement(
                frozenset(required),
                frozenset(required & span.numerals),
                distinctive=bool(distinctive),
            )
        )
    return requirements


def carried_share(answer: str, required: SpanRequirement, lemmatize: Lemmatizer) -> float:
    """Share of a span's required terms the answer holds; 0.0 on a missing required numeral."""
    if not required:
        return 0.0
    present = term_sets(answer, lemmatize).all_terms
    if not required.numerals <= present:
        return 0.0
    return len(required.terms & present) / len(required.terms)


def span_carried(answer: str, required: SpanRequirement, lemmatize: Lemmatizer) -> bool:
    """True when the answer holds enough of this span's fact to count as stating it."""
    return carried_share(answer, required, lemmatize) >= SPAN_CARRIED_MIN_SHARE


def answer_span_scores(
    answer: str,
    spans: Sequence[SourceSpanRecord],
    reference: str,
    question: str = "",
    lemmatize: Lemmatizer | None = None,
) -> AnswerSpanScores:
    """Per-item answer-side coverage of the labeled spans: the graded share and the all-spans gate.

    A span with no requirement at all -- the reference restates it in words the span never uses --
    cannot be judged and is left out of the denominator rather than counted as a miss. An item
    where NO span can be judged reads `VACUOUS_COVERAGE`, exactly as the retrieval side reads an
    item that labels no span.
    """
    lemmatize = lemmatize or default_lemmatizer()
    judged = [
        span_carried(answer, required, lemmatize)
        for required in span_requirements(spans, reference, question, lemmatize)
        if required
    ]
    if not judged:
        return {
            "answer_span_coverage": VACUOUS_COVERAGE,
            "answer_all_spans": VACUOUS_COVERAGE,
            "answer_spans_measured": 0,
        }
    return {
        "answer_span_coverage": sum(1.0 for carried in judged if carried) / len(judged),
        "answer_all_spans": 1.0 if all(judged) else 0.0,
        "answer_spans_measured": len(judged),
    }


def default_lemmatizer() -> Lemmatizer:
    """The pinned Ukrainian lemmatizer, resolved lazily so importing this module stays cheap."""
    from llb.rag.vector_store.lexical import ukrainian_lemma

    return ukrainian_lemma


__all__ = [
    "SPAN_CARRIED_MIN_SHARE",
    "VACUOUS_COVERAGE",
    "Lemmatizer",
    "SpanRequirement",
    "TermSets",
    "answer_span_scores",
    "carried_share",
    "default_lemmatizer",
    "span_carried",
    "span_requirements",
]
