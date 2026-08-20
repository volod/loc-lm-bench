"""Closed question-type taxonomy for drafted gold items (yield-max).

Every drafted item is tagged with one of a small, closed set of question types so reviewers and
miss-analyzers can filter drafts (e.g. "how do numeric questions retrieve vs definitions?"). The
classifier is a pure, deterministic Ukrainian-first keyword heuristic over the drafted question and
its reference answer -- no model call -- so the label is reproducible and unit-testable. Multi-hop
items are tagged directly by the graph-path drafter (`QUESTION_TYPE_MULTI_HOP`); this module labels
the single-span flat drafts.
"""

import re

from llb.prep.ontology.constants import (
    QUESTION_TYPE_COMPARATIVE,
    QUESTION_TYPE_DEFINITION,
    QUESTION_TYPE_FACTOID,
    QUESTION_TYPE_NUMERIC,
    QUESTION_TYPE_PROCEDURAL,
)

# A reference answer that is (mostly) digits, a year, a percentage, or a measured quantity ->
# numeric. Keep the pattern liberal: any run of >=1 digit inside a short answer counts.
_DIGIT = re.compile(r"\d")
_MOSTLY_NUMERIC = re.compile(r"^[\d\s.,:%/x×°-]+$")

# Ukrainian (+ a few English) cue phrases per type. Matched on the casefolded question; the order
# below is the resolution priority when several cues fire (definition and procedural before the
# generic factoid; numeric is decided from the answer shape first).
_DEFINITION_CUES = (
    "що таке",
    "що означає",
    "хто такий",
    "хто така",
    "яке визначення",
    "чим є",
    "що це",
    "what is",
    "define",
)
_PROCEDURAL_CUES = (
    "як ",
    "як?",
    "яким чином",
    "у який спосіб",
    "яка процедура",
    "які кроки",
    "що потрібно зробити",
    "how ",
    "how to",
)
_COMPARATIVE_CUES = (
    "порівн",
    "чим відрізня",
    "яка різниця",
    "більш",
    "менш",
    "краще",
    "гірше",
    "на відміну",
    "compared",
    "difference between",
)
_NUMERIC_CUES = (
    "скільки",
    "у якому році",
    "в якому році",
    "коли",
    "яка кількість",
    "який відсоток",
    "how many",
    "how much",
    "what year",
)


def _answer_is_numeric(reference_answer: str) -> bool:
    text = reference_answer.strip()
    if not text:
        return False
    if _MOSTLY_NUMERIC.match(text):
        return True
    # a short answer that is dominated by digits (e.g. "1256 рік", "18 відсотків")
    if len(text) <= 24 and _DIGIT.search(text):
        digits = sum(ch.isdigit() for ch in text)
        return digits >= 2
    return False


def _has_cue(text: str, cues: tuple[str, ...]) -> bool:
    return any(cue in text for cue in cues)


def classify_question_type(question: str, reference_answer: str) -> str:
    """Label a drafted (question, reference_answer) pair with a closed question type.

    Resolution priority: definition and procedural cue phrases win over the generic factoid;
    a comparative cue wins over a bare numeric cue; a numeric ANSWER shape or a numeric cue marks
    numeric; everything else is a factoid. Deterministic and dependency-free.
    """
    q = " ".join(question.split()).casefold()
    if not q:
        return QUESTION_TYPE_FACTOID
    if _has_cue(q, _DEFINITION_CUES):
        return QUESTION_TYPE_DEFINITION
    if _has_cue(q, _COMPARATIVE_CUES):
        return QUESTION_TYPE_COMPARATIVE
    if _answer_is_numeric(reference_answer) or _has_cue(q, _NUMERIC_CUES):
        return QUESTION_TYPE_NUMERIC
    if _has_cue(q, _PROCEDURAL_CUES):
        return QUESTION_TYPE_PROCEDURAL
    return QUESTION_TYPE_FACTOID
