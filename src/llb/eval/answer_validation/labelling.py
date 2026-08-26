"""Was a refused answer CORRECT? -- read from a signal that survives Ukrainian inflection.

The catch / false-rejection split is the whole reading of a validator, and it rests on an
automated correctness proxy. The shipped proxy is `contains`: every reference token appears
somewhere in the answer, over case- and punctuation-folded SURFACE tokens. That proxy has no
morphology, and Ukrainian references are inflected, so a correct short answer to a question whose
reference is in an oblique case reads as wrong -- and a rejection of it is then filed as a CATCH,
the most flattering possible label for the gate. On the one heavy run recorded so far, the single
catch was exactly this artefact.

Two more readings are added here, both over the lemmatizer the lexical index and the answer-span
scorer already use, so a form retrieval matches is a form this label matches. A term is carried by
its SURFACE FORM AND its lemma together, and two terms match when those sets intersect -- the
analyzer's first parse is not stable across inflections of one word (`роботою` normalizes to
`робота`, `робота` to `робот`, the robot), which is the same artefact `llb.scoring.function_words`
lists both forms for.

  - `contains_lemma` -- `contains`, matched on those key sets. The same claim the shipped proxy
    makes, made in a way inflection cannot break.
  - `answer_within_reference` -- every CONTENT term the answer states is in the reference. The
    TERSE reading: an answer that says only what the reference says, in fewer words, is not a
    wrong answer. It is what `Вишивка.` against `роботою вишивки` needs, and no containment in the
    other direction can express it.

A refusal is a FALSE REJECTION when ANY of the three says the answer was correct, and the label
records WHICH fired -- so the reading is auditable rather than a second opaque proxy replacing the
first.

**No embedding threshold decides a label.** `--score-semantic` records a cosine per case and the
refusal table reports it, because a reader adjudicating a rejection wants it; but a cosine cut
that separates "paraphrase" from "different answer" is not something this corpus establishes (the
repo's own near-duplicate work reads 0.9 as barely above noise), and a label nobody can reproduce
without the pinned embedder is a worse proxy than the two deterministic ones above.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llb.eval import common as eval_common
from llb.eval.answer_validation.constants import REFERENCE_CORRECT_COLUMN
from llb.eval.answer_validation.equivalence import Lemmatizer, default_lemmatizer
from llb.eval.paired_cases import CaseRows
from llb.scoring.correctness import normalize
from llb.scoring.function_words import FUNCTION_WORDS

LABEL_CATCH = "catch"
LABEL_FALSE_REJECTION = "false_rejection"

SIGNAL_CONTAINS = REFERENCE_CORRECT_COLUMN
SIGNAL_CONTAINS_LEMMA = "contains_lemma"
SIGNAL_ANSWER_WITHIN_REFERENCE = "answer_within_reference"
CORRECTNESS_SIGNALS: tuple[str, ...] = (
    SIGNAL_CONTAINS,
    SIGNAL_CONTAINS_LEMMA,
    SIGNAL_ANSWER_WITHIN_REFERENCE,
)

# The recorded answer text a re-labelling reads. It is a PREVIEW, capped when the row was written,
# which is why every added signal is containment-shaped: a truncated answer can only fail a
# containment test that the full answer would have passed, so the cap can lose a false rejection
# and can never invent one.
ANSWER_COLUMN = "answer_preview"


@dataclass(frozen=True, slots=True)
class RefusalLabel:
    """What one refusal is called, which signals said so, and what the shipped proxy alone said."""

    label: str
    signals: tuple[str, ...]
    shipped_label: str

    @property
    def correct(self) -> bool:
        return self.label == LABEL_FALSE_REJECTION

    @property
    def relabelled(self) -> bool:
        """Whether the added signals moved this refusal off the label `contains` alone gave it."""
        return self.label != self.shipped_label


def term_keys(token: str, lemmatize: Lemmatizer) -> frozenset[str]:
    """One token as the pair of forms it may be matched by: as written, and as a lemma."""
    return frozenset({token, lemmatize(token)})


def terms(text: str, lemmatize: Lemmatizer, *, content_only: bool = False) -> list[frozenset[str]]:
    """The text's terms over the SCORING tokenizer, one key set each.

    `content_only` drops a token whose surface form OR lemma is a function word, exactly as
    `llb.scoring.answer_spans` does, so an unlisted inflection is still caught by its lemma.
    """
    out = []
    for token in normalize(text).split():
        keys = term_keys(token, lemmatize)
        if content_only and keys & FUNCTION_WORDS:
            continue
        out.append(keys)
    return out


def _pool(text: str, lemmatize: Lemmatizer, *, content_only: bool = False) -> set[str]:
    return {key for keys in terms(text, lemmatize, content_only=content_only) for key in keys}


def contains_lemma(answer: str, reference: str, lemmatize: Lemmatizer) -> bool:
    """`contains`, over lemmas: every reference term is in the answer whatever case it was in."""
    required = terms(reference, lemmatize)
    if not required:
        return False
    stated = _pool(answer, lemmatize)
    return all(keys & stated for keys in required)


def answer_within_reference(answer: str, reference: str, lemmatize: Lemmatizer) -> bool:
    """Every content term the answer states is in the reference, and at least one is shared.

    The terse reading: an answer that says only what the reference says, in fewer words, is not a
    wrong answer. Requiring a shared term is what keeps it from passing an answer made entirely of
    function words, or an empty one.
    """
    stated = terms(answer, lemmatize, content_only=True)
    referenced = _pool(reference, lemmatize)
    if not stated or not referenced:
        return False
    return all(keys & referenced for keys in stated)


def label_refusal(
    row: Mapping[str, Any], reference: str | None, lemmatize: Lemmatizer
) -> RefusalLabel:
    """Label one refused case: catch, or false rejection, with the signals that decided it."""
    answer = str(row.get(ANSWER_COLUMN, "") or "")
    fired: list[str] = []
    if float(row.get(SIGNAL_CONTAINS, 0.0) or 0.0) >= 1.0:
        fired.append(SIGNAL_CONTAINS)
    if reference:
        if contains_lemma(answer, reference, lemmatize):
            fired.append(SIGNAL_CONTAINS_LEMMA)
        if answer_within_reference(answer, reference, lemmatize):
            fired.append(SIGNAL_ANSWER_WITHIN_REFERENCE)
    return RefusalLabel(
        label=LABEL_FALSE_REJECTION if fired else LABEL_CATCH,
        signals=tuple(fired),
        shipped_label=LABEL_FALSE_REJECTION if SIGNAL_CONTAINS in fired else LABEL_CATCH,
    )


def label_refusals(
    rows: CaseRows,
    references: Mapping[str, str] | None = None,
    lemmatize: Lemmatizer | None = None,
) -> dict[str, RefusalLabel]:
    """Every refused case of the gated lane, keyed by item id.

    Only refusals are labelled: a case the gate let through is scored by the run, not by a proxy,
    and labelling it here would invent a second correctness verdict for it.
    """
    refused = [
        row
        for row in rows
        if str(row.get("status")) == eval_common.ONTOLOGY_VIOLATION and row.get("item_id")
    ]
    if not refused:
        return {}
    lookup = dict(references or {})
    lemmatizer = lemmatize or default_lemmatizer()
    return {
        str(row["item_id"]): label_refusal(row, lookup.get(str(row["item_id"])), lemmatizer)
        for row in refused
    }


def relabelled(labels: Mapping[str, RefusalLabel]) -> Sequence[str]:
    """The item ids whose label the added signals moved, in item order -- the delta to report."""
    return sorted(item_id for item_id, label in labels.items() if label.relabelled)


__all__ = [
    "ANSWER_COLUMN",
    "CORRECTNESS_SIGNALS",
    "LABEL_CATCH",
    "LABEL_FALSE_REJECTION",
    "SIGNAL_ANSWER_WITHIN_REFERENCE",
    "SIGNAL_CONTAINS",
    "SIGNAL_CONTAINS_LEMMA",
    "RefusalLabel",
    "answer_within_reference",
    "contains_lemma",
    "term_keys",
    "terms",
    "label_refusal",
    "label_refusals",
    "relabelled",
]
