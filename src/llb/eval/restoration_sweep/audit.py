"""Per-edit precision audit for the restoration constraint sweep.

A retrieval number alone cannot say whether a relaxed constraint RECOVERED the user's word or
merely rewrote their question into something the corpus happens to contain. The noisy queries this
sweep measures are generated from a clean question, so the word the user meant is known: aligning
the normalized noisy tokens with the normalized clean ones gives every correction a REFERENCE, and
each correction is then `correct` (it restored the clean token), `wrong` (it produced some other
token), or `unaligned` (the two token sequences do not correspond, so the audit refuses to judge).

The same alignment also gives the denominator the retrieval numbers cannot: an OPPORTUNITY is a
token the noise made out-of-vocabulary whose clean form the corpus does contain -- a correction the
constraints could have made. Precision over corrections and recall over opportunities are the two
halves of what a conservative constant costs.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from llb.rag.vector_store.lexical import tokenize
from llb.rag.query_prep.base import STEP_TYPOS, QueryEdit
from llb.rag.query_prep.typos import TYPO_MIN_TOKEN_CHARS

LABEL_CORRECT = "correct"
LABEL_WRONG = "wrong"
LABEL_UNALIGNED = "unaligned"


@dataclass(frozen=True)
class CaseAlignment:
    """Normalized noisy tokens paired with the clean tokens they were generated from.

    `pairs` is None when the two sequences have different lengths: every character-noise class
    rewrites characters in place, so a length mismatch means the normalization step merged or split
    a token and no per-token reference can be trusted.
    """

    pairs: tuple[tuple[str, str], ...] | None

    @classmethod
    def build(cls, clean_normalized: str, variant_normalized: str) -> "CaseAlignment":
        clean = tokenize(clean_normalized)
        variant = tokenize(variant_normalized)
        if len(clean) != len(variant):
            return cls(None)
        return cls(tuple(zip(variant, clean)))

    @property
    def references(self) -> dict[str, str]:
        """Noisy token -> the clean token it came from, dropping any token with two origins."""
        if self.pairs is None:
            return {}
        origins: dict[str, set[str]] = {}
        for noisy, clean in self.pairs:
            origins.setdefault(noisy, set()).add(clean)
        return {noisy: next(iter(forms)) for noisy, forms in origins.items() if len(forms) == 1}

    def opportunities(self, vocabulary: "frozenset[str]") -> tuple[tuple[str, str], ...]:
        """Noised tokens the typo step could restore: out of vocabulary, clean form in it."""
        if self.pairs is None:
            return ()
        return tuple(
            (noisy, clean)
            for noisy, clean in self.pairs
            if noisy != clean
            and noisy not in vocabulary
            and clean in vocabulary
            and not noisy.isdigit()
            and len(noisy.replace("'", "")) >= TYPO_MIN_TOKEN_CHARS
        )


@dataclass(frozen=True)
class EditRecord:
    """One correction the typo step made, with the clean token it should have produced."""

    setting: str
    variant_class: str
    item_id: str
    original: str
    replacement: str
    reference: str | None
    label: str
    reference_in_vocabulary: bool

    def as_row(self) -> dict[str, object]:
        return {
            "setting": self.setting,
            "variant_class": self.variant_class,
            "item_id": self.item_id,
            "original": self.original,
            "replacement": self.replacement,
            "reference": self.reference,
            "label": self.label,
            "reference_in_vocabulary": self.reference_in_vocabulary,
        }


@dataclass(frozen=True)
class AuditCounts:
    """Correction and opportunity tallies, summable across items and noise classes."""

    corrections: int = 0
    correct: int = 0
    wrong: int = 0
    unaligned: int = 0
    opportunities: int = 0
    restored: int = 0

    def __add__(self, other: "AuditCounts") -> "AuditCounts":
        return AuditCounts(
            corrections=self.corrections + other.corrections,
            correct=self.correct + other.correct,
            wrong=self.wrong + other.wrong,
            unaligned=self.unaligned + other.unaligned,
            opportunities=self.opportunities + other.opportunities,
            restored=self.restored + other.restored,
        )

    @property
    def labeled(self) -> int:
        """Corrections the alignment could judge; an `unaligned` one is not evidence either way."""
        return self.correct + self.wrong

    @property
    def wrong_share(self) -> float:
        """Share of judged corrections that produced a token the user did not type."""
        return self.wrong / self.labeled if self.labeled else 0.0

    @property
    def restoration_recall(self) -> float:
        """Share of restorable noised tokens the constraints actually restored."""
        return self.restored / self.opportunities if self.opportunities else 0.0

    def as_row(self) -> dict[str, object]:
        return {
            "corrections": self.corrections,
            "correct": self.correct,
            "wrong": self.wrong,
            "unaligned": self.unaligned,
            "labeled": self.labeled,
            "wrong_share": round(self.wrong_share, 4),
            "opportunities": self.opportunities,
            "restored": self.restored,
            "restoration_recall": round(self.restoration_recall, 4),
        }


def typo_edits(edits: Iterable[QueryEdit]) -> list[QueryEdit]:
    """Only the vocabulary corrections; the normalize step's edits are not restorations."""
    return [edit for edit in edits if edit.step == STEP_TYPOS]


def audit_case(
    *,
    setting: str,
    variant_class: str,
    item_id: str,
    edits: Sequence[QueryEdit],
    alignment: CaseAlignment,
    vocabulary: "frozenset[str]",
) -> tuple[list[EditRecord], AuditCounts]:
    """Label one case's corrections and count the restorations it did and did not make."""
    references = alignment.references
    records: list[EditRecord] = []
    correct = wrong = unaligned = 0
    corrections = typo_edits(edits)
    for edit in corrections:
        reference = references.get(edit.original)
        if reference is None:
            label = LABEL_UNALIGNED
            unaligned += 1
        elif reference == edit.replacement:
            label = LABEL_CORRECT
            correct += 1
        else:
            label = LABEL_WRONG
            wrong += 1
        records.append(
            EditRecord(
                setting=setting,
                variant_class=variant_class,
                item_id=item_id,
                original=edit.original,
                replacement=edit.replacement,
                reference=reference,
                label=label,
                reference_in_vocabulary=reference is not None and reference in vocabulary,
            )
        )
    made = {(edit.original, edit.replacement) for edit in corrections}
    opportunities = alignment.opportunities(vocabulary)
    counts = AuditCounts(
        corrections=len(corrections),
        correct=correct,
        wrong=wrong,
        unaligned=unaligned,
        opportunities=len(opportunities),
        restored=sum(pair in made for pair in opportunities),
    )
    return records, counts
