"""Deterministic Ukrainian-output gate for ontology-drafted questions and answers."""

from llb.prep.ontology.constants import (
    UKRAINIAN_MIN_CYRILLIC_FRACTION,
    UKRAINIAN_SPECIFIC_LETTERS,
)


def is_ukrainian_dominant(text: str) -> bool:
    """Return whether text is Ukrainian-marked and Cyrillic-dominant.

    Ukrainian-specific letters distinguish the output from unmarked foreign text, while the
    Cyrillic share prevents a short Ukrainian wrapper around a foreign-language quotation from
    passing. Latin-script proper names can remain inside an otherwise Ukrainian answer.
    """
    folded = text.casefold()
    if not any(char in UKRAINIAN_SPECIFIC_LETTERS for char in folded):
        return False
    cyrillic = sum("\u0400" <= char <= "\u04ff" for char in folded)
    latin = sum("a" <= char <= "z" for char in folded)
    letters = cyrillic + latin
    return bool(letters) and cyrillic / letters >= UKRAINIAN_MIN_CYRILLIC_FRACTION
