"""Seeded deterministic Ukrainian query-noise generators."""

import hashlib
import math
import random
from collections.abc import Sequence

from llb.rag.query_prep.normalize import cyrillic_to_latin
from llb.scoring.security_cases import CYRILLIC_TO_LATIN_CONFUSABLES

TRANSLITERATION = "transliteration"
APOSTROPHE_VARIANT = "apostrophe_variant"
MIXED_SCRIPT = "mixed_script"
APOSTROPHE_MIXED_SCRIPT = "apostrophe_mixed_script"
KEYBOARD_TYPOS = "keyboard_typos"

# One class per mechanism, so a mitigation lane's recovery is attributable. The combined
# `apostrophe_mixed_script` class stays available for a measurement that explicitly wants both
# mechanisms at once, but it is not part of the default set: its apostrophe half is invisible
# whenever the homoglyph half dominates.
VARIANT_CLASSES = (TRANSLITERATION, APOSTROPHE_VARIANT, MIXED_SCRIPT, KEYBOARD_TYPOS)
COMBINED_VARIANT_CLASSES = (APOSTROPHE_MIXED_SCRIPT,)
ALL_VARIANT_CLASSES = VARIANT_CLASSES + COMBINED_VARIANT_CLASSES

APOSTROPHE_NOISE = ("`", "‘", "’", "ʼ")
_APOSTROPHES = frozenset(("'", *APOSTROPHE_NOISE))
_UKRAINIAN_CONFUSABLES = {
    cyrillic: latin
    for cyrillic, latin in CYRILLIC_TO_LATIN_CONFUSABLES.items()
    if cyrillic in "авеікмнорстух"
}

# Horizontal keyboard neighbors on the standard Ukrainian layout. Substitutions remain Cyrillic
# and preserve the word length, isolating realistic adjacent-key noise from insertion/deletion.
_KEYBOARD_ROWS = ("йцукенгшщзхї", "фівапролджє", "ячсмитьбю")
KEYBOARD_NEIGHBORS: dict[str, tuple[str, ...]] = {}
for _row in _KEYBOARD_ROWS:
    for _index, _char in enumerate(_row):
        _neighbors = _row[max(0, _index - 1) : _index] + _row[_index + 1 : _index + 2]
        KEYBOARD_NEIGHBORS[_char] = tuple(_neighbors)


def _rng(seed: int, item_id: str, mechanism: str) -> random.Random:
    material = f"{seed}\0{item_id}\0{mechanism}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def _selected_positions(positions: list[int], rate: float, rng: random.Random) -> list[int]:
    if not positions or rate <= 0:
        return []
    count = min(len(positions), max(1, math.ceil(len(positions) * rate)))
    return sorted(rng.sample(positions, count))


def _apostrophe_variants(text: str, rng: random.Random) -> str:
    """Re-type every apostrophe as a DIFFERENT variant, so the class always perturbs the query."""
    chars = list(text)
    for index, char in enumerate(chars):
        if char in _APOSTROPHES:
            chars[index] = rng.choice([variant for variant in APOSTROPHE_NOISE if variant != char])
    return "".join(chars)


def _mixed_script(text: str, rate: float, rng: random.Random) -> str:
    chars = list(text)
    positions = [
        index for index, char in enumerate(chars) if char.casefold() in _UKRAINIAN_CONFUSABLES
    ]
    for index in _selected_positions(positions, rate, rng):
        replacement = _UKRAINIAN_CONFUSABLES[chars[index].casefold()]
        chars[index] = replacement.upper() if chars[index].isupper() else replacement
    return "".join(chars)


def _keyboard_typos(text: str, rate: float, rng: random.Random) -> str:
    chars = list(text)
    positions = [index for index, char in enumerate(chars) if char.casefold() in KEYBOARD_NEIGHBORS]
    for index in _selected_positions(positions, rate, rng):
        replacement = rng.choice(KEYBOARD_NEIGHBORS[chars[index].casefold()])
        chars[index] = replacement.upper() if chars[index].isupper() else replacement
    return "".join(chars)


def resolve_variant_classes(classes: Sequence[str] | None) -> tuple[str, ...]:
    """Validate and de-duplicate a requested class selection, preserving the caller's order."""
    if classes is None:
        return VARIANT_CLASSES
    resolved: list[str] = []
    for variant_class in classes:
        if variant_class not in ALL_VARIANT_CLASSES:
            known = ", ".join(ALL_VARIANT_CLASSES)
            raise ValueError(f"unknown query robustness variant class: {variant_class!r} ({known})")
        if variant_class not in resolved:
            resolved.append(variant_class)
    if not resolved:
        raise ValueError("at least one query robustness variant class is required")
    return tuple(resolved)


def parse_variant_classes(spec: str) -> tuple[str, ...]:
    """Parse a comma-separated class selection, de-duplicated in the order given."""
    return resolve_variant_classes([token.strip() for token in spec.split(",") if token.strip()])


def generate_variant(
    question: str,
    variant_class: str,
    *,
    item_id: str,
    seed: int,
    typo_rate: float,
) -> str:
    """Generate one stable noisy query for an item and noise class.

    Each mechanism draws from its own seeded stream and rewrites characters in place, so the
    combined `apostrophe_mixed_script` text is exactly the composition of the two single-mechanism
    classes on the same item and seed -- the split loses no comparability with the combined class.
    """
    if not 0 <= typo_rate <= 1:
        raise ValueError("typo_rate must be between 0 and 1")
    if variant_class == TRANSLITERATION:
        return cyrillic_to_latin(question)
    if variant_class == KEYBOARD_TYPOS:
        return _keyboard_typos(question, typo_rate, _rng(seed, item_id, KEYBOARD_TYPOS))
    if variant_class not in (APOSTROPHE_VARIANT, MIXED_SCRIPT, APOSTROPHE_MIXED_SCRIPT):
        raise ValueError(f"unknown query robustness variant class: {variant_class!r}")
    text = question
    if variant_class in (MIXED_SCRIPT, APOSTROPHE_MIXED_SCRIPT):
        text = _mixed_script(text, typo_rate, _rng(seed, item_id, MIXED_SCRIPT))
    if variant_class in (APOSTROPHE_VARIANT, APOSTROPHE_MIXED_SCRIPT):
        text = _apostrophe_variants(text, _rng(seed, item_id, APOSTROPHE_VARIANT))
    return text
