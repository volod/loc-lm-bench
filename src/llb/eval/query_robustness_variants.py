"""Seeded deterministic Ukrainian query-noise generators."""

import hashlib
import math
import random

from llb.rag.query_prep.normalize import cyrillic_to_latin
from llb.scoring.security_cases import CYRILLIC_TO_LATIN_CONFUSABLES

TRANSLITERATION = "transliteration"
MIXED_SCRIPT = "apostrophe_mixed_script"
KEYBOARD_TYPOS = "keyboard_typos"
VARIANT_CLASSES = (TRANSLITERATION, MIXED_SCRIPT, KEYBOARD_TYPOS)

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


def _rng(seed: int, item_id: str, variant_class: str) -> random.Random:
    material = f"{seed}\0{item_id}\0{variant_class}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def _selected_positions(positions: list[int], rate: float, rng: random.Random) -> list[int]:
    if not positions or rate <= 0:
        return []
    count = min(len(positions), max(1, math.ceil(len(positions) * rate)))
    return sorted(rng.sample(positions, count))


def _mixed_script(text: str, rate: float, rng: random.Random) -> str:
    chars = list(text)
    positions = [
        index for index, char in enumerate(chars) if char.casefold() in _UKRAINIAN_CONFUSABLES
    ]
    for index in _selected_positions(positions, rate, rng):
        replacement = _UKRAINIAN_CONFUSABLES[chars[index].casefold()]
        chars[index] = replacement.upper() if chars[index].isupper() else replacement
    for index, char in enumerate(chars):
        if char in _APOSTROPHES:
            chars[index] = rng.choice(APOSTROPHE_NOISE)
    return "".join(chars)


def _keyboard_typos(text: str, rate: float, rng: random.Random) -> str:
    chars = list(text)
    positions = [index for index, char in enumerate(chars) if char.casefold() in KEYBOARD_NEIGHBORS]
    for index in _selected_positions(positions, rate, rng):
        replacement = rng.choice(KEYBOARD_NEIGHBORS[chars[index].casefold()])
        chars[index] = replacement.upper() if chars[index].isupper() else replacement
    return "".join(chars)


def generate_variant(
    question: str,
    variant_class: str,
    *,
    item_id: str,
    seed: int,
    typo_rate: float,
) -> str:
    """Generate one stable noisy query for an item and noise class."""
    if not 0 <= typo_rate <= 1:
        raise ValueError("typo_rate must be between 0 and 1")
    if variant_class == TRANSLITERATION:
        return cyrillic_to_latin(question)
    rng = _rng(seed, item_id, variant_class)
    if variant_class == MIXED_SCRIPT:
        return _mixed_script(question, typo_rate, rng)
    if variant_class == KEYBOARD_TYPOS:
        return _keyboard_typos(question, typo_rate, rng)
    raise ValueError(f"unknown query robustness variant class: {variant_class!r}")
