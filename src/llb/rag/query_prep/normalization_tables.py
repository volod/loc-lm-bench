"""Ukrainian romanization and mixed-script normalization tables."""

from llb.scoring.security_cases import CYRILLIC_TO_LATIN_CONFUSABLES

CYRILLIC_TO_LATIN: dict[str, str] = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "h",
    "ґ": "g",
    "д": "d",
    "е": "e",
    "є": "ye",
    "ж": "zh",
    "з": "z",
    "и": "y",
    "і": "i",
    "ї": "yi",
    "й": "j",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "c",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ю": "yu",
    "я": "ya",
}
LATIN_TO_CYRILLIC: dict[str, str] = {
    latin: cyrillic for cyrillic, latin in CYRILLIC_TO_LATIN.items()
}
LATIN_KEYS_LONGEST_FIRST: tuple[str, ...] = tuple(sorted(LATIN_TO_CYRILLIC, key=len, reverse=True))
ROMANIZE_DROP = frozenset("ь'’ʼ")
LATIN_ACRONYM_MAX_CHARS = 5
LATIN_TO_UKRAINIAN_CONFUSABLES: dict[str, str] = {
    latin: cyrillic
    for cyrillic, latin in CYRILLIC_TO_LATIN_CONFUSABLES.items()
    if cyrillic in "авеікмнорстух"
}
