"""Normalize step: matching-side casefold/apostrophe-unify plus Latin->Cyrillic transliteration.

The romanization table is injective, so the Latin->Cyrillic inverse is well-defined by
longest-key-first matching; the soft sign and a few rare digraph collisions (shch) are
intentionally out of the reversible core.
"""

import logging
import re

from llb.rag.query_prep.base import STEP_NORMALIZE, QueryEdit

_LOG = logging.getLogger(__name__)

# Reversible-ish Ukrainian romanization used to invert Latin-typed terms back to Cyrillic.
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
# Latin -> Cyrillic, matched longest-first so digraphs (`shch`, `kh`, `ye`) win over single letters.
LATIN_TO_CYRILLIC: dict[str, str] = {lat: cyr for cyr, lat in CYRILLIC_TO_LATIN.items()}
_LATIN_KEYS_LONGEST_FIRST: tuple[str, ...] = tuple(sorted(LATIN_TO_CYRILLIC, key=len, reverse=True))

# Characters dropped when romanizing (the soft sign and apostrophe variants have no Latin form).
_ROMANIZE_DROP = frozenset("ь'’ʼ")


def _is_latin_word(token: str) -> bool:
    """True for a non-empty token whose every character is an ASCII letter."""
    return bool(token) and all("a" <= ch <= "z" or "A" <= ch <= "Z" for ch in token)


def transliterate_latin_to_cyrillic(token: str) -> str:
    """Map a Latin-typed token to Cyrillic via the romanization table (longest-match, greedy).

    Characters with no table entry pass through unchanged, so a token that is not Latin-typed
    Ukrainian degrades to (mostly) itself. Case is folded first (the matching side never keeps
    case). Non-Latin tokens are returned unchanged by the caller.
    """
    lowered = token.casefold()
    out: list[str] = []
    i = 0
    while i < len(lowered):
        for key in _LATIN_KEYS_LONGEST_FIRST:
            if lowered.startswith(key, i):
                out.append(LATIN_TO_CYRILLIC[key])
                i += len(key)
                break
        else:
            out.append(lowered[i])
            i += 1
    return "".join(out)


def cyrillic_to_latin(text: str) -> str:
    """Romanize Ukrainian text with the reversible table (used to seed transliterated aliases).

    The soft sign and apostrophes are dropped so a seeded alias is clean Latin; any character with
    no table entry (a space, a digit) passes through unchanged.
    """
    return "".join(
        "" if ch in _ROMANIZE_DROP else CYRILLIC_TO_LATIN.get(ch, ch) for ch in text.casefold()
    )


def apply_normalize(query: str) -> tuple[str, list[QueryEdit]]:
    """Casefold + apostrophe-unify the whole query, then transliterate Latin-typed tokens.

    Casefolding and apostrophe unification are silent matching-side normalization (they never
    change which corpus terms match). Each Latin->Cyrillic transliteration is recorded as an edit
    because it is a real, auditable substitution.
    """
    from llb.rag.lexical import _APOSTROPHE_VARIANTS, _TOKEN_RE

    folded = query.translate(_APOSTROPHE_VARIANTS).casefold()
    edits: list[QueryEdit] = []

    def _replace(match: "re.Match[str]") -> str:
        token = match.group(0)
        if not _is_latin_word(token):
            return token
        cyrillic = transliterate_latin_to_cyrillic(token)
        if cyrillic == token:
            return token
        edits.append(
            QueryEdit(STEP_NORMALIZE, "transliterate", original=token, replacement=cyrillic)
        )
        _LOG.debug("[query-prep] transliterate %r -> %r", token, cyrillic)
        return cyrillic

    processed = _TOKEN_RE.sub(_replace, folded)
    return processed, edits
