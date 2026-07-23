"""Normalize Ukrainian punctuation, transliteration, and mixed-script homoglyphs."""

import logging
import re

from llb.rag.query_prep.base import (
    KIND_HOMOGLYPH,
    KIND_TRANSLITERATE,
    STEP_NORMALIZE,
    LanguageGate,
    PlausibilityProbe,
    QueryEdit,
)
from llb.scoring.security_cases import CYRILLIC_TO_LATIN_CONFUSABLES

_LOG = logging.getLogger(__name__)

# Below this share of a query's Latin word tokens decoding to plausible Ukrainian, the whole query
# is treated as foreign-language text and left untransliterated. Half is the natural split: a
# genuinely romanized Ukrainian query decodes (near-)entirely to known forms, while foreign text
# decodes to none, so the boundary is not delicate.
LANGUAGE_GATE_MIN_PLAUSIBLE_SHARE = 0.5

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
LATIN_ACRONYM_MAX_CHARS = 5

# Latin look-alikes that are safe to repair only inside a token that also contains Cyrillic.
LATIN_TO_UKRAINIAN_CONFUSABLES: dict[str, str] = {
    latin: cyrillic
    for cyrillic, latin in CYRILLIC_TO_LATIN_CONFUSABLES.items()
    if cyrillic in "авеікмнорстух"
}


def _is_latin_word(token: str) -> bool:
    """True for an ASCII romanized token, including internal disambiguation apostrophes."""
    parts = token.split("'")
    return bool(token) and all(
        part and all("a" <= ch <= "z" or "A" <= ch <= "Z" for ch in part) for part in parts
    )


def _decode_romanized_segment(segment: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(segment):
        for key in _LATIN_KEYS_LONGEST_FIRST:
            if segment.startswith(key, i):
                out.append(LATIN_TO_CYRILLIC[key])
                i += len(key)
                break
        else:
            out.append(segment[i])
            i += 1
    return "".join(out)


def transliterate_latin_to_cyrillic(token: str) -> str:
    """Map a Latin-typed token to Cyrillic via the romanization table (longest-match, greedy).

    Characters with no table entry pass through unchanged, so a token that is not Latin-typed
    Ukrainian degrades to (mostly) itself. Case is folded first (the matching side never keeps
    case). Non-Latin tokens are returned unchanged by the caller.
    """
    return "".join(_decode_romanized_segment(part) for part in token.casefold().split("'"))


def _romanize_cyrillic_run(text: str) -> str:
    segments: list[str] = []
    encoded: list[str] = []
    source: list[str] = []
    for char in text:
        if char in _ROMANIZE_DROP:
            continue
        code = CYRILLIC_TO_LATIN[char]
        candidate = "".join(encoded) + code
        if encoded and transliterate_latin_to_cyrillic(candidate) != "".join(source) + char:
            segments.append("".join(encoded))
            encoded = [code]
            source = [char]
        else:
            encoded.append(code)
            source.append(char)
    if encoded:
        segments.append("".join(encoded))
    return "'".join(segments)


def cyrillic_to_latin(text: str) -> str:
    """Romanize Ukrainian runs while preserving existing Latin terms and punctuation.

    The soft sign and apostrophes are dropped to model common Latin typing. Rare greedy digraph
    collisions inside a Cyrillic run receive an ASCII apostrophe separator; original Latin text
    never participates in that round-trip check.
    """
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if run:
            out.append(_romanize_cyrillic_run("".join(run)))
            run.clear()

    for raw_char in text:
        char = raw_char.casefold()
        if char in CYRILLIC_TO_LATIN or (char in _ROMANIZE_DROP and run):
            run.append(char)
        else:
            flush()
            out.append(raw_char)
    flush()
    return "".join(out)


def _repair_mixed_script(token: str) -> str:
    has_cyrillic = any("\u0400" <= char <= "\u04ff" for char in token)
    has_latin = any("a" <= char <= "z" for char in token)
    if not (has_cyrillic and has_latin):
        return token
    return "".join(LATIN_TO_UKRAINIAN_CONFUSABLES.get(char, char) for char in token)


def _transliteration_candidates(query: str) -> list[str]:
    """The casefolded Latin word tokens `apply_normalize` would try to transliterate.

    Short uppercase acronyms are excluded for the same reason the step leaves them Latin: `NP` is
    not romanized Ukrainian, so it neither romanizes nor speaks to whether the query is Ukrainian.
    """
    from llb.rag.lexical import _TOKEN_RE

    candidates: list[str] = []
    for raw in _TOKEN_RE.findall(query):
        token = raw.casefold()
        if not _is_latin_word(token):
            continue
        letters = raw.replace("'", "")
        if letters.isupper() and len(letters) <= LATIN_ACRONYM_MAX_CHARS:
            continue
        candidates.append(token)
    return candidates


def language_gate(
    query: str,
    plausible: PlausibilityProbe,
    threshold: float = LANGUAGE_GATE_MIN_PLAUSIBLE_SHARE,
) -> LanguageGate:
    """Decide whether the query as a whole should be transliterated to Cyrillic.

    Romanized Ukrainian decodes to tokens the corpus vocabulary or the morphology probe knows;
    foreign-language text (`what does the` -> `wгат доес тге`) decodes to Cyrillic nonsense the
    later restoration constraints correctly refuse to repair. When fewer than `threshold` of the
    query's Latin word tokens decode to a plausible Ukrainian form, the whole query is left
    untouched. A query with no Latin word tokens is transliterated vacuously -- there is nothing to
    gate, so homoglyph repair and Cyrillic passthrough behave exactly as before.
    """
    candidates = _transliteration_candidates(query)
    plausible_count = sum(
        1 for token in candidates if plausible(transliterate_latin_to_cyrillic(token))
    )
    total = len(candidates)
    transliterate = total == 0 or (plausible_count / total) >= threshold
    if not transliterate:
        _LOG.info(
            "[query-prep] normalize language gate refused %r: %d/%d Latin tokens decode to "
            "plausible Ukrainian (< %.2f)",
            query,
            plausible_count,
            total,
            threshold,
        )
    return LanguageGate(transliterate, total, plausible_count, threshold)


def apply_normalize(query: str, *, gate: LanguageGate | None = None) -> tuple[str, list[QueryEdit]]:
    """Casefold + apostrophe-unify the whole query, then transliterate Latin-typed tokens.

    Casefolding and apostrophe unification are silent matching-side normalization (they never
    change which corpus terms match). Each Latin->Cyrillic transliteration is recorded as an edit
    because it is a real, auditable substitution.

    When `gate` decided the query is not romanized Ukrainian, the query is returned untouched: per
    token transliteration would rewrite foreign text into unrecoverable Cyrillic. The gate stays
    off (None) unless the pipeline is wired with a plausibility probe, so a bare `apply_normalize`
    call transliterates unconditionally as before.
    """
    from llb.rag.lexical import _APOSTROPHE_VARIANTS, _TOKEN_RE

    if gate is not None and not gate.transliterate:
        return query, []

    folded = query.translate(_APOSTROPHE_VARIANTS)
    edits: list[QueryEdit] = []

    def _replace(match: "re.Match[str]") -> str:
        raw = match.group(0)
        token = raw.casefold()
        if _is_latin_word(token):
            letters = raw.replace("'", "")
            if letters.isupper() and len(letters) <= LATIN_ACRONYM_MAX_CHARS:
                return token
            replacement = transliterate_latin_to_cyrillic(token)
            kind = KIND_TRANSLITERATE
        else:
            replacement = _repair_mixed_script(token)
            kind = KIND_HOMOGLYPH
        if replacement == token:
            return token
        edits.append(QueryEdit(STEP_NORMALIZE, kind, original=token, replacement=replacement))
        _LOG.debug("[query-prep] %s %r -> %r", kind, token, replacement)
        return replacement

    processed = _TOKEN_RE.sub(_replace, folded)
    return processed, edits
