"""Restore the raw query's letter case onto the processed query for the dense lane.

`normalize` casefolds the WHOLE query because the lexical lane matches on a folded surface form
(`llb.rag.vector_store.lexical.normalize_token`), but the dense encoder is case-sensitive: an embedder never
asked for the fold, and it pays for it on queries the noise class never touched. The
`retrieve_queries` seam already carries separate dense and lexical text, so the fold can stay on
the lexical side alone.

This module transfers CASE ONLY. The processed characters are never replaced by the raw ones, so
apostrophe unification, transliteration, typo repair, and glossary expansion all survive intact --
a restored token differs from the processed token in capitalization and nothing else.
"""

import difflib
import re

from llb.rag.vector_store.lexical import _TOKEN_RE


def apply_case_pattern(source: str, target: str) -> str:
    """Re-case `target` the way `source` is cased, leaving `target`'s characters alone.

    Three patterns cover real queries: an all-lowercase source leaves the target untouched, an
    all-uppercase source (`NP`, an acronym the normalize step folds to `np`) uppercases it, and a
    leading capital (a sentence start or a proper noun) capitalizes it. A mixed-case source with
    the same length as the target is copied character by character; any other mixture falls back
    to the leading-capital rule, which is the only part of the pattern that carries meaning.

    A target that still carries case of its own is returned untouched. This only ever RESTORES
    case the fold removed -- every step that rewrites a token emits lowercase (a transliteration,
    a corpus-vocabulary correction, a glossary form), so cased target text came from somewhere the
    raw query cannot speak for, such as a model rewrite.
    """
    if not source or not target or source == source.casefold():
        return target
    if target != target.casefold():
        return target
    if source.isupper():
        return target.upper()
    if source[1:] == source[1:].casefold():
        return target[:1].upper() + target[1:]
    if len(source) == len(target):
        return "".join(char.upper() if src.isupper() else char for src, char in zip(source, target))
    return target[:1].upper() + target[1:]


def _tokens(text: str) -> list[re.Match[str]]:
    return list(_TOKEN_RE.finditer(text))


def _aligned_pairs(raw_tokens: list[str], processed_tokens: list[str]) -> dict[int, int]:
    """Map processed-token index -> raw-token index, over a casefolded token-sequence diff.

    Equal blocks pair off directly. A same-length `replace` block also pairs off, which is what
    carries case across a substitution the query-prep steps made (`Kyiv` -> `київ`, a corrected
    typo, a repaired homoglyph). Insertions -- the glossary's appended surface forms, a rewrite's
    new words -- have no raw counterpart and keep the case the step produced.
    """
    matcher = difflib.SequenceMatcher(
        a=[token.casefold() for token in raw_tokens],
        b=[token.casefold() for token in processed_tokens],
        autojunk=False,
    )
    pairs: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for offset in range(j2 - j1):
                pairs[j1 + offset] = i1 + offset
    return pairs


def restore_query_case(raw: str, processed: str) -> str:
    """Return `processed` with each aligned token re-cased the way `raw` typed it.

    Alignment is a token-sequence diff, so the transfer degrades gracefully: a step that rewrote,
    dropped, or appended tokens simply leaves those tokens with the case it produced. When nothing
    aligns to a differently cased token the processed text is returned unchanged.
    """
    raw_matches = _tokens(raw)
    processed_matches = _tokens(processed)
    if not raw_matches or not processed_matches:
        return processed
    raw_tokens = [match.group(0) for match in raw_matches]
    processed_tokens = [match.group(0) for match in processed_matches]
    pairs = _aligned_pairs(raw_tokens, processed_tokens)
    if not pairs:
        return processed
    out: list[str] = []
    cursor = 0
    for index, match in enumerate(processed_matches):
        source_index = pairs.get(index)
        if source_index is None:
            continue
        recased = apply_case_pattern(raw_tokens[source_index], match.group(0))
        if recased == match.group(0):
            continue
        out.append(processed[cursor : match.start()])
        out.append(recased)
        cursor = match.end()
    if not out:
        return processed
    out.append(processed[cursor:])
    return "".join(out)
