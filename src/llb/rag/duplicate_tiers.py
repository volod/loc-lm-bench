"""Duplicate-collapse tiers: which two chunk texts count as "the same passage".

Exact collapse (`llb.rag.duplicates`) merges byte-identical text only, which is loss-free but
misses the way converted PDFs actually repeat themselves: the same footer carries a page number,
the same heading picks up different whitespace, the same table header differs in one cell. Those
chunks keep their own index rows and score NEAR-ties instead of exact ties.

The tiers here are the graded answer, cheapest first, and each one is strictly coarser than the
one before:

- `exact`     -- byte-identical text. Loss-free: the survivor's text IS every copy's text.
- `normalized`-- identical after the corpus-conflict `hash` tier's normalizer (`llb.rag.lexical`
                 `tokenize`: casefold, apostrophe-variant unification, punctuation strip,
                 whitespace collapse). Merges copies that differ only in surface form.
- `masked`    -- `normalized` plus digit-run masking. Merges the page-numbered footer -- and, by
                 construction, ALSO merges two rows of a rate table that differ only in their
                 number, which is a genuine content difference. Never adopt it without measuring
                 that false-merge rate on the corpus at hand.

Only `exact` is loss-free; `normalized` and `masked` index one representative text for a group of
texts that genuinely differ, so a survivor's text is not a verbatim slice of every occurrence's
offsets. Collapse stays REVERSIBLE at every tier because a copy whose text differs from the
survivor's is stored with its own text (see `llb.rag.duplicates`).
"""

import re

from llb.rag.lexical import tokenize

TIER_NONE = "none"  # keep every copy indexed (`build-index --keep-duplicate-chunks`)
TIER_EXACT = "exact"
TIER_NORMALIZED = "normalized"
TIER_MASKED = "masked"

# Collapsing tiers in increasing coarseness; `TIER_NONE` is the absence of collapse, not a tier.
DUPLICATE_TIERS = (TIER_EXACT, TIER_NORMALIZED, TIER_MASKED)

# The tiers whose key is NOT the text itself -- the ones a residue measurement must key twice.
COARSE_TIERS = (TIER_NORMALIZED, TIER_MASKED)

# A digit run masked to one placeholder, so `Сторінка 12` and `Сторінка 137` share a key.
_DIGITS = re.compile(r"\d+")
_DIGIT_PLACEHOLDER = "#"

# Prefix on a normalized key, so a normalized key can never collide with the exact-text fallback
# a token-free chunk (pure punctuation or whitespace) falls back to.
_NORMALIZED_PREFIX = "n:"
_VERBATIM_PREFIX = "v:"


def duplicate_key(text: str, tier: str = TIER_EXACT) -> str:
    """The key `tier` groups `text` by: two chunks collapse when their keys are equal."""
    if tier == TIER_EXACT:
        return text
    if tier == TIER_NORMALIZED:
        return _normalized_key(text, mask_digits=False)
    if tier == TIER_MASKED:
        return _normalized_key(text, mask_digits=True)
    raise ValueError(f"unknown duplicate tier: {tier!r}; choose from {', '.join(DUPLICATE_TIERS)}")


def _normalized_key(text: str, mask_digits: bool) -> str:
    """The `hash` tier's normalized token stream, optionally with digit runs masked.

    A chunk with no word tokens at all (a rule line, a stray bullet) normalizes to the empty
    string, which would merge every such chunk into one. Those fall back to their verbatim text,
    so the tier never merges on the absence of content.
    """
    tokens = tokenize(text)
    if not tokens:
        return f"{_VERBATIM_PREFIX}{text}"
    stream = " ".join(tokens)
    if mask_digits:
        stream = _DIGITS.sub(_DIGIT_PLACEHOLDER, stream)
    return f"{_NORMALIZED_PREFIX}{stream}"
