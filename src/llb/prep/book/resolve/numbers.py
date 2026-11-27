"""The text a number parser may read from a resolved block.

The adopted text drops an unresolved span, so the characters on either side of it join:
`Figure 12-3` with its `1` unresolved adopts as `Figure 2-3`, a number the page never printed.
A parser that reads printed numbers -- a folio, a section number, a caption's figure number, a
contents entry -- reads this text instead. Every run of letters, digits, and number punctuation
that touches an unresolved span becomes one placeholder, so no part of that number survives to
be read as another.
"""

import re
from typing import Final

from llb.core.contracts.data_prep.book_resolve import ResolvedBlock

# Not a digit, a letter, a space, or number punctuation, so no number pattern can absorb it.
NUMBER_PLACEHOLDER: Final = "?"
_SENTINEL: Final = "\x00"
_TOUCHING: Final = re.compile(r"[0-9A-Za-z.,\-]*\x00[0-9A-Za-z.,\-\x00]*")


def number_text(resolved: ResolvedBlock) -> str:
    """The adopted text with each number an unresolved span touches replaced by a placeholder."""
    parts: list[str] = []
    for span in resolved.spans:
        if span.provenance == "unresolved":
            parts.append(_SENTINEL)
        elif span.adopted:
            parts.append(span.text)
    joined = "".join(parts)
    if _SENTINEL not in joined:
        return joined
    return _TOUCHING.sub(_masked, joined)


def _masked(match: re.Match[str]) -> str:
    """One placeholder for the number, keeping the sentence punctuation that followed it."""
    run = match.group()
    return NUMBER_PLACEHOLDER + run[len(run.rstrip(".,")) :]
