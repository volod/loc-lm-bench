"""The rule every optional fold in the bundle record obeys: take it only where it pays for ITSELF.

The record has removed three kinds of repetition -- the document id repeated across every map
(`document_index.py`), the head and tail every id shares (`document_affix.py`), and the count most
documents share in a count map (`interned_counts`, below the same module). Each removes something
that carries no information, and each costs something to record: a key, a table, a default.

An UNCONDITIONAL fold would therefore make some corpus pay for a saving it cannot have -- a corpus
whose documents share no directory, a map where every document's count is different. So every fold
is measured against the form it replaces and taken only when it is actually smaller, which is what
lets this page claim a fold costs nothing where it buys nothing rather than merely little. The
corpus that does not suit a fold gets the previous form byte for byte.
"""

import json


def json_bytes(value: object) -> int:
    """What `value` occupies in the record, which is written with `ensure_ascii=False`."""
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def smaller_form[T](folded: T, flat: T) -> T:
    """`folded` when it is genuinely smaller than `flat`, and `flat` whenever it is not.

    Ties go to `flat`: a fold that saves nothing is a form to explain for no reason.
    """
    return folded if json_bytes(folded) < json_bytes(flat) else flat
