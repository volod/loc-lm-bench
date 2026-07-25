"""Carry a gold set's span offsets onto a repeat-stripped corpus.

A `drop`/`anchor` rewrite moves characters, so a gold span's offsets no longer point at its words.
`remap_goldset` rewrites each item's spans onto the stripped document: a span inside a removed copy
follows onto the byte-identical survivor, a span that straddles a removed block boundary is either
dropped (default) or split and re-anchored on both sides (`recover_straddle`), and every kept piece
is verified against the stripped text so an off-by-one remap is refused rather than scored.
"""

from pathlib import Path
from typing import Any

from typing_extensions import TypedDict

from llb.goldset.schema import GoldItem, SourceSpan, load_goldset
from llb.prep.pdf.repeats import StrippedDoc, remap_span, remap_span_split, span_rehomed


class GoldsetRemap(TypedDict):
    """How a gold set survived the rewrite."""

    items: int
    remapped: int
    dropped: list[str]  # ids whose spans straddle a rewrite and can no longer be anchored
    rehomed: list[str]  # remapped ids whose evidence sat on a dropped copy (now on the survivor)


def remap_goldset(
    goldset: Path | str | None,
    goldset_out: Path | str | None,
    stripped: dict[str, StrippedDoc],
    recover_straddle: bool = False,
) -> GoldsetRemap | None:
    """Rewrite each item's span offsets onto the stripped corpus; drop what cannot be anchored.

    With `recover_straddle`, a span crossing a removed block boundary is split at the boundary and
    kept as several spans instead of dropping the item (`remap_span_split`).
    """
    if goldset is None:
        return None
    items = load_goldset(goldset)
    kept: list[GoldItem] = []
    dropped: list[str] = []
    rehomed: list[str] = []
    for item in items:
        remapped = [
            _remap_gold_span(span, stripped.get(span.doc_id), recover_straddle)
            for span in item.source_spans
        ]
        if any(spans is None for spans, _ in remapped):
            dropped.append(item.id)
            continue
        moved = [piece for spans, _ in remapped for piece in (spans or [])]
        if any(was_rehomed for _, was_rehomed in remapped):
            rehomed.append(item.id)
        kept.append(item.model_copy(update={"source_spans": moved}))
    if goldset_out is not None:
        out = Path(goldset_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(item.model_dump_json() + "\n" for item in kept), encoding="utf-8")
    return {"items": len(items), "remapped": len(kept), "dropped": dropped, "rehomed": rehomed}


def _remap_gold_span(
    span: SourceSpan, stripped: StrippedDoc | None, recover_straddle: bool
) -> tuple[list[SourceSpan] | None, bool]:
    """The span's image(s) in the stripped document (and whether it was re-homed onto a survivor).

    Returns a list of one span normally, several when a straddling span is split and recovered, or
    None when it is unanchorable. Every piece is verified against the stripped text itself, so a
    remap that is off by one character reads as unanchorable rather than scoring the wrong words.
    """
    if stripped is None:
        return None, False
    if not stripped["edits"]:
        return [span], False
    rehomed = span_rehomed(stripped["edits"], span.char_start, span.char_end)
    if recover_straddle:
        images = remap_span_split(stripped["edits"], span.char_start, span.char_end)
    else:
        single = remap_span(stripped["edits"], span.char_start, span.char_end)
        images = [single] if single is not None else None
    if images is None:
        return None, False
    pieces = [span.model_copy(update=_piece(stripped["text"], lo, hi)) for lo, hi in images]
    # the pieces are the original span cut at edit boundaries, so in order they reconstruct it
    # exactly; a mismatch means the remap moved a character and the item is refused.
    if "".join(piece.text for piece in pieces) != span.text:
        return None, False
    return pieces, rehomed


def _piece(stripped_text: str, lo: int, hi: int) -> dict[str, Any]:
    """The offset/text update for one remapped sub-span, read out of the stripped document."""
    return {"char_start": lo, "char_end": hi, "text": stripped_text[lo:hi]}
