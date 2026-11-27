"""Replace citations in one note with links, and leave the rest as text.

A match inside a fence, a display formula, an existing link, or an anchor line
is not a citation. Each number is resolved on its own, so ``Figs. 2-2 and 9-9``
links the figure the inventory names and leaves the other as text.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from llb.prep.book.render.catalog import Catalog, Target, canonical
from llb.prep.book.render.constants import CONTENT_DIRNAME
from llb.prep.book.render.grammar import ReferenceGrammar, page_token
from llb.prep.book.structure.text import norm_printed

_ANCHOR = re.compile(r"^\^([A-Za-z][A-Za-z0-9-]*)\s*$")
_INLINE = re.compile(
    r"`[^`\n]+`|(?<!!)\[(?:\\.|[^\]])*\]\([^)\s]+\)|!\[[^\]]*\]\([^)\s]+\)|!?\[\[[^\]]+\]\]"
)
_PAGE_TOKEN = page_token()


Resolve = Callable[[str], tuple[str, str, str, Target | None]]


@dataclass(frozen=True, slots=True)
class Event:
    """One cited number and what happened to it."""

    kind: str
    outcome: str
    key: str


@dataclass(frozen=True, slots=True)
class NumberHit:
    """One cited number. Offsets are relative to the slice passed to ``_rewrite``."""

    start: int
    end: int
    event: Event
    target: Target | None


@dataclass(frozen=True, slots=True)
class Cited:
    """One cited number and its span in the text that was scanned."""

    kind: str
    outcome: str
    key: str
    start: int
    end: int
    target: Target | None


@dataclass(frozen=True, slots=True)
class Replacement:
    """One span of the original note replaced by linked text."""

    start: int
    end: int
    text: str
    events: tuple[Event, ...]
    cited: tuple[Cited, ...] = ()


def link_text(
    text: str,
    *,
    here: str,
    catalog: Catalog,
    grammar: ReferenceGrammar,
    index: bool,
) -> tuple[str, tuple[Event, ...]]:
    """The note with unique citations linked, plus one event per cited number."""
    chosen = _chosen(text, here=here, catalog=catalog, grammar=grammar, index=index)
    events = tuple(
        event for rep in sorted(chosen, key=lambda item: item.start) for event in rep.events
    )
    return _apply(text, chosen), events


def citation_hits(
    text: str,
    *,
    here: str,
    catalog: Catalog,
    grammar: ReferenceGrammar,
    index: bool,
) -> tuple[Cited, ...]:
    """Every cited number the vault linker would see, with its span in ``text``."""
    chosen = _chosen(text, here=here, catalog=catalog, grammar=grammar, index=index)
    return tuple(
        cited for rep in sorted(chosen, key=lambda item: item.start) for cited in rep.cited
    )


def place_cited(hits: tuple[NumberHit, ...], base: int) -> tuple[Cited, ...]:
    """Move rewrite-relative hits onto the document that contained the slice."""
    return tuple(
        Cited(
            hit.event.kind,
            hit.event.outcome,
            hit.event.key,
            base + hit.start,
            base + hit.end,
            hit.target,
        )
        for hit in hits
    )


def _chosen(
    text: str,
    *,
    here: str,
    catalog: Catalog,
    grammar: ReferenceGrammar,
    index: bool,
) -> list[Replacement]:
    blocked = protected_spans(text)
    found = _kind_reps(text, blocked, here, catalog, grammar)
    found.extend(_page_reps(text, blocked, here, catalog, grammar))
    if index:
        from llb.prep.book.render.index_links import index_replacements

        found.extend(index_replacements(text, blocked, here, catalog))
    return _select(found)


def protected_spans(text: str) -> list[tuple[int, int]]:
    """Regions that already have a link, or are not prose."""
    spans: list[tuple[int, int]] = []
    fenced = False
    math = False
    offset = 0
    for line in text.splitlines(keepends=True):
        fenced, math = _line(line, offset, fenced, math, spans)
        offset += len(line)
    return spans


def href_for(target: Target, here: str) -> str:
    """A relative link to one block anchor. Root notes address ``content/``."""
    fragment = f"#^{target.anchor}"
    if here == "":
        return f"{CONTENT_DIRNAME}/{target.stem}.md{fragment}"
    return f"{target.stem}.md{fragment}"


def _line(
    line: str,
    offset: int,
    fenced: bool,
    math: bool,
    spans: list[tuple[int, int]],
) -> tuple[bool, bool]:
    stripped = line.strip()
    if stripped.startswith("```"):
        spans.append((offset, offset + len(line)))
        return not fenced, math
    if fenced or _ANCHOR.match(stripped):
        spans.append((offset, offset + len(line)))
        return fenced, math
    if stripped == "$$":
        spans.append((offset, offset + len(line)))
        return fenced, not math
    if math:
        spans.append((offset, offset + len(line)))
        return fenced, math
    for match in _INLINE.finditer(line):
        spans.append((offset + match.start(), offset + match.end()))
    return fenced, math


def _kind_reps(
    text: str,
    blocked: list[tuple[int, int]],
    here: str,
    catalog: Catalog,
    grammar: ReferenceGrammar,
) -> list[Replacement]:
    found: list[Replacement] = []
    for kind in grammar.kinds:
        for match in kind.phrase.finditer(text):
            if _blocked(match.start(), match.end(), blocked):
                continue
            resolve = _item_resolver(kind.name, here, catalog)
            body, hits = _rewrite_after(match.group(0), kind.cues, kind.number, resolve, kind.name)
            if hits:
                found.append(_replacement(match.start(), match.end(), body, hits))
    return found


def _page_reps(
    text: str,
    blocked: list[tuple[int, int]],
    here: str,
    catalog: Catalog,
    grammar: ReferenceGrammar,
) -> list[Replacement]:
    found: list[Replacement] = []
    resolve = _page_resolver(here, catalog, "page")
    for match in grammar.page_phrase.finditer(text):
        if _blocked(match.start(), match.end(), blocked):
            continue
        body, hits = _rewrite_after(match.group(0), grammar.page_cues, _PAGE_TOKEN, resolve, "page")
        if hits:
            found.append(_replacement(match.start(), match.end(), body, hits))
    return found


def _replacement(start: int, end: int, body: str, hits: tuple[NumberHit, ...]) -> Replacement:
    return Replacement(
        start,
        end,
        body,
        tuple(hit.event for hit in hits),
        place_cited(hits, start),
    )


def _rewrite_after(
    body: str,
    cues: tuple[str, ...],
    number: re.Pattern[str],
    resolve: Resolve,
    kind: str,
) -> tuple[str, tuple[NumberHit, ...]]:
    """Link numbers after the cue, so a roman letter inside the cue is not a number."""
    cut = _cue_end(body, cues)
    rewritten, hits = _rewrite(body[cut:], number, resolve, kind)
    shifted = tuple(
        NumberHit(cut + hit.start, cut + hit.end, hit.event, hit.target) for hit in hits
    )
    return body[:cut] + rewritten, shifted


def _cue_end(body: str, cues: tuple[str, ...]) -> int:
    lowered = body.lower()
    for cue in sorted(cues, key=len, reverse=True):
        if lowered.startswith(cue.lower()):
            return len(cue)
    return 0


def _rewrite(
    body: str, number: re.Pattern[str], resolve: Resolve, kind: str
) -> tuple[str, tuple[NumberHit, ...]]:
    pieces: list[str] = []
    hits: list[NumberHit] = []
    cursor = 0
    for match in number.finditer(body):
        pieces.append(body[cursor : match.start(1)])
        token = match.group(1)
        outcome, href, key, target = resolve(token)
        hits.append(NumberHit(match.start(1), match.end(1), Event(kind, outcome, key), target))
        pieces.append(f"[{token}]({href})" if href else token)
        cursor = match.end(1)
    pieces.append(body[cursor:])
    return "".join(pieces), tuple(hits)


def _item_resolver(kind: str, here: str, catalog: Catalog) -> Resolve:
    def resolve(token: str) -> tuple[str, str, str, Target | None]:
        key = canonical(token)
        ident = f"{kind} {key}"
        if (kind, key) in catalog.ambiguous:
            return "ambiguous", "", ident, None
        target = catalog.unique.get((kind, key))
        if target is None:
            return "unresolved", "", ident, None
        return "resolved", href_for(target, here), ident, target

    return resolve


def _page_resolver(here: str, catalog: Catalog, kind: str) -> Resolve:
    def resolve(token: str) -> tuple[str, str, str, Target | None]:
        key = norm_printed(token)
        ident = f"{kind} {key}"
        if key in catalog.page_ambiguous:
            return "ambiguous", "", ident, None
        if key in catalog.gaps:
            return "gap", "", ident, None
        target = catalog.pages.get(key)
        if target is None:
            return "unresolved", "", ident, None
        return "resolved", href_for(target, here), ident, target

    return resolve


def _select(found: list[Replacement]) -> list[Replacement]:
    chosen: list[Replacement] = []
    ordered = sorted(found, key=lambda item: (item.start, -(item.end - item.start)))
    for item in ordered:
        if any(item.start < kept.end and item.end > kept.start for kept in chosen):
            continue
        chosen.append(item)
    return chosen


def _apply(text: str, reps: list[Replacement]) -> str:
    for rep in sorted(reps, key=lambda item: item.start, reverse=True):
        text = text[: rep.start] + rep.text + text[rep.end :]
    return text


def _blocked(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < stop and end > begin for begin, stop in spans)
