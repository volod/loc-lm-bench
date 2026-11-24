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


Resolve = Callable[[str], tuple[str, str, str]]


@dataclass(frozen=True, slots=True)
class Event:
    """One cited number and what happened to it."""

    kind: str
    outcome: str
    key: str


@dataclass(frozen=True, slots=True)
class Replacement:
    """One span of the original note replaced by linked text."""

    start: int
    end: int
    text: str
    events: tuple[Event, ...]


def link_text(
    text: str,
    *,
    here: str,
    catalog: Catalog,
    grammar: ReferenceGrammar,
    index: bool,
) -> tuple[str, tuple[Event, ...]]:
    """The note with unique citations linked, plus one event per cited number."""
    blocked = protected_spans(text)
    found = _kind_reps(text, blocked, here, catalog, grammar)
    found.extend(_page_reps(text, blocked, here, catalog, grammar))
    if index:
        from llb.prep.book.render.index_links import index_replacements

        found.extend(index_replacements(text, blocked, here, catalog))
    chosen = _select(found)
    events = tuple(
        event for rep in sorted(chosen, key=lambda item: item.start) for event in rep.events
    )
    return _apply(text, chosen), events


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
            body, events = _rewrite_after(
                match.group(0), kind.cues, kind.number, resolve, kind.name
            )
            if events:
                found.append(Replacement(match.start(), match.end(), body, events))
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
        body, events = _rewrite_after(
            match.group(0), grammar.page_cues, _PAGE_TOKEN, resolve, "page"
        )
        if events:
            found.append(Replacement(match.start(), match.end(), body, events))
    return found


def _rewrite_after(
    body: str,
    cues: tuple[str, ...],
    number: re.Pattern[str],
    resolve: Resolve,
    kind: str,
) -> tuple[str, tuple[Event, ...]]:
    """Link numbers after the cue, so a roman letter inside the cue is not a number."""
    cut = _cue_end(body, cues)
    rewritten, events = _rewrite(body[cut:], number, resolve, kind)
    return body[:cut] + rewritten, events


def _cue_end(body: str, cues: tuple[str, ...]) -> int:
    lowered = body.lower()
    for cue in sorted(cues, key=len, reverse=True):
        if lowered.startswith(cue.lower()):
            return len(cue)
    return 0


def _rewrite(
    body: str, number: re.Pattern[str], resolve: Resolve, kind: str
) -> tuple[str, tuple[Event, ...]]:
    pieces: list[str] = []
    events: list[Event] = []
    cursor = 0
    for match in number.finditer(body):
        pieces.append(body[cursor : match.start(1)])
        token = match.group(1)
        outcome, href, key = resolve(token)
        events.append(Event(kind, outcome, key))
        pieces.append(f"[{token}]({href})" if href else token)
        cursor = match.end(1)
    pieces.append(body[cursor:])
    return "".join(pieces), tuple(events)


def _item_resolver(kind: str, here: str, catalog: Catalog) -> Resolve:
    def resolve(token: str) -> tuple[str, str, str]:
        key = canonical(token)
        ident = f"{kind} {key}"
        if (kind, key) in catalog.ambiguous:
            return "ambiguous", "", ident
        target = catalog.unique.get((kind, key))
        if target is None:
            return "unresolved", "", ident
        return "resolved", href_for(target, here), ident

    return resolve


def _page_resolver(here: str, catalog: Catalog, kind: str) -> Resolve:
    def resolve(token: str) -> tuple[str, str, str]:
        key = norm_printed(token)
        ident = f"{kind} {key}"
        if key in catalog.page_ambiguous:
            return "ambiguous", "", ident
        if key in catalog.gaps:
            return "gap", "", ident
        target = catalog.pages.get(key)
        if target is None:
            return "unresolved", "", ident
        return "resolved", href_for(target, here), ident

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
