"""The per-book reference grammar render uses to find citations.

A file replaces the default entirely. The keys are the inventory and hierarchy
kinds the structure file already names, plus ``page`` and ``index``. Cues are
the words the book prints in front of a number. A comma in place of the period
(``Fig,``) is the scanned form of that cue. A number pattern is the token
itself. A dash between two tokens is a range. The trailing boundary keeps
``Chapter 22`` from swallowing ``22.1.5``, and refuses a number that runs into an
uncertainty marker.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

# A number that runs into an uncertainty marker is not the number the page printed: `Fig. 12-`
# before the marker of an unresolved `3` is not a citation of Figure 12. Both the vault marker
# and the corpus token are refused.
_MARKER_AHEAD = r"[.\-\u2013\u2014]?(?:\[!|\{\{llb-marker:)"
_BOUNDARY = rf"(?![A-Za-z0-9]|[.\-]\d|{_MARKER_AHEAD})"
_DASH = r"[\-\u2013\u2014]"
_INVENTORY = ("figure", "table", "equation")
_HIERARCHY = ("section", "chapter")
_KEYS = (*_INVENTORY, *_HIERARCHY, "page", "index")

DEFAULT_GRAMMAR = """
figure:
  cues: [figures, figs., "figs,", figure, fig., "fig,"]
  number: '[A-Za-z]?\\d+(?:[.\\-]\\d+)*'
table:
  cues: [tables, table]
  number: '[A-Za-z]?\\d+(?:[.\\-]\\d+)*'
equation:
  cues: [equations, eqs., "eqs,", equation, eq., "eq,"]
  number: '[A-Za-z]?\\d+(?:[.\\-]\\d+)*'
section:
  cues: [sections, secs., section, sec.]
  number: '\\d+(?:\\.\\d+)*'
  levels: [section, subsection]
chapter:
  cues: [chapters, chapter, chaps., chap., ch.]
  number: '\\d+|[ivxlcdm]+'
  levels: [chapter]
page:
  cues: [pages, page, pp., p.]
index:
  titles: [index]
"""


@dataclass(frozen=True, slots=True)
class KindPattern:
    """One citation kind compiled from the grammar."""

    name: str
    cues: tuple[str, ...]
    phrase: re.Pattern[str]
    number: re.Pattern[str]
    levels: tuple[str, ...]
    inventory: bool


@dataclass(frozen=True, slots=True)
class ReferenceGrammar:
    """Cues and number shapes for one book."""

    name: str
    kinds: tuple[KindPattern, ...]
    page_cues: tuple[str, ...]
    page_phrase: re.Pattern[str]
    index_titles: frozenset[str]


def load_grammar(path: Path | None) -> ReferenceGrammar:
    """The default handbook grammar, or the file an operator passed."""
    if path is None:
        return parse_grammar(DEFAULT_GRAMMAR, "default")
    if not path.is_file():
        raise ValueError(f"reference grammar not found: {path}")
    return parse_grammar(path.read_text(encoding="utf-8"), path.name)


def parse_grammar(text: str, name: str) -> ReferenceGrammar:
    """Compile one grammar document. Raises ValueError when it is not usable."""
    loaded = _mapping(yaml.safe_load(text), "reference")
    extra = sorted(set(loaded) - set(_KEYS))
    if extra:
        raise ValueError(f"unknown reference grammar keys: {', '.join(extra)}")
    kinds = tuple(_kind(key, loaded[key]) for key in (*_INVENTORY, *_HIERARCHY) if key in loaded)
    if "page" not in loaded:
        raise ValueError("reference grammar needs page cues")
    if "index" not in loaded:
        raise ValueError("reference grammar needs index titles")
    page_cues = tuple(_cues(loaded["page"]))
    return ReferenceGrammar(
        name=name,
        kinds=kinds,
        page_cues=page_cues,
        page_phrase=_page_phrase(list(page_cues)),
        index_titles=_titles(loaded["index"]),
    )


def page_token() -> re.Pattern[str]:
    """One printed-page token, arabic or roman, after its cue or inside an index tail."""
    return re.compile(r"(?i)(?<![A-Za-z])([ivxlcdm]+|\d+)(?![A-Za-z])")


def _kind(name: str, raw: object) -> KindPattern:
    mapping = _mapping(raw, name)
    cues = _string_list(mapping.get("cues"), f"{name} cues")
    number = mapping.get("number")
    if not isinstance(number, str) or not number.strip():
        raise ValueError(f"{name} number pattern is missing")
    _compile(number, name)
    inventory = name in _INVENTORY
    levels = () if inventory else tuple(_string_list(mapping.get("levels"), f"{name} levels"))
    if inventory and "levels" in mapping:
        raise ValueError(f"{name} does not take levels")
    if not inventory and not levels:
        raise ValueError(f"{name} needs levels")
    return KindPattern(name, tuple(cues), _phrase(cues, number), _number(number), levels, inventory)


def _cues(raw: object) -> list[str]:
    return _string_list(_mapping(raw, "page").get("cues"), "page cues")


def _titles(raw: object) -> frozenset[str]:
    found = _string_list(_mapping(raw, "index").get("titles"), "index titles")
    return frozenset(title.casefold() for title in found)


def _mapping(raw: object, name: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} grammar must be a mapping")
    found: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} grammar keys must be strings")
        found[key] = value
    return found


def _string_list(raw: object, label: str) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in raw):
        raise ValueError(f"{label} must be non-empty strings")
    return [item.strip() for item in raw]


def _compile(pattern: str, name: str) -> None:
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"{name} number pattern does not compile") from exc


def _phrase(cues: list[str], number: str) -> re.Pattern[str]:
    """Cue, then one or more numbers joined by a comma, ``and``, or a dash.

    The dash joins a range (``Chaps. 3-6``) without becoming part of a number
    the pattern already allows to contain hyphens (``Fig. 1-1``). A dotted
    tail (``Chapter 22.1.5``) still fails the trailing boundary, so it is not
    a citation of chapter 22.
    """
    core = rf"\(?(?:{number})\)?"
    joiner = rf"(?:\s*(?:,|and)\s*|\s*{_DASH}\s*)"
    return re.compile(
        rf"(?i)(?<![A-Za-z])(?:{_alternation(cues)})\s*{core}(?:{joiner}{core})*{_BOUNDARY}"
    )


def _number(number: str) -> re.Pattern[str]:
    # Group the caller's pattern: a top-level ``|`` would escape the cue and match a
    # bare roman letter. The letter boundaries keep the ``d`` in ``and`` from counting.
    return re.compile(rf"(?i)(?<![A-Za-z])\(?((?:{number}))\)?(?![A-Za-z])")


def _page_phrase(cues: list[str]) -> re.Pattern[str]:
    token = r"(?:[ivxlcdm]+|\d+)"
    span = rf"{token}(?:\s*{_DASH}\s*{token})?"
    return re.compile(rf"(?i)(?<![A-Za-z])(?:{_alternation(cues)})\s*{span}{_BOUNDARY}")


def _alternation(cues: list[str]) -> str:
    ordered = sorted(cues, key=len, reverse=True)
    return "|".join(re.escape(cue) for cue in ordered)
