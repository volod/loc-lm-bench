"""Glossary step: alias/glossary expansion that appends the corpus's spelling variants of a term.

When the query mentions a known term (or one of its surzhyk / transliterated aliases) the entry's
other surface forms are appended, so retrieval catches the variant the corpus actually uses.
Sourced from a `query_glossary.json` built from a draft bundle's dictionary candidates.
"""

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.rag.lexical import tokenize
from llb.rag.query_prep.base import STEP_GLOSSARY, QUERY_GLOSSARY_VERSION, QueryEdit
from llb.rag.query_prep.normalize import cyrillic_to_latin

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlossaryEntry:
    """A canonical term and its alias surface forms (aliases include surzhyk / transliteration)."""

    canonical: str
    aliases: tuple[str, ...] = ()

    def surface_forms(self) -> tuple[str, ...]:
        forms = [self.canonical, *self.aliases]
        seen: set[str] = set()
        unique: list[str] = []
        for form in forms:
            if form and form not in seen:
                seen.add(form)
                unique.append(form)
        return tuple(unique)


def _normalized_form(text: str) -> str:
    """Space-joined normalized token string, so alias matching respects word boundaries."""
    return " ".join(tokenize(text))


@dataclass(frozen=True)
class Glossary:
    """Alias-expansion lookup built from a draft bundle's dictionary candidates (or hand-authored).

    `expand` appends the missing surface forms of any entry whose surface form appears in the query,
    so the retriever sees every spelling the corpus might use without the raw query being lost.
    """

    entries: tuple[GlossaryEntry, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Glossary":
        entries = [
            GlossaryEntry(
                canonical=str(row["canonical"]),
                aliases=tuple(str(a) for a in row.get("aliases", []) if str(a).strip()),
            )
            for row in data.get("entries", [])
            if str(row.get("canonical", "")).strip()
        ]
        return cls(tuple(entries))

    @classmethod
    def load(cls, path: Path | str) -> "Glossary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def to_dict(self, source_bundle: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": QUERY_GLOSSARY_VERSION,
            "entries": [
                {"canonical": entry.canonical, "aliases": list(entry.aliases)}
                for entry in self.entries
            ],
        }
        if source_bundle is not None:
            payload["source_bundle"] = source_bundle
        return payload


def apply_glossary(query: str, glossary: "Glossary") -> tuple[str, list[QueryEdit]]:
    """Append the alias/canonical surface forms of every glossary entry the query triggers.

    Matching is word-boundary substring matching on the normalized token string, so a multi-word
    canonical term matches as a phrase. The raw query text is preserved; expansions are appended.
    """
    normalized = _normalized_form(query)
    if not normalized:
        return query, []
    haystack = f" {normalized} "
    present: set[str] = set(normalized.split())
    additions: list[str] = []
    edits: list[QueryEdit] = []
    for entry in glossary.entries:
        forms = [(_normalized_form(form), form) for form in entry.surface_forms()]
        if not any(norm and f" {norm} " in haystack for norm, _ in forms):
            continue
        for norm, original in forms:
            if not norm or all(token in present for token in norm.split()):
                continue
            additions.append(norm)
            present.update(norm.split())
            edits.append(
                QueryEdit(STEP_GLOSSARY, "alias", original=entry.canonical, replacement=original)
            )
            _LOG.info("[query-prep] alias expand %r += %r", entry.canonical, original)
    if not additions:
        return query, []
    return f"{query} {' '.join(additions)}", edits


def build_glossary_from_candidates(
    rows: Iterable[dict[str, Any]], *, add_transliterations: bool = True
) -> Glossary:
    """Turn `prompt_dictionary_candidates.jsonl` rows into glossary entries.

    Each candidate `term` becomes a canonical entry; its recorded `aliases` carry over, and (when
    `add_transliterations`) a romanized Latin variant of the term is added so a Latin-typed query
    still expands. Deterministic: entries are sorted by canonical term. Hand-added surzhyk /
    transliteration aliases can be appended to the emitted JSON afterwards.
    """
    entries: list[GlossaryEntry] = []
    for row in rows:
        term = str(row.get("term", "")).strip()
        if not term:
            continue
        aliases = _candidate_aliases(term, row, add_transliterations=add_transliterations)
        entries.append(GlossaryEntry(canonical=term, aliases=tuple(aliases)))
    entries.sort(key=lambda entry: entry.canonical.casefold())
    return Glossary(tuple(entries))


def _candidate_aliases(term: str, row: dict[str, Any], *, add_transliterations: bool) -> list[str]:
    """Distinct recorded aliases for a term, optionally plus its romanized Latin variant."""
    aliases: list[str] = []
    seen: set[str] = {term.casefold()}
    for alias in row.get("aliases", []) or []:
        text = str(alias).strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            aliases.append(text)
    if add_transliterations:
        romanized = cyrillic_to_latin(term)
        if romanized and romanized != term.casefold() and romanized not in seen:
            aliases.append(romanized)
    return aliases
