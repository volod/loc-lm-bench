"""Focused extraction parsing implementation."""

import logging
from typing import Any
from llb.prep.ontology.entity_types import entity_types_prompt_block, normalize_entity_type
from llb.prep.ontology.grounding import ground_quote
from llb.prep.ontology.models import Claim, DocExtraction, Entity, Event, SROFact
from llb.prompts.registry import render_text

_LOG = logging.getLogger(__name__)

_WINDOW_LOG_INTERVAL = 10


def extraction_prompt(doc_id: str, text: str) -> str:
    """Ask for entities/coreference/events/claims/SRO facts, each quoting EXACT evidence."""
    return render_text(
        "prep.ontology.extraction",
        {"doc_id": doc_id, "text": text, "entity_types": entity_types_prompt_block()},
    )


def _str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _grounded_mentions(doc_id: str, text: str, entry: dict[str, Any]) -> list[Any]:
    """The entry's mention quotes that ground verbatim in the document."""
    quotes = entry.get("mentions", []) if isinstance(entry.get("mentions"), list) else []
    spans = (ground_quote(doc_id, text, _str(quote)) for quote in quotes)
    return [span for span in spans if span is not None]


def _entity_from(doc_id: str, text: str, entry: Any) -> Entity | None:
    """One evidence-backed Entity from a raw entry, or None when unnamed/ungrounded."""
    if not isinstance(entry, dict):
        return None
    name = _str(entry.get("name"))
    if not name:
        return None
    mentions = _grounded_mentions(doc_id, text, entry)
    if not mentions:  # entity must be evidence-backed
        return None
    etype = normalize_entity_type(_str(entry.get("type")))  # enforce the closed vocabulary
    aliases = [_str(a) for a in entry.get("aliases", []) if _str(a)]
    return Entity(name=name, type=etype, aliases=aliases, mentions=mentions)


def _entities(doc_id: str, text: str, raw: Any) -> list[Entity]:
    entries = raw if isinstance(raw, list) else []
    candidates = (_entity_from(doc_id, text, entry) for entry in entries)
    return [entity for entity in candidates if entity is not None]


def _evidenced(doc_id: str, text: str, raw: Any, build: Any) -> list[Any]:
    out: list[Any] = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        span = ground_quote(doc_id, text, _str(entry.get("evidence")))
        if span is None:
            continue
        item = build(entry, span)
        if item is not None:
            out.append(item)
    return out


def _fact_entries(payload: dict[str, Any]) -> Any:
    facts = payload.get("facts")
    if facts is not None:
        return facts
    return payload.get("relations")


def _entry_text(entry: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = _str(entry.get(key))
        if value:
            return value
    return ""


def parse_extraction(doc_id: str, text: str, payload: Any) -> DocExtraction:
    """Turn a parsed extraction payload into a fully grounded DocExtraction."""
    if not isinstance(payload, dict):
        _LOG.warning("[ontology] extraction for %s is not a JSON object; empty", doc_id)
        return DocExtraction(doc_id=doc_id)

    def build_event(entry: dict[str, Any], span: Any) -> Event | None:
        desc = _str(entry.get("description"))
        return Event(description=desc, evidence=span) if desc else None

    def build_claim(entry: dict[str, Any], span: Any) -> Claim | None:
        claim = _str(entry.get("text"))
        return Claim(text=claim, evidence=span) if claim else None

    def build_fact(entry: dict[str, Any], span: Any) -> SROFact | None:
        s = _entry_text(entry, "subject", "source")
        r = _entry_text(entry, "relation", "predicate", "type")
        o = _entry_text(entry, "object", "target")
        return SROFact(subject=s, relation=r, object=o, evidence=span) if (s and r and o) else None

    return DocExtraction(
        doc_id=doc_id,
        entities=_entities(doc_id, text, payload.get("entities")),
        events=_evidenced(doc_id, text, payload.get("events"), build_event),
        claims=_evidenced(doc_id, text, payload.get("claims"), build_claim),
        facts=_evidenced(doc_id, text, _fact_entries(payload), build_fact),
    )


def _dedup(getter: Any, key: Any, parts: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[Any] = set()
    for part in parts:
        for item in getter(part):
            marker = key(item)
            if marker not in seen:
                seen.add(marker)
                out.append(item)
    return out


def _merge_entities(entities: dict[tuple[str, str], Entity], part: DocExtraction) -> None:
    for entity in part.entities:
        key = (entity.name, entity.type)
        if key not in entities:
            entities[key] = Entity(name=entity.name, type=entity.type)
        merged = entities[key]
        seen_aliases = set(merged.aliases)
        merged.aliases.extend(a for a in entity.aliases if a not in seen_aliases)
        seen_spans = {(m.char_start, m.char_end) for m in merged.mentions}
        merged.mentions.extend(
            m for m in entity.mentions if (m.char_start, m.char_end) not in seen_spans
        )


def merge_extractions(doc_id: str, parts: list[DocExtraction]) -> DocExtraction:
    """Merge per-window extractions for one document (dedup entities by (name,type), merging their
    mentions + aliases; events/claims/facts deduped by their grounded evidence span + payload)."""
    entities: dict[tuple[str, str], Entity] = {}
    for part in parts:
        _merge_entities(entities, part)

    return DocExtraction(
        doc_id=doc_id,
        entities=list(entities.values()),
        events=_dedup(lambda p: p.events, lambda e: (e.description, e.evidence.char_start), parts),
        claims=_dedup(lambda p: p.claims, lambda c: (c.text, c.evidence.char_start), parts),
        facts=_dedup(
            lambda p: p.facts,
            lambda f: (f.subject, f.relation, f.object, f.evidence.char_start),
            parts,
        ),
    )


def _log_window_progress(doc_id: str, index: int, total: int) -> None:
    if total <= 1:
        return
    if index == 1 or index == total or index % _WINDOW_LOG_INTERVAL == 0:
        _LOG.info("[ontology] extracting %s window %d/%d", doc_id, index, total)
