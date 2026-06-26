"""Stage 2 -- extract entities, aliases/coreference, events, claims, and SRO facts.

The extractor is a pluggable seam (`ExtractionAdapter`). The DEFAULT is LLM-only via the
injectable `complete` (local endpoint by default). A Python-native NER/coreference adapter
(Stanza or spaCy `uk_core_news`) is an opt-in plug-in kept OUT of the base deps: implement the
`ExtractionAdapter` protocol and pass it to the pipeline.

Every extracted artifact must quote EXACT evidence; each quote is grounded back to offsets via
`ground_quote`, and anything ungrounded is dropped, so the extraction links to exact evidence
(M4.4 acceptance). Aliases collected per entity are the lightweight coreference signal.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from llb.prep.frontier import LLMComplete, parse_json_block
from llb.prep.ontology.constants import EXTRACT_CHUNK_OVERLAP, EXTRACT_MAX_CHARS
from llb.prep.ontology.entity_types import entity_types_prompt_block, normalize_entity_type
from llb.prep.ontology.grounding import ground_quote
from llb.prep.ontology.models import Claim, DocExtraction, DocRecord, Entity, Event, SROFact

_LOG = logging.getLogger(__name__)


class ExtractionAdapter(Protocol):
    """A document -> DocExtraction extractor. Inject any implementation (LLM, spaCy, Stanza)."""

    def extract(self, doc: DocRecord) -> DocExtraction: ...


def extraction_prompt(doc_id: str, text: str) -> str:
    """Ask for entities/coreference/events/claims/SRO facts, each quoting EXACT evidence."""
    return (
        "Ти аналітик, що будує онтологію з україномовного документа для оцінювання RAG.\n"
        "Виокреми з тексту нижче (нічого не вигадуй; усе має спиратися на текст):\n"
        "1. named entities -- name, type, aliases, mentions. Тип ОБОВ'ЯЗКОВО один із набору "
        "(якщо не підходить жоден -- став MISC): " + entity_types_prompt_block() + ";\n"
        "2. events -- короткий опис + evidence;\n"
        "3. claims -- твердження + evidence;\n"
        "4. facts -- трійки subject-relation-object + evidence.\n"
        "Кожне поле evidence та кожен елемент mentions МАЄ бути дослівною цитатою з тексту "
        "(скопіюй точно, символ у символ).\n"
        "Поверни лише JSON-об'єкт:\n"
        '{"entities": [{"name": ..., "type": ..., "aliases": [...], "mentions": [<цитата>]}],\n'
        ' "events": [{"description": ..., "evidence": <цитата>}],\n'
        ' "claims": [{"text": ..., "evidence": <цитата>}],\n'
        ' "facts": [{"subject": ..., "relation": ..., "object": ..., "evidence": <цитата>}]}\n\n'
        f"Документ [{doc_id}]:\n{text}\n"
    )


def _str(value: Any) -> str:
    return str(value).strip()


def _entities(doc_id: str, text: str, raw: Any) -> list[Entity]:
    entities: list[Entity] = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        name = _str(entry.get("name"))
        etype = normalize_entity_type(_str(entry.get("type")))  # enforce the closed vocabulary
        if not name:
            continue
        aliases = [_str(a) for a in entry.get("aliases", []) if _str(a)]
        mentions = []
        for quote in entry.get("mentions", []) if isinstance(entry.get("mentions"), list) else []:
            span = ground_quote(doc_id, text, _str(quote))
            if span is not None:
                mentions.append(span)
        if not mentions:  # entity must be evidence-backed
            continue
        entities.append(Entity(name=name, type=etype, aliases=aliases, mentions=mentions))
    return entities


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
        s, r, o = _str(entry.get("subject")), _str(entry.get("relation")), _str(entry.get("object"))
        return SROFact(subject=s, relation=r, object=o, evidence=span) if (s and r and o) else None

    return DocExtraction(
        doc_id=doc_id,
        entities=_entities(doc_id, text, payload.get("entities")),
        events=_evidenced(doc_id, text, payload.get("events"), build_event),
        claims=_evidenced(doc_id, text, payload.get("claims"), build_claim),
        facts=_evidenced(doc_id, text, payload.get("facts"), build_fact),
    )


def merge_extractions(doc_id: str, parts: list[DocExtraction]) -> DocExtraction:
    """Merge per-window extractions for one document (dedup entities by (name,type), merging their
    mentions + aliases; events/claims/facts deduped by their grounded evidence span + payload)."""
    entities: dict[tuple[str, str], Entity] = {}
    for part in parts:
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

    def _dedup(getter: Any, key: Any) -> list[Any]:
        out: list[Any] = []
        seen: set[Any] = set()
        for part in parts:
            for item in getter(part):
                marker = key(item)
                if marker not in seen:
                    seen.add(marker)
                    out.append(item)
        return out

    return DocExtraction(
        doc_id=doc_id,
        entities=list(entities.values()),
        events=_dedup(lambda p: p.events, lambda e: (e.description, e.evidence.char_start)),
        claims=_dedup(lambda p: p.claims, lambda c: (c.text, c.evidence.char_start)),
        facts=_dedup(
            lambda p: p.facts,
            lambda f: (f.subject, f.relation, f.object, f.evidence.char_start),
        ),
    )


@dataclass
class LLMExtractionAdapter:
    """Default extractor via the injectable `complete`. A document longer than `max_chars` is
    CHUNKED into overlapping windows (M5.6) -- one extraction call per window, merged -- instead of
    one truncated call, so a long doc's later content is no longer dropped. Offsets stay exact:
    grounding always runs against the FULL original text."""

    complete: LLMComplete
    max_chars: int = EXTRACT_MAX_CHARS
    chunk_overlap: int = EXTRACT_CHUNK_OVERLAP

    def _extract_window(self, doc_id: str, full_text: str, window_text: str) -> DocExtraction:
        try:
            payload = parse_json_block(self.complete(extraction_prompt(doc_id, window_text)))
        except json.JSONDecodeError:
            _LOG.warning("[ontology] unparseable extraction for %s; empty", doc_id)
            return DocExtraction(doc_id=doc_id)
        except Exception as exc:  # endpoint/transport error -> skip this window, keep going
            _LOG.warning("[ontology] extraction call failed for %s: %s", doc_id, exc)
            return DocExtraction(doc_id=doc_id)
        # ground against the FULL original text so offsets are exact even for a windowed call
        return parse_extraction(doc_id, full_text, payload)

    def extract(self, doc: DocRecord) -> DocExtraction:
        if len(doc.text) <= self.max_chars:
            return self._extract_window(doc.doc_id, doc.text, doc.text)
        from llb.eval.map_reduce import split_document

        windows = split_document(doc.text, self.max_chars, self.chunk_overlap)
        parts = [self._extract_window(doc.doc_id, doc.text, w) for w in windows]
        return merge_extractions(doc.doc_id, parts)


def extract_corpus(docs: list[DocRecord], adapter: ExtractionAdapter) -> list[DocExtraction]:
    """Run the extractor over every inventoried document."""
    extractions = [adapter.extract(doc) for doc in docs]
    n_facts = sum(len(e.facts) for e in extractions)
    n_ent = sum(len(e.entities) for e in extractions)
    _LOG.info("[ontology] stage 2: %d entities, %d facts across %d docs", n_ent, n_facts, len(docs))
    return extractions
