"""Stage 2 -- extract entities, aliases/coreference, events, claims, and SRO facts.

The extractor is a pluggable seam (`ExtractionAdapter`). The DEFAULT is LLM-only via the
injectable `complete` (local endpoint by default). A Python-native NER/coreference adapter
(Stanza or spaCy `uk_core_news`) is an opt-in plug-in kept OUT of the base deps: implement the
`ExtractionAdapter` protocol and pass it to the pipeline.

Every extracted artifact must quote EXACT evidence; each quote is grounded back to offsets via
`ground_quote`, and anything ungrounded is dropped, so the extraction links to exact evidence
(ontology-assisted drafting acceptance). Aliases collected per entity are the lightweight coreference signal.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol

from llb.prep.frontier import parse_json_block
from llb.prep.frontier_telemetry import DraftBudgetExceeded, LLMComplete
from llb.prep.ontology.constants import (
    EXTRACT_CHUNK_OVERLAP,
    EXTRACT_CONCURRENCY,
    EXTRACT_MAX_CHARS,
    EXTRACT_PARSE_RETRIES,
)
from llb.prep.ontology.entity_types import entity_types_prompt_block, normalize_entity_type
from llb.prep.ontology.grounding import ground_quote
from llb.prep.ontology.journal import ExtractionJournal
from llb.prep.ontology.models import Claim, DocExtraction, DocRecord, Entity, Event, SROFact
from llb.prompts import render_text

_LOG = logging.getLogger(__name__)
_WINDOW_LOG_INTERVAL = 10


class ExtractionAdapter(Protocol):
    """A document -> DocExtraction extractor. Inject any implementation (LLM, spaCy, Stanza)."""

    def extract(self, doc: DocRecord) -> DocExtraction: ...


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


@dataclass
class LLMExtractionAdapter:
    """Default extractor via the injectable `complete`. A document longer than `max_chars` is
    CHUNKED into overlapping windows (verified-data hardening) -- one extraction call per window, merged -- instead of
    one truncated call, so a long doc's later content is no longer dropped. Offsets stay exact:
    grounding always runs against the FULL original text.

    An optional `journal` makes the stage resumable: a completed window is recorded and, on a later
    run over the same bundle, reused instead of re-calling the model. Window identity is the
    deterministic `split_document` index, so a journaled window is valid as long as the extraction
    settings are unchanged (the pipeline pins them in the journal meta)."""

    complete: LLMComplete
    max_chars: int = EXTRACT_MAX_CHARS
    chunk_overlap: int = EXTRACT_CHUNK_OVERLAP
    concurrency: int = EXTRACT_CONCURRENCY
    parse_retries: int = EXTRACT_PARSE_RETRIES
    journal: ExtractionJournal | None = None

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.parse_retries < 0:
            raise ValueError("parse_retries must be >= 0")

    def _call_window(self, doc_id: str, full_text: str, window_text: str) -> DocExtraction | None:
        attempts = self.parse_retries + 1
        prompt = extraction_prompt(doc_id, window_text)
        for attempt in range(1, attempts + 1):
            try:
                payload = parse_json_block(self.complete(prompt))
            except json.JSONDecodeError:
                _LOG.warning(
                    "[ontology] unparseable extraction for %s (attempt %d/%d)",
                    doc_id,
                    attempt,
                    attempts,
                )
                continue
            except DraftBudgetExceeded:
                raise
            except Exception as exc:  # endpoint/transport error -> retry, then skip the window
                _LOG.warning(
                    "[ontology] extraction call failed for %s (attempt %d/%d): %s",
                    doc_id,
                    attempt,
                    attempts,
                    exc,
                )
                continue
            if not isinstance(payload, dict):
                _LOG.warning(
                    "[ontology] extraction for %s is not a JSON object (attempt %d/%d)",
                    doc_id,
                    attempt,
                    attempts,
                )
                continue
            # Ground against the FULL original text so offsets stay exact for windowed calls.
            return parse_extraction(doc_id, full_text, payload)
        _LOG.warning(
            "[ontology] extraction for %s failed after %d attempts; window remains resumable",
            doc_id,
            attempts,
        )
        return None

    def _extract_window(
        self, doc_id: str, full_text: str, window_text: str, window_index: int, window_total: int
    ) -> DocExtraction:
        if self.journal is not None:
            cached = self.journal.get(doc_id, window_index)
            if cached is not None:
                return cached
        extraction = self._call_window(doc_id, full_text, window_text)
        if extraction is None:
            return DocExtraction(doc_id=doc_id)
        if self.journal is not None:
            self.journal.record(doc_id, window_index, window_total, extraction)
        return extraction

    def extract(self, doc: DocRecord) -> DocExtraction:
        if len(doc.text) <= self.max_chars:
            return self._extract_window(doc.doc_id, doc.text, doc.text, 1, 1)
        from llb.eval.map_reduce import split_document

        windows = split_document(doc.text, self.max_chars, self.chunk_overlap)
        total = len(windows)
        if self.concurrency == 1 or total <= 1:
            sequential_parts = []
            for index, window in enumerate(windows, start=1):
                _log_window_progress(doc.doc_id, index, total)
                sequential_parts.append(
                    self._extract_window(doc.doc_id, doc.text, window, index, total)
                )
            return merge_extractions(doc.doc_id, sequential_parts)

        worker_count = min(self.concurrency, total)
        _LOG.info(
            "[ontology] extracting %s with %d windows at concurrency %d",
            doc.doc_id,
            total,
            worker_count,
        )
        parallel_parts: list[DocExtraction | None] = [None] * total
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {}
            for index, window in enumerate(windows, start=1):
                _log_window_progress(doc.doc_id, index, total)
                future = executor.submit(
                    self._extract_window, doc.doc_id, doc.text, window, index, total
                )
                future_to_index[future] = index - 1
            for future in as_completed(future_to_index):
                parallel_parts[future_to_index[future]] = future.result()
        ordered_parts = [part for part in parallel_parts if part is not None]
        return merge_extractions(doc.doc_id, ordered_parts)


def extract_corpus(docs: list[DocRecord], adapter: ExtractionAdapter) -> list[DocExtraction]:
    """Run the extractor over every inventoried document."""
    extractions = []
    for index, doc in enumerate(docs, start=1):
        _LOG.info(
            "[ontology] stage 2: extracting doc %d/%d %s (%d chars)",
            index,
            len(docs),
            doc.doc_id,
            doc.n_chars,
        )
        extractions.append(adapter.extract(doc))
    n_facts = sum(len(e.facts) for e in extractions)
    n_ent = sum(len(e.entities) for e in extractions)
    _LOG.info("[ontology] stage 2: %d entities, %d facts across %d docs", n_ent, n_facts, len(docs))
    return extractions
