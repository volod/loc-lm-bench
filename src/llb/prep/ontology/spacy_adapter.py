"""Opt-in Stanza / spaCy `uk_core_news` extraction adapter (M5.6).

A Python-native NER extractor implementing the `ExtractionAdapter` seam (`extract.py`), kept OUT of
the base deps: install spaCy + the `uk_core_news` pipeline yourself, then pass this adapter to
`draft_goldset(extraction_adapter=...)`. It extracts NAMED ENTITIES (spaCy's strength) with exact-
grounded mention spans, grouping repeated surfaces of the same type into one `Entity` (a lightweight
coreference signal). Events / claims / SRO facts are left to the LLM adapter -- this plug-in is the
deterministic, no-egress entity layer of the ontology pipeline.

`nlp` is injectable, so the mapping (spaCy label -> ontology type, grounding, grouping) is unit-tested
with a tiny FAKE pipeline and no spaCy install. `spacy` is imported lazily only when `nlp` is built
from a model name.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from llb.goldset.schema import SourceSpan
from llb.prep.ontology.constants import EXTRACT_MAX_CHARS
from llb.prep.ontology.grounding import ground_quote
from llb.prep.ontology.models import DocExtraction, DocRecord, Entity

_LOG = logging.getLogger(__name__)

DEFAULT_SPACY_MODEL = "uk_core_news_sm"

# spaCy `uk_core_news` entity labels -> the ontology's entity-type vocabulary (extract.py uses
# PERSON/ORG/LOC/EVENT/DATE/MISC). Unknown labels pass through unchanged.
_LABEL_MAP = {
    "PER": "PERSON",
    "PERS": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC",
    "GPE": "LOC",
    "DATE": "DATE",
    "MISC": "MISC",
}


def map_label(label: str) -> str:
    """Map a spaCy entity label onto the ontology entity-type vocabulary."""
    return _LABEL_MAP.get(label.upper(), label.upper() or "MISC")


@dataclass
class SpacyExtractionAdapter:
    """spaCy `uk_core_news` NER adapter (entities only; opt-in, lazy spaCy import)."""

    nlp: Any = None  # a spaCy Language; built lazily from `model` when None
    model: str = DEFAULT_SPACY_MODEL
    max_chars: int = EXTRACT_MAX_CHARS
    _examples: dict[str, Any] = field(default_factory=dict, repr=False)

    def _ensure_nlp(self) -> Any:
        if self.nlp is None:
            try:
                import spacy
            except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
                raise SystemExit(
                    "ERROR: the spaCy extraction adapter needs spaCy + the uk_core_news pipeline. "
                    "Run: uv pip install spacy && python -m spacy download uk_core_news_sm"
                ) from exc
            self.nlp = spacy.load(self.model)
        return self.nlp

    def _span_for(self, doc: DocRecord, ent: Any, surface: str) -> SourceSpan | None:
        """Prefer the ent's OWN char offsets (they index a prefix of the doc, so they are exact
        offsets into the full text); fall back to grounding the surface when offsets are absent."""
        start = getattr(ent, "start_char", None)
        end = getattr(ent, "end_char", None)
        if start is not None and end is not None and doc.text[start:end] == ent.text:
            return SourceSpan(doc_id=doc.doc_id, char_start=start, char_end=end, text=ent.text)
        return ground_quote(doc.doc_id, doc.text, surface)

    def extract(self, doc: DocRecord) -> DocExtraction:
        nlp = self._ensure_nlp()
        spacy_doc = nlp(doc.text[: self.max_chars])
        # group repeated surfaces of the same type into one entity (lightweight coreference).
        grouped: dict[tuple[str, str], Entity] = {}
        for ent in getattr(spacy_doc, "ents", []):
            surface = str(getattr(ent, "text", "")).strip()
            if not surface:
                continue
            etype = map_label(str(getattr(ent, "label_", "") or ""))
            span = self._span_for(doc, ent, surface)
            if span is None:  # only evidence-backed entities survive
                continue
            key = (etype, surface.casefold())
            if key not in grouped:
                grouped[key] = Entity(name=surface, type=etype)
            mentions = grouped[key].mentions
            if (span.char_start, span.char_end) not in {
                (m.char_start, m.char_end) for m in mentions
            }:
                mentions.append(span)
        entities = [e for e in grouped.values() if e.mentions]
        _LOG.info("[ontology] spaCy NER: %d entities in %s", len(entities), doc.doc_id)
        return DocExtraction(doc_id=doc.doc_id, entities=entities)
