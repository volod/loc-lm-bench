"""ontology-assisted gold-set drafting.

A multi-stage pipeline over a supplied text directory that drafts UNVERIFIED RAG gold items
linked to exact evidence. Default extraction is LLM-only via the endpoint adapter (local by
default, frontier opt-in); a Python-native NER/coreference adapter (Stanza / spaCy
`uk_core_news`) is a pluggable opt-in implementing `ExtractionAdapter`, kept out of base deps.
This is a data-preparation ontology, NOT a GraphRAG runtime (that is GraphRAG backend).
"""

from llb.prep.ontology.endpoint import (
    ENDPOINT_FRONTIER,
    ENDPOINT_LOCAL,
    EndpointConfig,
    build_complete,
)
from llb.prep.ontology.extract import ExtractionAdapter, LLMExtractionAdapter
from llb.prep.ontology.models import (
    Claim,
    DocExtraction,
    DocRecord,
    DraftSeed,
    Entity,
    Event,
    OntologyCandidate,
    OntologyType,
    Section,
    SROFact,
)
from llb.prep.ontology.pipeline.journaling import default_out_dir, load_journal_meta
from llb.prep.ontology.pipeline.run import draft_goldset
from llb.prep.ontology.pipeline.settings import PipelineResult

__all__ = [
    "ENDPOINT_FRONTIER",
    "ENDPOINT_LOCAL",
    "EndpointConfig",
    "build_complete",
    "ExtractionAdapter",
    "LLMExtractionAdapter",
    "Claim",
    "DocExtraction",
    "DocRecord",
    "DraftSeed",
    "Entity",
    "Event",
    "OntologyCandidate",
    "OntologyType",
    "Section",
    "SROFact",
    "PipelineResult",
    "default_out_dir",
    "draft_goldset",
    "load_journal_meta",
]
