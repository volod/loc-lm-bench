"""ontology drafting units, stages 1-3: inventory, LLM extraction grounding, ontology induction.

No server, no provider key, no GPU: every LLM call is an injected fake. The coverage/draft/refine
and endpoint-adapter units live in `test_ontology_coverage.py`; the fake-endpoint full flow and
calibration artifacts live in `test_ontology_draft.py`.
"""

import json
import threading
import time

import pytest

from llb.prep.ontology.extract import LLMExtractionAdapter, parse_extraction
from llb.prep.ontology.induce import induce_ontology
from llb.prep.ontology.inventory import (
    inventory_corpus,
    section_at,
    segment_sections,
    sha256_text,
)
from llb.prep.ontology.models import DocRecord

from tests.llb.prep.ontology._ontology_fixtures import DOC1, DOC2


# --- stage 1: inventory ----------------------------------------------------------------------


def test_segment_sections_markdown_headings_cover_text_with_exact_offsets():
    sections = segment_sections(DOC1)
    assert [s.title for s in sections] == ["Київ"]
    sec = sections[0]
    assert DOC1[sec.char_start : sec.char_end] == DOC1  # heading -> end, offsets exact
    assert section_at(sections, DOC1.index("Дніпро")) == "Київ"


def test_segment_sections_paragraph_fallback_when_no_headings():
    text = "Перший абзац тут.\n\nДругий абзац тут."
    sections = segment_sections(text)
    assert len(sections) == 2
    assert text[sections[0].char_start : sections[0].char_end] == "Перший абзац тут."


def test_inventory_corpus_relative_ids_hash_and_empty_raises(tmp_path):
    (tmp_path / "a.md").write_text(DOC1, encoding="utf-8")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "b.txt").write_text(DOC2, encoding="utf-8")
    docs = inventory_corpus(tmp_path)
    assert [d.doc_id for d in docs] == ["a.md", "nested/b.txt"]
    assert docs[0].sha256 == sha256_text(DOC1) and docs[0].n_chars == len(DOC1)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no .* documents"):
        inventory_corpus(empty)


# --- stage 2: extraction ---------------------------------------------------------------------


def test_parse_extraction_grounds_spans_and_drops_ungrounded():
    payload = {
        "entities": [
            {"name": "Київ", "type": "LOC", "aliases": ["місто"], "mentions": ["Київ"]},
            {"name": "Привид", "type": "MISC", "mentions": ["Лондон"]},  # ungrounded -> dropped
        ],
        "facts": [
            {
                "subject": "Київ",
                "relation": "столиця",
                "object": "України",
                "evidence": "Київ є столицею України",
            },
            {"subject": "x", "relation": "y", "object": "z", "evidence": "absent"},  # dropped
        ],
        "claims": [{"text": "теза", "evidence": "столицею України"}],
    }
    extraction = parse_extraction("a.md", DOC1, payload)
    assert [e.name for e in extraction.entities] == ["Київ"]  # ungrounded entity dropped
    assert extraction.entities[0].aliases == ["місто"]
    span = extraction.entities[0].mentions[0]
    assert DOC1[span.char_start : span.char_end] == "Київ"  # offsets exact
    assert len(extraction.facts) == 1 and extraction.facts[0].object == "України"
    assert len(extraction.claims) == 1


def test_parse_extraction_accepts_relations_synonym_when_evidenced():
    payload = {
        "relations": [
            {
                "source": "Київ",
                "type": "столиця",
                "target": "України",
                "evidence": "Київ є столицею України",
            }
        ]
    }
    extraction = parse_extraction("a.md", DOC1, payload)
    assert len(extraction.facts) == 1
    assert extraction.facts[0].subject == "Київ"
    assert extraction.facts[0].relation == "столиця"
    assert extraction.facts[0].object == "України"


def test_llm_extraction_adapter_grounds_against_full_text_when_truncated():
    # truncate the call input, but evidence still grounds against the full doc
    adapter = LLMExtractionAdapter(
        complete=lambda _p: json.dumps(
            {
                "facts": [
                    {
                        "subject": "Місто",
                        "relation": "на",
                        "object": "Дніпро",
                        "evidence": "Місто розташоване на річці Дніпро",
                    }
                ]
            }
        ),
        max_chars=5,
    )
    extraction = adapter.extract(DocRecord(doc_id="a.md", text=DOC1, sha256="x", n_chars=len(DOC1)))
    assert len(extraction.facts) == 1
    span = extraction.facts[0].evidence
    assert DOC1[span.char_start : span.char_end] == "Місто розташоване на річці Дніпро"


def test_llm_extraction_adapter_swallows_endpoint_error():
    def boom(_p):
        raise RuntimeError("endpoint down")

    extraction = LLMExtractionAdapter(complete=boom).extract(
        DocRecord(doc_id="a.md", text=DOC1, sha256="x", n_chars=len(DOC1))
    )
    assert extraction.facts == [] and extraction.entities == []


def test_llm_extraction_adapter_retries_malformed_response():
    responses = iter(
        [
            "not json",
            json.dumps(
                {
                    "facts": [
                        {
                            "subject": "Київ",
                            "relation": "столиця",
                            "object": "України",
                            "evidence": "Київ є столицею України",
                        }
                    ]
                }
            ),
        ]
    )
    extraction = LLMExtractionAdapter(complete=lambda _p: next(responses)).extract(
        DocRecord(doc_id="a.md", text=DOC1, sha256="x", n_chars=len(DOC1))
    )
    assert len(extraction.facts) == 1


def test_llm_extraction_adapter_rejects_invalid_concurrency():
    with pytest.raises(ValueError, match="concurrency"):
        LLMExtractionAdapter(complete=lambda _p: "{}", concurrency=0)
    with pytest.raises(ValueError, match="parse_retries"):
        LLMExtractionAdapter(complete=lambda _p: "{}", parse_retries=-1)


def test_llm_extraction_adapter_parallel_windows_match_sequential_order():
    markers = ("Alpha", "Beta", "Gamma", "Delta")
    blocks = [
        "Alpha fact about transit.",
        "Beta fact about energy.",
        "Gamma fact about water.",
        "Delta fact about culture.",
    ]
    block_size = max(len(block) for block in blocks)
    doc_text = "".join(block.ljust(block_size) for block in blocks).rstrip()
    doc = DocRecord(doc_id="parallel.md", text=doc_text, sha256="x", n_chars=len(doc_text))

    def payload_for(prompt: str) -> str:
        facts = [
            {
                "subject": marker,
                "relation": "mentions",
                "object": "fact",
                "evidence": f"{marker} fact",
            }
            for marker in markers
            if marker in prompt
        ]
        entities = [
            {"name": marker, "type": "MISC", "mentions": [marker]}
            for marker in markers
            if marker in prompt
        ]
        return json.dumps({"entities": entities, "facts": facts})

    sequential = LLMExtractionAdapter(
        complete=payload_for, max_chars=block_size, chunk_overlap=0, concurrency=1
    ).extract(doc)

    active = 0
    max_active = 0
    lock = threading.Lock()
    delays = {"Alpha": 0.04, "Beta": 0.03, "Gamma": 0.02, "Delta": 0.01}

    def delayed_payload(prompt: str) -> str:
        nonlocal active, max_active
        marker_delay = next((delay for marker, delay in delays.items() if marker in prompt), 0.0)
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(marker_delay)
            return payload_for(prompt)
        finally:
            with lock:
                active -= 1

    parallel = LLMExtractionAdapter(
        complete=delayed_payload,
        max_chars=block_size,
        chunk_overlap=0,
        concurrency=len(markers),
    ).extract(doc)

    assert max_active > 1
    assert parallel.model_dump() == sequential.model_dump()


# --- stage 3: ontology induction -------------------------------------------------------------


def test_induce_ontology_counts_confidence_and_deterministic_order():
    e1 = parse_extraction(
        "a.md",
        DOC1,
        {
            "entities": [
                {"name": "Київ", "type": "LOC", "mentions": ["Київ"]},
                {"name": "Дніпро", "type": "LOC", "mentions": ["Дніпро"]},
            ],
            "facts": [
                {
                    "subject": "Київ",
                    "relation": "столиця",
                    "object": "України",
                    "evidence": "Київ є столицею України",
                },
            ],
        },
    )
    e2 = parse_extraction(
        "b.md",
        DOC2,
        {
            "entities": [{"name": "Львів", "type": "LOC", "mentions": ["Львів"]}],
            "facts": [
                {
                    "subject": "Львів",
                    "relation": "столиця",
                    "object": "культурним центром",
                    "evidence": "Львів є культурним центром заходу",
                },
            ],
        },
    )
    ontology = induce_ontology([e1, e2])
    loc = ontology.entity_types[0]
    assert loc.name == "LOC" and loc.count == 3 and loc.confidence == 1.0
    assert ontology.relation_types[0].name == "столиця" and ontology.relation_types[0].count == 2
