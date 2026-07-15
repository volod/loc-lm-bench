"""verified-data hardening ontology data-prep residuals: long-doc chunking, richer confidence, spaCy adapter."""

import json

from llb.prep.ontology.draft import draft_prompt
from llb.prep.ontology.extract import LLMExtractionAdapter, merge_extractions, parse_extraction
from llb.prep.ontology.induce import induce_ontology, ontology_constraints
from llb.prep.ontology.models import DocRecord, DraftSeed, OntologyCandidate, OntologyType
from llb.prep.ontology.spacy_adapter import SpacyExtractionAdapter, map_label

# --- long-doc chunking (no longer one truncated call) --------------------------------------

DOC_LONG = (
    "Перша частина згадує Київ. "
    + ("звичайний фактичний текст. " * 30)
    + "Наприкінці згадано Львів."
)


def _entities_in_window(prompt):
    ents = []
    if "Київ" in prompt:
        ents.append({"name": "Київ", "type": "LOC", "mentions": ["Київ"]})
    if "Львів" in prompt:
        ents.append({"name": "Львів", "type": "LOC", "mentions": ["Львів"]})
    return json.dumps({"entities": ents})


def test_long_doc_is_chunked_so_late_content_survives():
    adapter = LLMExtractionAdapter(complete=_entities_in_window, max_chars=120, chunk_overlap=20)
    ext = adapter.extract(DocRecord(doc_id="d", text=DOC_LONG, sha256="x", n_chars=len(DOC_LONG)))
    names = {e.name for e in ext.entities}
    assert names == {"Київ", "Львів"}  # an entity from the LAST window is captured, not truncated


def test_short_doc_stays_single_call():
    calls = []

    def complete(prompt):
        calls.append(prompt)
        return json.dumps({"entities": [{"name": "Київ", "type": "LOC", "mentions": ["Київ"]}]})

    adapter = LLMExtractionAdapter(complete=complete, max_chars=10000)
    adapter.extract(DocRecord(doc_id="d", text="Київ є столицею.", sha256="x", n_chars=16))
    assert len(calls) == 1  # below max_chars -> one extraction window


def test_merge_extractions_dedups_entities_and_facts():
    e1 = parse_extraction(
        "d",
        "Київ є столицею. Київ велике місто.",
        {"entities": [{"name": "Київ", "type": "LOC", "mentions": ["Київ є столицею"]}]},
    )
    e2 = parse_extraction(
        "d",
        "Київ є столицею. Київ велике місто.",
        {"entities": [{"name": "Київ", "type": "LOC", "mentions": ["Київ велике місто"]}]},
    )
    merged = merge_extractions("d", [e1, e2])
    assert len(merged.entities) == 1  # same (name, type) merged
    assert len(merged.entities[0].mentions) == 2  # both window mentions kept


# --- richer ontology-type confidence (document-frequency aware) -----------------------------


def test_confidence_rewards_document_spread_over_concentration():
    # doc a: 3x WORK (concentrated in one doc) + 1x LAW; doc b: 1x LAW -> LAW is in both docs.
    # (canonical vocabulary types, since extraction now normalizes to the closed set.)
    ea = parse_extraction(
        "a.md",
        "alpha beta gamma delta",
        {
            "entities": [
                {"name": "alpha", "type": "WORK", "mentions": ["alpha"]},
                {"name": "beta", "type": "WORK", "mentions": ["beta"]},
                {"name": "gamma", "type": "WORK", "mentions": ["gamma"]},
                {"name": "delta", "type": "LAW", "mentions": ["delta"]},
            ]
        },
    )
    eb = parse_extraction(
        "b.md",
        "delta epsilon",
        {"entities": [{"name": "epsilon", "type": "LAW", "mentions": ["epsilon"]}]},
    )
    ontology = induce_ontology([ea, eb])
    by_name = {t.name: t for t in ontology.entity_types}
    # LAW has a lower count (2 < 3) but appears in BOTH docs -> higher confidence
    assert by_name["LAW"].confidence > by_name["WORK"].confidence
    assert ontology.entity_types[0].name == "LAW"  # sorted by confidence


def test_ontology_constraints_lists_high_confidence_types():
    candidate = OntologyCandidate(
        entity_types=[
            OntologyType(name="PERSON", count=5, confidence=0.9, examples=[]),
            OntologyType(name="RARE", count=1, confidence=0.1, examples=[]),
        ],
        relation_types=[OntologyType(name="працює_в", count=4, confidence=0.8, examples=[])],
    )
    hint = ontology_constraints(candidate, min_confidence=0.5)
    assert "PERSON" in hint and "працює_в" in hint and "RARE" not in hint


def test_ontology_constraints_empty_when_nothing_clears_floor():
    candidate = OntologyCandidate(
        entity_types=[OntologyType(name="X", count=1, confidence=0.2, examples=[])]
    )
    assert ontology_constraints(candidate, min_confidence=0.5) == ""


def test_draft_prompt_includes_ontology_hint():
    seed = DraftSeed(
        doc_id="d",
        kind="fact",
        section_title="s",
        difficulty="medium",
        evidence={"doc_id": "d", "char_start": 0, "char_end": 1, "text": "x"},
    )
    prompt = draft_prompt(seed, "контекст", "Орієнтуйся на типи: PERSON.")
    assert "Орієнтуйся на типи: PERSON." in prompt
    # absent hint -> no extra line
    assert "Орієнтуйся" not in draft_prompt(seed, "контекст")


# --- spaCy / Stanza NER adapter (opt-in, injectable nlp) -----------------------------------


class _FakeEnt:
    def __init__(self, text, label_, start_char, end_char):
        self.text = text
        self.label_ = label_
        self.start_char = start_char
        self.end_char = end_char


class _FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class _FakeNlp:
    """A spaCy-shaped stub: yields one ent per occurrence of each surface, with real offsets."""

    def __init__(self, surfaces):  # surfaces: list of (text, label)
        self._surfaces = surfaces

    def __call__(self, text):
        ents = []
        for surface, label in self._surfaces:
            start = 0
            while (i := text.find(surface, start)) >= 0:
                ents.append(_FakeEnt(surface, label, i, i + len(surface)))
                start = i + len(surface)
        ents.sort(key=lambda e: e.start_char)
        return _FakeDoc(ents)


def test_map_label_maps_spacy_to_ontology_vocab():
    assert map_label("PER") == "PERSON"
    assert map_label("LOC") == "LOC"
    assert map_label("GPE") == "LOC"  # OntoNotes geo-political folds into LOC
    assert map_label("xyz") == "MISC"  # out-of-vocabulary collapses to MISC (closed set)


def test_spacy_adapter_extracts_grounded_entities():
    doc_text = "Олена Коваль працює у Києві. Олена Коваль -- інженерка."
    nlp = _FakeNlp([("Олена Коваль", "PER"), ("Києві", "LOC")])
    adapter = SpacyExtractionAdapter(nlp=nlp)
    ext = adapter.extract(DocRecord(doc_id="d", text=doc_text, sha256="x", n_chars=len(doc_text)))
    by_name = {e.name: e for e in ext.entities}
    assert by_name["Олена Коваль"].type == "PERSON"
    # repeated surface grouped into one entity with both grounded mentions
    assert len(by_name["Олена Коваль"].mentions) == 2
    span = by_name["Олена Коваль"].mentions[0]
    assert doc_text[span.char_start : span.char_end] == "Олена Коваль"


def test_spacy_adapter_drops_ungrounded_entity():
    # an ent whose offsets do not match the doc text (e.g. a hallucinated surface) is dropped
    bad = _FakeEnt("Марс", "LOC", 0, 4)
    nlp = type("N", (), {"__call__": lambda _self, _t: _FakeDoc([bad])})()
    adapter = SpacyExtractionAdapter(nlp=nlp)
    ext = adapter.extract(DocRecord(doc_id="d", text="Земля і Місяць.", sha256="x", n_chars=14))
    assert ext.entities == []


# --- closed entity-type vocabulary (granular + enforced) -----------------------------------


def test_normalize_entity_type_maps_into_the_closed_set():
    from llb.prep.ontology.entity_types import ENTITY_TYPES, normalize_entity_type

    # canonical types pass through
    for t in ENTITY_TYPES:
        assert normalize_entity_type(t) == t
    # granular additions are reachable via common synonyms / alternate labels
    assert normalize_entity_type("treaty") == "LAW"
    assert normalize_entity_type("work of art") == "WORK"  # space/case folded
    assert normalize_entity_type("INVENTION") == "WORK"
    assert normalize_entity_type("GPE") == "LOC"
    assert normalize_entity_type("Period") == "DURATION"
    assert normalize_entity_type("percent") == "QUANTITY"
    # out-of-vocabulary and empty collapse to MISC, so the schema stays closed
    assert normalize_entity_type("Gadget") == "MISC"
    assert normalize_entity_type("") == "MISC"


def test_extraction_enforces_closed_vocabulary():
    text = "Патент охороняє винахід."
    payload = {
        "entities": [
            {"name": "патент", "type": "patent", "mentions": ["Патент"]},  # synonym -> WORK
            {"name": "винахід", "type": "Widget", "mentions": ["винахід"]},  # unknown -> MISC
        ]
    }
    ext = parse_extraction("d", text, payload)
    by_name = {e.name: e.type for e in ext.entities}
    assert by_name["патент"] == "WORK"
    assert by_name["винахід"] == "MISC"


def test_extraction_prompt_lists_the_vocabulary():
    from llb.prep.ontology.extract import extraction_prompt

    prompt = extraction_prompt("d", "текст")
    for canonical in ("PERSON", "ORG", "LOC", "LAW", "WORK", "DURATION", "MISC"):
        assert canonical in prompt
