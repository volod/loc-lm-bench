"""ontology drafting units, stages 4-6 + endpoint adapter.

Coverage sampling / seed building, drafting, refinement (grounding + circularity + dedup), and the
local/frontier endpoint adapter. No server, no provider key, no GPU: every LLM call is an injected
fake and the endpoint HTTP client is monkeypatched. The inventory/extraction/induction units live
in `test_ontology_extract.py`; the full flow lives in `test_ontology_draft.py`.
"""

import json


from llb.goldset.schema import SourceSpan
from llb.prep.ontology.coverage import build_seeds, classify_difficulty, sample_seeds
from llb.prep.ontology.draft import context_window, draft_for_seed, draft_prompt
from llb.prep.ontology.extract import parse_extraction
from llb.prep.ontology.inventory import segment_sections
from llb.prep.ontology.models import DocRecord, DraftSeed, SROFact

from tests.llb.prep.ontology._ontology_fixtures import DOC1, DOC2


# --- stage 4: coverage sampling --------------------------------------------------------------


def test_classify_difficulty_rare_long_short():
    assert classify_difficulty(10, rare=True) == "hard"
    assert classify_difficulty(500, rare=False) == "hard"
    assert classify_difficulty(10, rare=False) == "easy"
    assert classify_difficulty(120, rare=False) == "medium"


def test_sample_seeds_is_deterministic_and_capped():
    docs = [
        DocRecord(
            doc_id="a.md", text=DOC1, sha256="x", n_chars=len(DOC1), sections=segment_sections(DOC1)
        )
    ]
    extraction = parse_extraction(
        "a.md",
        DOC1,
        {
            "entities": [{"name": "Київ", "type": "LOC", "mentions": ["Київ"]}],
            "facts": [
                {
                    "subject": "Київ",
                    "relation": "столиця",
                    "object": "України",
                    "evidence": "Київ є столицею України",
                },
                {
                    "subject": "Місто",
                    "relation": "на",
                    "object": "Дніпро",
                    "evidence": "Місто розташоване на річці Дніпро",
                },
            ],
        },
    )
    seeds_a = sample_seeds(docs, [extraction], max_items=2, seed=7)
    seeds_b = sample_seeds(docs, [extraction], max_items=2, seed=7)
    assert len(seeds_a) == 2
    assert [s.fact.relation if s.fact else s.entity.type for s in seeds_a] == [
        s.fact.relation if s.fact else s.entity.type for s in seeds_b
    ]


def test_build_seeds_tags_section_and_difficulty():
    docs = [
        DocRecord(
            doc_id="a.md", text=DOC1, sha256="x", n_chars=len(DOC1), sections=segment_sections(DOC1)
        )
    ]
    extraction = parse_extraction(
        "a.md",
        DOC1,
        {
            "facts": [
                {
                    "subject": "Київ",
                    "relation": "столиця",
                    "object": "України",
                    "evidence": "Київ є столицею України",
                }
            ]
        },
    )
    seeds = build_seeds(docs, [extraction])
    fact_seed = next(s for s in seeds if s.kind == "fact")
    assert fact_seed.strata["section"] == "Київ"
    assert fact_seed.strata["doc"] == "a.md"
    assert fact_seed.strata["relation"] == "столиця"
    assert fact_seed.difficulty == "hard"  # rare relation (count 1)


def test_build_seeds_includes_claims_and_events():
    docs = [
        DocRecord(
            doc_id="a.md", text=DOC1, sha256="x", n_chars=len(DOC1), sections=segment_sections(DOC1)
        ),
        DocRecord(
            doc_id="b.md", text=DOC2, sha256="y", n_chars=len(DOC2), sections=segment_sections(DOC2)
        ),
    ]
    extraction_a = parse_extraction(
        "a.md",
        DOC1,
        {"claims": [{"text": "Київ є столицею", "evidence": "Київ є столицею України"}]},
    )
    extraction_b = parse_extraction(
        "b.md",
        DOC2,
        {
            "events": [
                {"description": "заснування міста", "evidence": "Місто засноване у 1256 році"}
            ]
        },
    )

    seeds = build_seeds(docs, [extraction_a, extraction_b])
    claim_seed = next(s for s in seeds if s.kind == "claim")
    event_seed = next(s for s in seeds if s.kind == "event")

    assert claim_seed.claim is not None and claim_seed.strata["doc"] == "a.md"
    # the claim's distinguishing coverage bucket is the claim text (not the section, which the
    # base strata already covers), so distinct claims in one section each get sampled
    assert claim_seed.strata["claim"] == "Київ є столицею"
    assert event_seed.event is not None and event_seed.strata["event"] == "заснування міста"
    assert "Сфокусуйся на твердженні:" in draft_prompt(claim_seed, DOC1)
    assert "Сфокусуйся на події:" in draft_prompt(event_seed, DOC2)


# --- stage 5: drafting -----------------------------------------------------------------------


def test_context_window_clamps_to_document():
    assert context_window("0123456789", 4, 6, radius=2) == "23456789"[:6]  # [2:8]
    assert context_window("abc", 0, 3, radius=100) == "abc"


def test_draft_for_seed_parses_and_tags_doc_id():
    seed = DraftSeed(
        doc_id="a.md",
        kind="fact",
        section_title="Київ",
        difficulty="hard",
        strata={"relation": "столиця"},
        evidence=SourceSpan(
            doc_id="a.md", char_start=8, char_end=31, text="Київ є столицею України"
        ),
        fact=SROFact(
            subject="Київ",
            relation="столиця",
            object="України",
            evidence=SourceSpan(
                doc_id="a.md", char_start=8, char_end=31, text="Київ є столицею України"
            ),
        ),
    )
    payload = json.dumps(
        {"question": "Чим є Київ?", "reference_answer": "столицею", "answer_span": "столицею"}
    )
    draft = draft_for_seed(lambda _p: payload, DOC1, seed)
    assert draft is not None and draft["doc_id"] == "a.md"
    assert draft_for_seed(lambda _p: "not json", DOC1, seed) is None


# --- stage 6: refine -------------------------------------------------------------------------


# --- endpoint adapter ------------------------------------------------------------------------
