"""ontology-assisted gold-set drafting: per-stage units + a fake-endpoint full flow.

No server, no provider key, no GPU: every LLM call is an injected fake, so the inventory,
extraction grounding, ontology induction, coverage sampling, drafting, refinement, endpoint
adapter, and the end-to-end bundle are all exercised deterministically.
"""

import json

import pytest

from llb.backends.base import ChatResult
from llb.goldset.schema import SourceSpan, load_goldset
from llb.goldset.validate import validate_items
from llb.prep.frontier import ProvenanceLog
from llb.prep.ontology import endpoint as ep
from llb.prep.ontology.constants import PROVENANCE_KIND
from llb.prep.ontology.coverage import build_seeds, classify_difficulty, sample_seeds
from llb.prep.ontology.draft import context_window, draft_for_seed
from llb.prep.ontology.endpoint import EndpointConfig, build_complete
from llb.prep.ontology.extract import LLMExtractionAdapter, parse_extraction
from llb.prep.ontology.induce import induce_ontology
from llb.prep.ontology.inventory import (
    inventory_corpus,
    section_at,
    segment_sections,
    sha256_text,
)
from llb.prep.ontology.models import DocRecord, DraftSeed, SROFact
from llb.prep.ontology.pipeline import draft_goldset
from llb.prep.ontology.refine import is_circular, refine_drafts

DOC1 = "# Київ\n\nКиїв є столицею України. Місто розташоване на річці Дніпро.\n"
DOC2 = "# Львів\n\nЛьвів є культурним центром заходу. Місто засноване у 1256 році.\n"


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
    assert fact_seed.strata["relation"] == "столиця"
    assert fact_seed.difficulty == "hard"  # rare relation (count 1)


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


def test_is_circular_rejects_answer_in_question_or_equal():
    assert is_circular("Що таке столицею?", "столицею", "столицею") is True
    assert is_circular("столицею", "столицею", "столицею") is True
    assert is_circular("Чим є місто для держави?", "столицею", "столицею") is False


def test_refine_grounds_dedups_and_rejects_circular():
    docs = [DocRecord(doc_id="a.md", text=DOC1, sha256="x", n_chars=len(DOC1))]
    drafts = [
        {
            "doc_id": "a.md",
            "question": "Що відомо про Київ?",
            "reference_answer": "України",
            "answer_span": "України",
        },
        {
            "doc_id": "a.md",
            "question": "Що відомо про Київ?",
            "reference_answer": "України",
            "answer_span": "України",
        },  # duplicate question+span -> dropped
        {
            "doc_id": "a.md",
            "question": "Назви Дніпро.",
            "reference_answer": "Дніпро",
            "answer_span": "Дніпро",
        },  # circular (answer in question) -> dropped
        {
            "doc_id": "a.md",
            "question": "Куди тече річка?",
            "reference_answer": "Лондон",
            "answer_span": "Лондон",
        },  # ungrounded -> dropped
    ]
    items = refine_drafts(docs, drafts)
    assert len(items) == 1
    item = items[0]
    assert item.provenance == PROVENANCE_KIND and item.verified is False
    span = item.source_spans[0]
    assert DOC1[span.char_start : span.char_end] == "України"


# --- endpoint adapter ------------------------------------------------------------------------


def test_endpoint_config_validates_kind_model_and_egress():
    with pytest.raises(ValueError, match="endpoint kind"):
        EndpointConfig(kind="cloud", model="m")
    with pytest.raises(ValueError, match="model must be set"):
        EndpointConfig(kind="local", model="")
    assert EndpointConfig(kind="local", model="m").egress is False
    frontier = EndpointConfig(kind="frontier", model="gpt")
    assert frontier.egress is True and frontier.provenance()["egress"] is True


def test_build_complete_local_records_tokens_and_raises_on_error(monkeypatch):
    monkeypatch.setattr(ep, "make_client", lambda base_url, api_key="x": object())
    monkeypatch.setattr(
        ep, "chat_once", lambda *a, **k: ChatResult(text="OK", prompt_tokens=5, completion_tokens=2)
    )
    log = ProvenanceLog()
    complete = build_complete(EndpointConfig(kind="local", model="m"), log)
    assert complete("hi") == "OK"
    summary = log.summary()
    assert summary["calls"] == 1 and summary["total_prompt_tokens"] == 5

    monkeypatch.setattr(ep, "chat_once", lambda *a, **k: ChatResult(text="", error="timeout"))
    with pytest.raises(RuntimeError, match="local endpoint error"):
        build_complete(EndpointConfig(kind="local", model="m"), ProvenanceLog())("hi")


def test_native_chat_url_maps_v1_to_api_chat():
    assert ep._native_chat_url("http://localhost:11434/v1") == "http://localhost:11434/api/chat"
    assert ep._native_chat_url("http://h:8000/v1/") == "http://h:8000/api/chat"
    assert ep._native_chat_url("http://h:11434") == "http://h:11434/api/chat"


def test_think_disabled_routes_through_native_endpoint(monkeypatch):
    # think is honored only by Ollama's native /api/chat, so a think-set config must NOT use /v1
    monkeypatch.setattr(
        ep, "make_client", lambda *a, **k: pytest.fail("must not use the /v1 client when think set")
    )
    captured: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, object]:
            return {"message": {"content": "OK"}, "prompt_eval_count": 7, "eval_count": 3}

    def fake_post(url, json, timeout):  # noqa: A002 - mirror httpx.post signature
        captured["url"] = url
        captured["think"] = json["think"]
        captured["num_predict"] = json["options"]["num_predict"]
        return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    log = ProvenanceLog()
    cfg = EndpointConfig(kind="local", model="gemma4:26b", think=False, max_tokens=4096)
    assert cfg.provenance()["think"] is False
    assert build_complete(cfg, log)("hi") == "OK"
    assert captured["url"].endswith("/api/chat")
    assert captured["think"] is False and captured["num_predict"] == 4096
    assert log.summary()["total_prompt_tokens"] == 7


# --- stage 7: full flow over a fake local endpoint -------------------------------------------


def _extraction_json(prompt: str) -> str:
    if "столицею" in prompt:  # DOC1
        return json.dumps(
            {
                "entities": [
                    {"name": "Київ", "type": "LOC", "aliases": ["місто"], "mentions": ["Київ"]},
                    {"name": "Дніпро", "type": "LOC", "mentions": ["Дніпро"]},
                ],
                "claims": [{"text": "Київ є столицею", "evidence": "Київ є столицею України"}],
                "facts": [
                    {
                        "subject": "Київ",
                        "relation": "столиця",
                        "object": "України",
                        "evidence": "Київ є столицею України",
                    },
                    {
                        "subject": "Місто",
                        "relation": "розташоване",
                        "object": "Дніпро",
                        "evidence": "Місто розташоване на річці Дніпро",
                    },
                ],
            }
        )
    return json.dumps(
        {  # DOC2
            "entities": [{"name": "Львів", "type": "LOC", "mentions": ["Львів"]}],
            "events": [{"description": "заснування", "evidence": "Місто засноване у 1256 році"}],
            "facts": [
                {
                    "subject": "Львів",
                    "relation": "є",
                    "object": "культурним центром",
                    "evidence": "Львів є культурним центром заходу",
                },
                {
                    "subject": "Місто",
                    "relation": "засноване",
                    "object": "1256",
                    "evidence": "Місто засноване у 1256 році",
                },
            ],
        }
    )


def _draft_json(prompt: str) -> str:
    if "Сфокусуйся на факті:" in prompt:
        seg = prompt.split("Сфокусуйся на факті:")[1].split("\n")[0]
        subject = seg.split("|")[0].strip()
        obj = seg.rsplit("|", 1)[-1].strip().rstrip(".").strip()
        return json.dumps(
            {"question": f"Що відомо про {subject}?", "reference_answer": obj, "answer_span": obj}
        )
    if "Сфокусуйся на сутності:" in prompt:
        seg = prompt.split("Сфокусуйся на сутності:")[1].split("\n")[0]
        name = seg.split("(тип")[0].strip()
        return json.dumps(
            {"question": "Що згадано у документі?", "reference_answer": name, "answer_span": name}
        )
    return "{}"


def fake_endpoint(prompt: str) -> str:
    """One callable answering BOTH extraction and drafting prompts -- like a real local model."""
    if "будує онтологію" in prompt:
        return _extraction_json(prompt)
    if "укладач набору запитань" in prompt:
        return _draft_json(prompt)
    return "{}"


def test_full_flow_drafts_grounded_unverified_bundle(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc1.md").write_text(DOC1, encoding="utf-8")
    (corpus / "doc2.md").write_text(DOC2, encoding="utf-8")
    out = tmp_path / "bundle"

    result = draft_goldset(
        corpus,
        EndpointConfig(kind="local", model="fake"),
        complete=fake_endpoint,
        max_items=20,
        out_dir=out,
    )

    # items: unverified, ontology-drafted, grounded, split-assigned
    assert len(result.items) > 0
    assert all(it.verified is False and it.provenance == PROVENANCE_KIND for it in result.items)
    assert all(it.split in ("calibration", "tuning", "final") for it in result.items)

    # the emitted bundle self-validates against its copied corpus
    loaded = load_goldset(out / "goldset.jsonl")
    report = validate_items(loaded, out / "corpus")
    assert report["errors"] == []

    # ontology + extraction artifacts written
    ontology = json.loads((out / "ontology.json").read_text(encoding="utf-8"))
    assert ontology["entity_types"] and ontology["relation_types"]
    assert (out / "extraction.jsonl").exists()

    # provenance links endpoint / prompts / document hashes / cost
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert prov["kind"] == PROVENANCE_KIND and prov["synthetic"] is False
    assert prov["endpoint"]["kind"] == "local" and prov["endpoint"]["egress"] is False
    assert set(prov["prompts"]) == {"extraction", "draft"}
    assert {d["doc_id"] for d in prov["documents"]} == {"doc1.md", "doc2.md"}
    assert prov["stages"]["facts"] == 4 and prov["n_items"] == len(result.items)


def test_full_flow_does_not_write_when_write_false(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc1.md").write_text(DOC1, encoding="utf-8")
    out = tmp_path / "bundle"
    result = draft_goldset(
        corpus,
        EndpointConfig(kind="local", model="fake"),
        complete=fake_endpoint,
        out_dir=out,
        write=False,
    )
    assert not out.exists() and len(result.items) >= 0
