"""ontology drafting units, stages 4-6 + endpoint adapter.

Coverage sampling / seed building, drafting, refinement (grounding + circularity + dedup), and the
local/frontier endpoint adapter. No server, no provider key, no GPU: every LLM call is an injected
fake and the endpoint HTTP client is monkeypatched. The inventory/extraction/induction units live
in `test_ontology_extract.py`; the full flow lives in `test_ontology_draft.py`.
"""

import json

import pytest

from llb.backends.base import ChatResult
from llb.goldset.schema import SourceSpan
from llb.prep.frontier_telemetry import ProvenanceLog
import llb.prep.ontology.endpoint as ep
from llb.prep.ontology.constants import PROVENANCE_KIND
from llb.prep.ontology.coverage import build_seeds, classify_difficulty, sample_seeds
from llb.prep.ontology.draft import context_window, draft_for_seed, draft_prompt
from llb.prep.ontology.endpoint import build_complete
from llb.prep.ontology.endpoint_config import EndpointConfig
from llb.prep.ontology.extract import parse_extraction
from llb.prep.ontology.inventory import segment_sections
from llb.prep.ontology.language import is_ukrainian_dominant
from llb.prep.ontology.models import DocRecord, DraftSeed, SROFact
from llb.prep.ontology.refine import is_circular, refine_drafts

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


def test_is_circular_rejects_answer_in_question_or_equal():
    assert is_circular("Що таке столицею?", "столицею", "столицею") is True
    assert is_circular("столицею", "столицею", "столицею") is True
    assert is_circular("Чим є місто для держави?", "столицею", "столицею") is False


def test_ukrainian_output_gate_rejects_foreign_answer_and_allows_latin_proper_name():
    assert is_ukrainian_dominant("Організація Beta є кінцевою сутністю.") is True
    assert is_ukrainian_dominant("More than all the hawks was the brave heart.") is False
    assert (
        is_ukrainian_dominant("Це відповідь: More than all the hawks was the brave heart.") is False
    )


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


def test_refine_rejects_non_ukrainian_question_or_reference_answer():
    docs = [DocRecord(doc_id="a.md", text=DOC1, sha256="x", n_chars=len(DOC1))]
    drafts = [
        {
            "doc_id": "a.md",
            "question": "Що відомо про Київ?",
            "reference_answer": "The capital of Ukraine.",
            "answer_span": "України",
        },
        {
            "doc_id": "a.md",
            "question": "What is known about Kyiv?",
            "reference_answer": "Столицею є Київ.",
            "answer_span": "Київ",
        },
    ]

    assert refine_drafts(docs, drafts) == []


# --- endpoint adapter ------------------------------------------------------------------------


def test_endpoint_config_validates_kind_model_and_egress():
    with pytest.raises(ValueError, match="endpoint kind"):
        EndpointConfig(kind="cloud", model="m")
    with pytest.raises(ValueError, match="model must be set"):
        EndpointConfig(kind="local", model="")
    with pytest.raises(ValueError, match="local backend"):
        EndpointConfig(kind="local", model="m", backend="bad")
    with pytest.raises(ValueError, match="local backend can only"):
        EndpointConfig(
            kind="frontier", model="gpt", backend="vllm", egress_consent=True, max_calls=10
        )
    with pytest.raises(ValueError, match="explicit egress consent"):
        EndpointConfig(kind="frontier", model="gpt")
    with pytest.raises(ValueError, match="egress consent can only"):
        EndpointConfig(kind="local", model="m", egress_consent=True)
    assert EndpointConfig(kind="local", model="m").egress is False
    assert EndpointConfig(kind="local", model="m").provenance()["backend"] == "ollama"
    frontier = EndpointConfig(kind="frontier", model="gpt", egress_consent=True, max_calls=10)
    assert frontier.egress is True and frontier.provenance()["egress"] is True


def test_build_complete_local_records_tokens_and_raises_on_error(monkeypatch):
    monkeypatch.setattr(ep, "make_client", lambda base_url, api_key="x": object())
    monkeypatch.setattr(
        ep, "chat_once", lambda *a, **k: ChatResult(text="OK", prompt_tokens=5, completion_tokens=2)
    )
    log = ProvenanceLog()
    cfg = EndpointConfig(kind="local", model="m", backend="openai")
    complete = build_complete(cfg, log)
    assert complete("hi") == "OK"
    summary = log.summary()
    assert summary["calls"] == 1 and summary["total_prompt_tokens"] == 5

    monkeypatch.setattr(ep, "chat_once", lambda *a, **k: ChatResult(text="", error="timeout"))
    with pytest.raises(RuntimeError, match="local endpoint error"):
        build_complete(cfg, ProvenanceLog())("hi")


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
        captured["format"] = json["format"]
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
    assert captured["format"] == "json"
    assert log.summary()["total_prompt_tokens"] == 7


def test_vllm_think_disabled_uses_openai_extra_body(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(ep, "make_client", lambda *a, **k: sentinel)
    captured: dict[str, object] = {}

    def fake_chat_once(client, model, messages, **kwargs):
        captured["client"] = client
        captured["extra_body"] = kwargs.get("extra_body")

        class _Result:
            text = "OK"
            prompt_tokens = 11
            completion_tokens = 4
            error = None

        return _Result()

    monkeypatch.setattr(ep, "chat_once", fake_chat_once)
    cfg = EndpointConfig(
        kind="local",
        backend="vllm",
        model="hf/reasoning-model",
        base_url="http://localhost:8000/v1",
        think=False,
    )
    assert cfg.provenance()["backend"] == "vllm"
    assert build_complete(cfg, ProvenanceLog())("hi") == "OK"
    extra_body = captured["extra_body"]
    assert isinstance(extra_body, dict)
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": False}
    assert extra_body["include_reasoning"] is False
    assert extra_body["reasoning_effort"] == "none"
    assert captured["client"] is sentinel


def test_vllm_host_for_port_rewrites_default_host():
    from llb.cli.prep.draft_endpoints import _vllm_host_for_port

    assert _vllm_host_for_port("http://localhost:8000", 8010) == "http://localhost:8010"


def test_num_ctx_routes_through_native_endpoint_and_bounds_context(monkeypatch):
    # num_ctx (like think) exists only on Ollama's native /api/chat; a num_ctx-set config must
    # right-size the loaded context instead of inheriting the modelfile default (CPU offload).
    monkeypatch.setattr(
        ep,
        "make_client",
        lambda *a, **k: pytest.fail("must not use the /v1 client when num_ctx set"),
    )
    captured: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, object]:
            return {"message": {"content": "OK"}, "prompt_eval_count": 5, "eval_count": 2}

    def fake_post(url, json, timeout):  # noqa: A002 - mirror httpx.post signature
        captured["url"] = url
        captured["options"] = json["options"]
        return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    cfg = EndpointConfig(kind="local", model="qwen3.6:35b", num_ctx=16384)
    assert cfg.provenance()["num_ctx"] == 16384
    assert build_complete(cfg, ProvenanceLog())("hi") == "OK"
    assert captured["url"].endswith("/api/chat")
    options = captured["options"]
    assert isinstance(options, dict) and options["num_ctx"] == 16384


def test_default_ollama_config_uses_native_json_mode(monkeypatch):
    monkeypatch.setattr(
        ep, "make_client", lambda *a, **k: pytest.fail("Ollama must use native JSON mode")
    )
    seen: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, object]:
            return {"message": {"content": "{}"}, "prompt_eval_count": 1, "eval_count": 1}

    def fake_post(url, json, timeout):  # noqa: A002 - mirror httpx.post signature
        seen["url"] = url
        seen["payload"] = json
        return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    cfg = EndpointConfig(kind="local", model="any:model")
    assert "num_ctx" not in cfg.provenance()
    assert build_complete(cfg, ProvenanceLog())("hi") == "{}"
    assert str(seen["url"]).endswith("/api/chat")
    payload = seen["payload"]
    assert isinstance(payload, dict) and payload["format"] == "json"
