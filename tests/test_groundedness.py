"""Answer-side RAG quality (groundedness-citation-metrics): deterministic scorer + abstention + wiring."""

from llb.eval import common
from llb.eval import graph
from llb.executor.cases import ScoreOptions, score_case
from llb.goldset.schema import GoldItem
from llb.scoring import groundedness as g

# Chunks carry doc_id + char offsets like real retrieved records (score_case also scores
# retrieval hits); the groundedness scorer itself reads only `text`.
CHUNKS = [
    {
        "doc_id": "d",
        "char_start": 0,
        "char_end": 36,
        "text": "Київ є столицею України з 1991 року.",
    },
    {
        "doc_id": "d",
        "char_start": 40,
        "char_end": 78,
        "text": "Дніпро протікає через кілька областей.",
    },
    {
        "doc_id": "d",
        "char_start": 80,
        "char_end": 127,
        "text": "Патент захищає винахід протягом двадцяти років.",
    },
]


# --------------------------------------------------------------------------------------------
# citation parsing / claim splitting
# --------------------------------------------------------------------------------------------


def test_parse_and_strip_citations():
    assert g.parse_citations("факт один [1] і факт два [3].") == [1, 3]
    assert "[1]" not in g.strip_citations("факт [1]")
    assert g.parse_citations("без посилань") == []


def test_split_claims_breaks_on_sentence_boundaries():
    claims = g.split_claims("Перше твердження. Друге твердження! Третє?")
    assert claims == ["Перше твердження", "Друге твердження", "Третє"]


# --------------------------------------------------------------------------------------------
# groundedness fraction: fully-supported vs injected-unsupported (zero cross-class leakage)
# --------------------------------------------------------------------------------------------


def test_groundedness_fully_supported_is_one():
    assert g.groundedness_fraction("Київ є столицею України з 1991 року.", CHUNKS) == 1.0


def test_groundedness_injected_unsupported_is_zero():
    assert g.groundedness_fraction("Париж є столицею Бразилії назавжди.", CHUNKS) == 0.0


def test_groundedness_partial_answer_is_between():
    answer = "Київ є столицею України з 1991 року. Марс є четвертою планетою сонячної системи."
    frac = g.groundedness_fraction(answer, CHUNKS)
    assert frac == 0.5  # one of two claims supported


def test_groundedness_empty_answer_is_zero():
    assert g.groundedness_fraction("", CHUNKS) == 0.0


def test_chunk_supports_claim_threshold():
    assert g.chunk_supports_claim("Київ столиця України", CHUNKS[0]["text"]) is True
    assert g.chunk_supports_claim("Париж столиця Бразилії футбол", CHUNKS[0]["text"]) is False


# --------------------------------------------------------------------------------------------
# citation validity: valid / flagged-invalid / hallucinated
# --------------------------------------------------------------------------------------------


def test_citation_valid_when_chunk_supports_claim():
    report = g.citation_report("Київ є столицею України з 1991 року [1].", CHUNKS)
    assert report["citation_validity"] == 1.0
    assert report["hallucinated_citation_rate"] == 0.0


def test_citation_pointing_at_wrong_chunk_is_flagged_invalid():
    # cites chunk [2] (a river) for a claim it does not support -> invalid, NOT hallucinated
    report = g.citation_report("Київ є столицею України з 1991 року [2].", CHUNKS)
    assert report["citation_validity"] == 0.0
    assert report["hallucinated_citation_rate"] == 0.0
    assert report["n_valid"] == 0


def test_out_of_range_citation_is_hallucinated():
    report = g.citation_report("Київ є столицею України [9].", CHUNKS)
    assert report["hallucinated_citation_rate"] == 1.0
    assert report["n_hallucinated"] == 1


def test_mixed_citations_across_claims():
    answer = "Київ є столицею України з 1991 року [1]. Патент захищає винахід двадцять років [3]."
    report = g.citation_report(answer, CHUNKS)
    assert report["n_citations"] == 2
    assert report["citation_validity"] == 1.0


def test_no_citations_yields_zero_rates():
    report = g.citation_report("Київ є столицею України.", CHUNKS)
    assert report["n_citations"] == 0
    assert report["citation_validity"] == 0.0
    assert report["hallucinated_citation_rate"] == 0.0


# --------------------------------------------------------------------------------------------
# abstention detection (probe correctness signal)
# --------------------------------------------------------------------------------------------


def test_is_insufficient_context_detects_ua_ru_en():
    assert common.is_insufficient_context("У контексті немає такої інформації.")
    assert common.is_insufficient_context("Інформації недостатньо для відповіді.")
    assert common.is_insufficient_context("В контексте нет ответа.")
    assert common.is_insufficient_context("The context does not contain this.")


def test_is_abstention_covers_refusal_and_insufficient():
    assert common.is_abstention("Вибачте, але я не можу відповісти.")  # refusal
    assert common.is_abstention("На жаль, інформації недостатньо.")  # insufficient
    assert not common.is_abstention("Київ є столицею України з 1991 року.")  # substantive


def test_substantive_answer_is_not_insufficient():
    assert not common.is_insufficient_context("Столиця України — Київ.")


# --------------------------------------------------------------------------------------------
# cited-answer prompt wiring
# --------------------------------------------------------------------------------------------


def test_build_messages_cited_uses_citation_prompt():
    plain = graph.build_messages("q", "ctx")
    cited = graph.build_messages("q", "ctx", cited=True)
    assert plain[0]["content"] != cited[0]["content"]
    assert "[1]" in cited[0]["content"]  # the cited-answer system prompt instructs [i] citations


# --------------------------------------------------------------------------------------------
# score_case options: additive columns present only when enabled
# --------------------------------------------------------------------------------------------


def _item():
    return GoldItem(
        id="x1",
        lang="uk",
        question="Яка столиця України?",
        reference_answer="Київ",
        source_doc_id="d",
        source_spans=[{"doc_id": "d", "char_start": 0, "char_end": 4, "text": "Київ"}],
        provenance="human-authored",
        verified=True,
        split="final",
    )


def _state():
    return {
        "answer": "Київ є столицею України з 1991 року [1].",
        "status": common.OK,
        "retrieved": CHUNKS,
        "usage": {},
    }


def test_score_case_without_options_has_no_answer_side_fields():
    row = score_case(_item(), _state())
    assert "groundedness" not in row
    assert "citation_validity" not in row


def test_score_case_records_groundedness_and_citations_when_enabled():
    row = score_case(
        _item(),
        _state(),
        options=ScoreOptions(score_groundedness=True, cited_answers=True),
    )
    assert row["groundedness"] == 1.0
    assert row["citation_validity"] == 1.0
    assert row["hallucinated_citation_rate"] == 0.0
    assert row["n_citations"] == 1


def test_score_case_citation_order_follows_context_order():
    # reverse_rank flips prompt positions, so [1] now points at the LAST retrieved chunk
    state = dict(_state())
    state["answer"] = "Патент захищає винахід двадцять років [1]."
    row = score_case(
        _item(),
        state,
        options=ScoreOptions(cited_answers=True, context_order=common.ORDER_REVERSE_RANK),
    )
    assert row["citation_validity"] == 1.0  # [1] == reversed()[0] == the patent chunk
