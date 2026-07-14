"""Answer-side RAG quality (groundedness-citation-metrics): deterministic scorer + abstention + wiring."""

from llb.eval import common
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
# citation coverage: separates "does not cite" from "cites wrongly" (citation-coverage-metric)
# --------------------------------------------------------------------------------------------


def test_coverage_fully_cited_answer_is_one():
    answer = "Київ є столицею України [1]. Патент захищає винахід двадцять років [3]."
    report = g.citation_report(answer, CHUNKS)
    assert report["n_claims"] == 2 and report["n_covered_claims"] == 2
    assert report["citation_coverage"] == 1.0


def test_coverage_uncited_answer_is_zero():
    report = g.citation_report("Київ є столицею України. Дніпро тече через області.", CHUNKS)
    assert report["citation_coverage"] == 0.0
    assert report["n_claims"] == 2 and report["n_covered_claims"] == 0


def test_coverage_partially_cited_answer_is_between():
    answer = "Київ є столицею України [1]. Дніпро тече через області."
    report = g.citation_report(answer, CHUNKS)
    assert report["citation_coverage"] == 0.5


def test_coverage_separates_no_citation_from_wrong_citation():
    """The failure the metric exists for: both answers score 0.0 validity, but coverage differs."""
    uncited = g.citation_report("Київ є столицею України з 1991 року.", CHUNKS)
    wrongly_cited = g.citation_report("Київ є столицею України з 1991 року [2].", CHUNKS)
    assert uncited["citation_validity"] == wrongly_cited["citation_validity"] == 0.0
    assert uncited["citation_coverage"] == 0.0
    assert wrongly_cited["citation_coverage"] == 1.0


def test_coverage_is_independent_of_validity():
    hallucinated = g.citation_report("Київ є столицею України [9].", CHUNKS)
    assert hallucinated["citation_coverage"] == 1.0  # covered, even though the citation is bogus
    assert hallucinated["citation_validity"] == 0.0


def test_coverage_ignores_fragmentary_claims():
    # "Так [1]" has fewer than MIN_CLAIM_TOKENS content tokens -> not countable either way.
    report = g.citation_report("Так [1].", CHUNKS)
    assert report["n_claims"] == 0
    assert report["citation_coverage"] == 0.0


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
