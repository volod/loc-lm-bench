"""answer-side-span-coverage-metric -- did the ANSWER carry each gold span's fact?

Fixtures are a two-span (two-hop) item plus single-span items, scored with the real Ukrainian
lemmatizer (a base dependency), so the morphology these assertions rest on is the morphology a run
uses.
"""

from llb.scoring import answer_spans, correctness

# One two-hop item: the patent term lives in one span, the trademark term in the other, and the
# reference answer states both. Each span is longer than the fact it contributes, exactly as a
# labeled gold span is.
QUESTION = "Скільки діє патент на винахід і скільки діє свідоцтво на торговельну марку?"
REFERENCE = "Патент на винахід діє двадцять років, а свідоцтво на торговельну марку - десять років."
SPAN_PATENT = {
    "doc_id": "d1",
    "char_start": 0,
    "char_end": 120,
    "text": (
        "Патент на винахід надає його власнику виключне право на використання винаходу "
        "і діє двадцять років від дати подання заявки."
    ),
}
SPAN_MARK = {
    "doc_id": "d2",
    "char_start": 0,
    "char_end": 90,
    "text": (
        "Свідоцтво на торговельну марку діє десять років від дати подання заявки "
        "і може бути продовжене."
    ),
}
BOTH_SPANS = [SPAN_PATENT, SPAN_MARK]


def _scores(answer: str, spans=None, reference: str = REFERENCE, question: str = QUESTION):
    return answer_spans.answer_span_scores(
        answer, BOTH_SPANS if spans is None else spans, reference, question
    )


def test_answer_carrying_both_hops_covers_every_span():
    scores = _scores(
        "Патент на винахід діє двадцять років, а свідоцтво на торговельну марку - десять років."
    )

    assert scores == {
        "answer_span_coverage": 1.0,
        "answer_all_spans": 1.0,
        "answer_spans_measured": 2,
    }


def test_one_hop_answer_covers_half_the_spans_and_fails_the_gate():
    scores = _scores("Патент на винахід діє двадцять років від дати подання заявки.")

    assert scores["answer_span_coverage"] == 0.5
    assert scores["answer_all_spans"] == 0.0


def test_vague_answer_touching_neither_fact_covers_nothing():
    scores = _scores("Строки охорони встановлює чинне законодавство України.")

    assert scores["answer_span_coverage"] == 0.0
    assert scores["answer_all_spans"] == 0.0


def test_paraphrase_in_other_grammatical_forms_still_counts_as_carried():
    # Same two facts, different cases/number/word order -- the failure token overlap alone makes.
    scores = _scores(
        "Строк дії патенту на винаходи - двадцять років; "
        "свідоцтва на торговельні марки діють десять років."
    )

    assert scores["answer_span_coverage"] == 1.0
    assert scores["answer_all_spans"] == 1.0


def test_the_objective_cannot_tell_a_both_hops_answer_from_a_one_hop_answer():
    """Why the metric exists: these two earn the SAME token F1 -- one states both facts tersely,
    the other states one of them in the reference's own words -- and the coverage pair separates
    them."""
    both = "Патент - двадцять років, свідоцтво - десять років."
    one_hop = "Патент на винахід діє двадцять років."

    assert correctness.token_f1(one_hop, REFERENCE) == correctness.token_f1(both, REFERENCE)
    assert _scores(both)["answer_all_spans"] == 1.0
    assert _scores(one_hop)["answer_all_spans"] == 0.0
    assert _scores(one_hop)["answer_span_coverage"] == 0.5


def test_naming_a_hops_subject_without_its_fact_does_not_carry_the_hop():
    """The question's own words are subtracted, so echoing them cannot carry a hop by itself."""
    scores = _scores(
        "Патент на винахід діє двадцять років, а свідоцтво на торговельну марку - невідомо."
    )

    assert scores["answer_span_coverage"] == 0.5
    assert scores["answer_all_spans"] == 0.0


def test_wrong_numeral_is_not_a_carried_fact():
    number = {
        "doc_id": "d3",
        "char_start": 0,
        "char_end": 60,
        "text": "Обліковий запис 4001 внесено до реєстру 02 числа звітного місяця.",
    }
    reference = "Обліковий запис 4001."

    assert _scores("Обліковий запис 4001.", [number], reference)["answer_all_spans"] == 1.0
    assert _scores("Обліковий запис 5002.", [number], reference)["answer_all_spans"] == 0.0


def test_span_the_reference_shares_no_content_with_is_not_judged():
    # `на` is shared, and is exactly the kind of function word that would otherwise let an
    # unrelated span be "carried" by any Ukrainian sentence at all.
    unrelated = {"doc_id": "d4", "char_start": 0, "char_end": 20, "text": "Курс валют на біржі."}

    scores = _scores("Патент діє двадцять років.", [SPAN_PATENT, unrelated])

    assert scores["answer_spans_measured"] == 1
    assert scores["answer_span_coverage"] == 1.0


def test_item_with_no_judgeable_span_reads_vacuous_and_says_so():
    scores = _scores("будь-яка відповідь", [])

    assert scores["answer_span_coverage"] == answer_spans.VACUOUS_COVERAGE
    assert scores["answer_spans_measured"] == 0


def test_single_span_coverage_agrees_with_the_contains_signal():
    """The acceptance check: where `contains` fires on a single-span item, coverage is 1.0."""
    span = [SPAN_PATENT]
    reference = "Патент на винахід діє двадцять років."
    for answer in (
        "Патент на винахід діє двадцять років.",
        "Як зазначено, патент на винахід діє двадцять років від дати подання заявки.",
    ):
        assert correctness.contains(answer, reference) == 1.0
        assert _scores(answer, span, reference)["answer_all_spans"] == 1.0


def test_a_span_keeps_its_grounded_terms_when_the_question_leaves_nothing_distinctive():
    """The fallback: a reference that restates its question verbatim is still judged."""
    span = [{"doc_id": "d", "char_start": 0, "char_end": 60, "text": SPAN_PATENT["text"]}]
    reference = "Патент на винахід."
    question = "Що таке патент на винахід?"

    assert _scores("Патент на винахід.", span, reference, question)["answer_all_spans"] == 1.0
    assert _scores("Нічого не знайдено.", span, reference, question)["answer_all_spans"] == 0.0


def test_requirements_record_which_definition_judged_each_span():
    distinctive, fallback = answer_spans.span_requirements(
        [
            SPAN_PATENT,
            {"doc_id": "d", "char_start": 0, "char_end": 30, "text": "Патент на винахід."},
        ],
        REFERENCE,
        QUESTION,
    )

    assert distinctive.distinctive and "двадцять" in distinctive.terms
    # Everything the second span grounds is either in the question or in its sibling.
    assert not fallback.distinctive


def test_lemmatizer_is_injectable_so_the_metric_needs_no_analyzer():
    scores = answer_spans.answer_span_scores(
        "alpha beta",
        [{"doc_id": "d", "char_start": 0, "char_end": 16, "text": "alpha beta gamma"}],
        "alpha beta",
        lemmatize=lambda token: token,
    )

    assert scores["answer_span_coverage"] == 1.0
