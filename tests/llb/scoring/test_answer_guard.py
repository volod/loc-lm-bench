"""Response-integrity guard: leaked reasoning and off-language answers
(thinking-suppression-and-answer-language-guard).

The leak fixtures are the measured shapes, not invented ones: `qwen3:30b` on this host emits its
deliberation into the answer body with the OPENING `<think>` consumed by the chat template, and on
a bounded budget it emits the deliberation alone with no terminator at all.
"""

from llb.scoring import answer_guard

UA_QUESTION = "Скільки років діють майнові авторські права після смерті автора?"

# Measured qwen3:30b body: English deliberation, a BARE closing tag, then the real answer.
LEAK_WITH_TERMINATOR = (
    "Okay, let's see. The user is asking about the duration of economic copyright rights.\n\n"
    "The context says 70 years after the author's death.\n</think>\n\n70 років"
)
# Measured qwen3:30b body on a 256-token budget: the deliberation consumed it, so no terminator
# and no answer were ever emitted.
LEAK_WITHOUT_TERMINATOR = (
    "Okay, I need to explain what copyright is. First, I need to check the context. "
    "The first point says the work is protected from creation."
)
CLEAN_UK = "Майнові права діють 70 років після смерті автора."


def test_bare_closing_tag_is_a_leak_and_prices_the_prefix():
    verdict = answer_guard.reasoning_leak(LEAK_WITH_TERMINATOR)
    assert verdict.leaked
    assert verdict.marker == "</think>"
    # Everything up to and including the terminator is the leak; the answer after it is not.
    assert verdict.leak_chars == LEAK_WITH_TERMINATOR.index("</think>") + len("</think>")
    assert verdict.leak_chars < len(LEAK_WITH_TERMINATOR)


def test_unterminated_leak_is_caught_by_its_deliberation_opener():
    verdict = answer_guard.reasoning_leak(LEAK_WITHOUT_TERMINATOR)
    assert verdict.leaked
    assert verdict.marker == "okay,"
    # No terminator means no answer boundary in the text, so the whole completion is the leak.
    assert verdict.leak_chars == len(LEAK_WITHOUT_TERMINATOR)


def test_clean_answer_is_not_flagged():
    assert answer_guard.reasoning_leak(CLEAN_UK) == answer_guard.LeakVerdict(False, "", 0)
    assert answer_guard.reasoning_leak("") == answer_guard.LeakVerdict(False, "", 0)
    assert answer_guard.reasoning_leak("   ") == answer_guard.LeakVerdict(False, "", 0)


def test_ukrainian_deliberation_frame_is_a_leak():
    """The RAG-benchmark shape: the same tag deliberates in UKRAINIAN once the prompt carries
    retrieved Ukrainian context, and never emits a terminator before the budget runs out."""
    body = (
        "Давайте проаналізуємо контекст, щоб знайти відповідь на питання про те, коли було "
        "засновано Герцогство Нормандія.\n\nУ контексті [1] згадується:"
    )
    verdict = answer_guard.reasoning_leak(body)
    assert verdict.leaked and verdict.marker == "давайте проаналізуємо"
    assert verdict.leak_chars == len(body)
    # The reading is UKRAINIAN, so the language guard correctly does NOT fire on it.
    assert not answer_guard.language_verdict(UA_QUESTION, body).mismatch


def test_a_bare_let_us_frame_is_not_a_leak():
    """ "Давайте" opens ordinary prose; only the deliberation verb after it makes it a leak."""
    assert not answer_guard.reasoning_leak(
        "Давайте я поясню: авторське право виникає з моменту створення твору."
    ).leaked


def test_opener_only_matches_the_answer_head():
    """A delivered answer that mentions the asker mid-body is discussing them, not deliberating."""
    body = "Авторське право виникає з моменту створення твору. " * 4 + "The user is asking again."
    assert not answer_guard.reasoning_leak(body).leaked


def test_third_person_prose_is_not_a_deliberation_opener():
    assert not answer_guard.reasoning_leak("Okays are not words a model writes here.").leaked
    assert not answer_guard.reasoning_leak("Користувачі мають право на захист твору.").leaked


def test_answer_language_reads_dominant_script_then_cyrillic_evidence():
    assert answer_guard.answer_language(CLEAN_UK) == answer_guard.UK
    assert answer_guard.answer_language("70 років.") == answer_guard.UK
    assert answer_guard.answer_language("Имущественные права действуют 70 лет.") == answer_guard.RU
    assert answer_guard.answer_language(LEAK_WITHOUT_TERMINATOR) == answer_guard.EN
    # Cyrillic that carries no distinguishing letter settles the script and stops there. Plenty of
    # genuine Ukrainian looks like this -- including this benchmark's own questions.
    assert answer_guard.answer_language("Права действуют 70 лет.") == answer_guard.CYRILLIC
    assert answer_guard.answer_language("Що таке авторське право?") == answer_guard.CYRILLIC
    # No letter evidence at all.
    assert answer_guard.answer_language("70") == answer_guard.UNDETERMINED


def test_latin_terms_do_not_flip_a_ukrainian_answer():
    answer = "Ліцензія Apache-2.0 дозволяє комерційне використання твору без реєстрації."
    assert answer_guard.answer_language(answer) == answer_guard.UK


def test_acronyms_are_not_language_evidence_on_either_side():
    """The measured false positive: a benchmark item whose correct answer is three Latin acronyms
    scored objective 1.0 and was the only off-language flag on both clean roster models."""
    assert answer_guard.answer_language("BPP, ZPP та RP.") == answer_guard.UNDETERMINED
    assert not answer_guard.language_verdict(UA_QUESTION, "BPP, ZPP та RP.").mismatch
    # Symmetric: a Cyrillic acronym does not make an answer more Ukrainian either.
    assert answer_guard.answer_language("Столицею США є Вашингтон.") == answer_guard.UK


def test_script_mismatch_fires_without_a_distinguishing_letter():
    """The measured failure -- a Ukrainian prompt answered in English -- is a SCRIPT disagreement,
    so it must not depend on either side carrying an "i" or a "ï"."""
    assert answer_guard.language_verdict(UA_QUESTION, LEAK_WITHOUT_TERMINATOR) == (
        answer_guard.UK,
        answer_guard.EN,
        True,
    )
    plain_ua = "Що таке авторське право?"
    assert answer_guard.language_verdict(plain_ua, LEAK_WITHOUT_TERMINATOR) == (
        answer_guard.CYRILLIC,
        answer_guard.EN,
        True,
    )


def test_language_mismatch_inside_cyrillic_needs_both_sides_decided():
    assert not answer_guard.language_verdict(UA_QUESTION, CLEAN_UK).mismatch
    assert answer_guard.language_verdict(
        UA_QUESTION, "Имущественные права действуют 70 лет."
    ).mismatch
    # Cyrillic the letters do not settle is recorded, never counted against the model.
    undecided = answer_guard.language_verdict(UA_QUESTION, "Права действуют 70 лет.")
    assert undecided.answer_language == answer_guard.CYRILLIC
    assert not undecided.mismatch
    # No letter evidence at all is likewise never a mismatch.
    assert not answer_guard.language_verdict(UA_QUESTION, "70").mismatch


def test_guard_verdicts_returns_both_readings():
    leak, language = answer_guard.guard_verdicts(UA_QUESTION, LEAK_WITH_TERMINATOR)
    assert leak.leaked
    # The reading is over the WHOLE completion: English deliberation dominates the two UA words.
    assert language.answer_language == answer_guard.EN and language.mismatch
