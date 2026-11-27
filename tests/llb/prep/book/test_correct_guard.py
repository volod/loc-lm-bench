"""The correction guard: admitted classes, the edit budget, and refusals."""

from llb.prep.book.constants import DEFAULT_STAGES
from llb.prep.book.correct.constants import (
    ADMITTED_CLASSES,
    CLASS_DIGIT,
    CLASS_EQUATION,
    CLASS_SIGN,
    EDIT_BUDGET,
    MAX_CHANGES_PER_EDIT,
    guard_identity,
)
from llb.prep.book.correct.guard import (
    REASON_BUDGET,
    REASON_NO_VISUAL,
    REASON_REWRITE,
    REASON_UNSUPPORTED,
    judge,
)
from llb.prep.book.correct.ollama import _reply_text, fill_prompt, parse_reply
from llb.prep.book.correct.proposal import BlockPrompt, Proposal
from llb.prep.book.correct.report import adoption_report
from llb.prep.book.metrics.constants import CLASS_DECIMAL, CLASS_FIVE_S, CLASS_ZERO_O
from llb.prep.book.resolve.constants import CORRECT_STAGE

_TEXT = "The value is 0.05."
_REFERENCE = "The value is 0.05."


def _proposal(
    start: int,
    end: int,
    replacement: str,
    confusion_class: str = CLASS_DIGIT,
    *,
    visual: bool = True,
    evidence: str = "crop shows the glyph",
) -> Proposal:
    return Proposal(start, end, replacement, confusion_class, evidence, visual)


def test_the_guard_classes_and_budget_stay_frozen():
    assert EDIT_BUDGET == 3
    assert MAX_CHANGES_PER_EDIT == 4
    assert ADMITTED_CLASSES == (
        "digit",
        "sign",
        "decimal",
        "unit",
        "index",
        "latin-greek",
        "bracket",
        "equation-number",
    )
    assert guard_identity().startswith("llb-book-correct 1 budget=3 changes=4 classes=digit,")
    assert CORRECT_STAGE not in DEFAULT_STAGES


def test_a_rewrite_is_refused_and_a_digit_without_the_crop_is_not_admitted():
    rewrite = _proposal(0, len(_TEXT), "A rewritten sentence.", "prose")
    judged = judge(_TEXT, [rewrite], saw_crop=True)
    assert judged.admitted == ()
    assert judged.refused[0].reason == REASON_REWRITE
    assert judged.refused[0].raw_text == _TEXT
    widened = _proposal(16, 17, "15")
    assert judge(_TEXT, [widened], saw_crop=True).refused[0].reason == REASON_REWRITE
    hidden = _proposal(13, 14, "O", visual=True)
    no_crop = judge(_TEXT, [hidden], saw_crop=False)
    assert no_crop.admitted == ()
    assert no_crop.refused[0].reason == REASON_NO_VISUAL
    ambiguous = _proposal(13, 14, "O", visual=False)
    marked = judge(_TEXT, [ambiguous], saw_crop=True)
    assert marked.admitted == ()
    assert marked.refused[0].reason == REASON_NO_VISUAL
    assert marked.refused[0].raw_text == "0"


def test_an_admitted_digit_stays_inside_the_budget_and_needs_evidence():
    admitted = judge(_TEXT, [_proposal(16, 17, "S")], saw_crop=True)
    assert len(admitted.admitted) == 1
    assert admitted.admitted[0].replacement == "S"
    assert admitted.refused == ()
    blank = _proposal(16, 17, "S", evidence="  ")
    assert judge(_TEXT, [blank], saw_crop=True).refused[0].reason == REASON_UNSUPPORTED
    sign = _proposal(0, 1, "\u2212", CLASS_SIGN)
    # "T" is not a sign. The class does not stretch to cover it.
    assert judge("T", [sign], saw_crop=True).refused[0].reason == REASON_REWRITE
    minus = _proposal(0, 1, "\u2212", CLASS_SIGN)
    assert judge("-", [minus], saw_crop=True).admitted[0].confusion_class == CLASS_SIGN
    number = _proposal(0, 5, "(3-5)", CLASS_EQUATION)
    assert judge("(3-4)", [number], saw_crop=True).admitted[0].replacement == "(3-5)"
    edits = [_proposal(index, index + 1, "9") for index in range(4)]
    budget = judge("0123", edits, saw_crop=True)
    assert len(budget.admitted) == EDIT_BUDGET
    assert [item.reason for item in budget.refused] == [REASON_BUDGET]


def test_the_adoption_report_separates_fixes_from_new_errors():
    fixed = adoption_report(_REFERENCE, "The value is O.05.", _REFERENCE)
    assert fixed.errors_fixed == 1
    assert fixed.tokens_harmed == 0
    assert fixed.classes_fixed[CLASS_ZERO_O] == 1
    assert fixed.classes_harmed[CLASS_ZERO_O] == 0
    harmed = adoption_report(_REFERENCE, _REFERENCE, "The value is 0,05.")
    assert harmed.errors_fixed == 0
    assert harmed.tokens_harmed == 1
    assert harmed.classes_harmed[CLASS_DECIMAL] == 1
    mixed = adoption_report(_REFERENCE, "The value is O.05.", "The value is 0.0S.")
    assert mixed.errors_fixed == 1
    assert mixed.tokens_harmed == 1
    assert mixed.classes_fixed[CLASS_ZERO_O] == 1
    assert mixed.classes_harmed[CLASS_FIVE_S] == 1
    assert set(mixed.classes_fixed) == set(mixed.classes_harmed)


def test_the_prompt_keeps_braces_in_the_block():
    filled = fill_prompt(BlockPrompt("p0001", "b", "set {0}", "a{b}", "", "", "", b""))
    assert '{"edits":[]}' in filled
    assert "set {0}" in filled
    assert "a{b}" in filled
    assert "{text}" not in filled


def test_an_empty_content_uses_the_thinking_field():
    text = _reply_text({"message": {"content": "", "thinking": '{"edits":[]}'}})
    assert parse_reply(text) == []
    assert _reply_text({"message": {"content": "kept", "thinking": "ignored"}}) == "kept"


def test_a_model_reply_is_edits_or_nothing():
    reply = '{"edits":[{"start":16,"end":17,"replacement":"S","class":"digit",'
    reply += '"evidence":"crop shows S","visual":true}]}'
    parsed = parse_reply(reply)
    assert parsed is not None and parsed[0].replacement == "S" and parsed[0].visual is True
    assert parse_reply("I would rewrite the sentence as follows.") is None
    assert parse_reply('{"edits":[{"start":1,"replacement":"x"}]}') is None
