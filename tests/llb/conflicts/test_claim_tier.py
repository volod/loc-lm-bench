"""Claim adjudication: verdict parsing, offset narrowing, and the supersession promotion.

The model never sees dates, so `superseded_by` is derived here rather than asked for. That split
is what makes the partial-supersession case work: one revision can supersede the fact it changed
and duplicate the fact it restated, because relations are assigned per claim pair.
"""

import pytest

from llb.conflicts.claim_prompt import (
    AdjudicationError,
    adjudication_prompt,
    parse_adjudication,
)
from llb.conflicts.claim_tier import adjudicate_pairs, apply_supersession, narrow_to_claim
from llb.conflicts.constants import (
    REL_COMPLEMENTARY,
    REL_CONTRADICTS,
    REL_DUPLICATE,
    REL_SUBSUMED_BY,
    REL_SUPERSEDED_BY,
)
from llb.conflicts.models import ClaimRef

CHUNK = {
    "doc_id": "regulation-2024.md",
    "chunk_id": "regulation-2024.md#heading#0002",
    "char_start": 404,
    "char_end": 636,
    "text": (
        "## Розділ 2. Строки розгляду\n\nТермін розгляду письмового звернення громадянина "
        "становить п'ятнадцять робочих днів з дня його реєстрації в установі."
    ),
}
OLD_CHUNK = {
    "doc_id": "regulation-2021.md",
    "chunk_id": "regulation-2021.md#heading#0002",
    "char_start": 383,
    "char_end": 616,
    "text": (
        "## Розділ 2. Строки розгляду\n\nТермін розгляду письмового звернення громадянина "
        "становить тридцять календарних днів з дня його реєстрації в установі."
    ),
}
NEW_GOV = {"effective_date": "2024-03-01", "version": "2.0"}
OLD_GOV = {"effective_date": "2021-01-15", "version": "1.0"}


def test_prompt_asks_for_verbatim_quotes_and_hides_provenance():
    prompt = adjudication_prompt(OLD_CHUNK["text"], CHUNK["text"])
    assert "VERBATIM" in prompt
    assert "2024-03-01" not in prompt and "regulation-2024.md" not in prompt


@pytest.mark.parametrize(
    "completion",
    [
        '{"relation": "contradicts", "confidence": 0.9, "claim_a": "a", "claim_b": "b"}',
        '```json\n{"relation": "contradicts", "confidence": 0.9, "claim_a": "a",'
        ' "claim_b": "b"}\n```',
        'Here you go:\n{"relation": "CONTRADICTS", "confidence": 0.9, "claim_a": "a",'
        ' "claim_b": "b"}\nHope that helps.',
    ],
)
def test_parse_tolerates_fences_and_prose(completion):
    assert parse_adjudication(completion)["relation"] == REL_CONTRADICTS


@pytest.mark.parametrize(
    "completion",
    ["not json at all", '{"relation": "sort-of-related"}', '["contradicts"]', "{}"],
)
def test_parse_rejects_unusable_verdicts(completion):
    with pytest.raises(AdjudicationError):
        parse_adjudication(completion)


def test_confidence_is_clamped():
    parsed = parse_adjudication(
        '{"relation": "duplicate", "confidence": 5, "claim_a": "", "claim_b": ""}'
    )
    assert parsed["confidence"] == 1.0


def test_quoted_claim_narrows_to_exact_corpus_offsets():
    quote = "становить п'ятнадцять робочих днів"
    ref = narrow_to_claim(CHUNK, quote, NEW_GOV)
    assert ref.offsets_exact
    assert ref.text == quote
    assert (
        CHUNK["text"][ref.char_start - CHUNK["char_start"] : ref.char_end - CHUNK["char_start"]]
        == quote
    )


def test_unlocatable_quote_falls_back_to_the_chunk_and_says_so():
    ref = narrow_to_claim(CHUNK, "цього речення тут немає взагалі", NEW_GOV)
    assert not ref.offsets_exact
    assert (ref.char_start, ref.char_end) == (CHUNK["char_start"], CHUNK["char_end"])


def test_empty_quote_is_inexact_rather_than_an_error():
    assert not narrow_to_claim(CHUNK, "", NEW_GOV).offsets_exact


def _ref(doc_id, governance):
    return ClaimRef(doc_id=doc_id, char_start=0, char_end=1, text="x", governance=governance)


def test_dated_contradiction_becomes_supersession_with_the_stale_claim_first():
    old, new = _ref("regulation-2021.md", OLD_GOV), _ref("regulation-2024.md", NEW_GOV)
    relation, a, b, staleness = apply_supersession(REL_CONTRADICTS, old, new)
    assert relation == REL_SUPERSEDED_BY
    assert a.doc_id == "regulation-2021.md", "side a is the deprecated claim"
    assert staleness.newer_side == "b"


def test_supersession_reorders_when_the_newer_document_came_first():
    old, new = _ref("regulation-2021.md", OLD_GOV), _ref("regulation-2024.md", NEW_GOV)
    relation, a, b, staleness = apply_supersession(REL_CONTRADICTS, new, old)
    assert (relation, a.doc_id, staleness.newer_side) == (
        REL_SUPERSEDED_BY,
        "regulation-2021.md",
        "b",
    )


def test_undated_contradiction_stays_a_contradiction():
    relation, _, _, staleness = apply_supersession(
        REL_CONTRADICTS, _ref("a.md", {}), _ref("b.md", {})
    )
    assert relation == REL_CONTRADICTS
    assert staleness.newer_side is None


def test_non_contradictions_are_never_promoted():
    for relation in (REL_DUPLICATE, REL_SUBSUMED_BY, REL_COMPLEMENTARY):
        got, _, _, _ = apply_supersession(
            relation, _ref("regulation-2021.md", OLD_GOV), _ref("regulation-2024.md", NEW_GOV)
        )
        assert got == relation


def test_partial_supersession_yields_one_relation_per_claim_pair():
    """The revision supersedes the fact it changed and duplicates the fact it restated."""
    restated_old = {**OLD_CHUNK, "text": "Звернення реєструється у день його надходження."}
    restated_new = {**CHUNK, "text": "Звернення реєструється у день його надходження."}
    chunks = [OLD_CHUNK, CHUNK, restated_old, restated_new]
    governance = {"regulation-2021.md": OLD_GOV, "regulation-2024.md": NEW_GOV}

    def complete(prompt: str) -> str:
        if "тридцять" in prompt:
            return (
                '{"relation": "contradicts", "confidence": 0.95,'
                ' "claim_a": "становить тридцять календарних днів",'
                ' "claim_b": "становить п\'ятнадцять робочих днів",'
                ' "rationale": "different deadlines"}'
            )
        return (
            '{"relation": "duplicate", "confidence": 0.99,'
            ' "claim_a": "Звернення реєструється у день його надходження.",'
            ' "claim_b": "Звернення реєструється у день його надходження.",'
            ' "rationale": "same rule"}'
        )

    findings, stats = adjudicate_pairs([(0, 1, 0.96), (2, 3, 0.99)], chunks, governance, complete)
    assert {f.relation for f in findings} == {REL_SUPERSEDED_BY, REL_DUPLICATE}
    assert (stats.extra["model_calls"], stats.extra["unparsed_verdicts"]) == (2, 0)
    superseded = next(f for f in findings if f.relation == REL_SUPERSEDED_BY)
    assert superseded.a.doc_id == "regulation-2021.md"
    assert superseded.a.offsets_exact and superseded.b.offsets_exact


def test_unparsed_verdicts_are_counted_and_skipped_not_fatal():
    findings, stats = adjudicate_pairs(
        [(0, 1, 0.9)], [OLD_CHUNK, CHUNK], {}, lambda prompt: "the model rambled"
    )
    assert findings == []
    assert stats.extra["unparsed_verdicts"] == 1


def test_adjudication_is_deterministic_under_a_fixed_completion():
    chunks = [OLD_CHUNK, CHUNK]
    governance = {"regulation-2021.md": OLD_GOV, "regulation-2024.md": NEW_GOV}
    response = (
        '{"relation": "contradicts", "confidence": 0.9,'
        ' "claim_a": "становить тридцять календарних днів",'
        ' "claim_b": "становить п\'ятнадцять робочих днів", "rationale": "r"}'
    )
    first, _ = adjudicate_pairs([(0, 1, 0.9)], chunks, governance, lambda p: response)
    second, _ = adjudicate_pairs([(0, 1, 0.9)], chunks, governance, lambda p: response)
    assert [f.payload() for f in first] == [f.payload() for f in second]
