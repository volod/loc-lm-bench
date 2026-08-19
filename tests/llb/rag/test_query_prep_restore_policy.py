"""Every swept restoration setting's selection decisions over the committed candidate fixture.

The three constants in `restore_policy` are design choices, not preferences, so what each value
DOES has to be pinned in CI: the sweep on the CUDA host then measures which of those behaviors is
worth having. `tests/fixtures/restoration_candidates.json` holds one candidate set per constant,
written so exactly one constant changes its outcome.
"""

import json
from pathlib import Path

import pytest

from llb.eval.restoration_sweep import policy_grid
from llb.rag.query_prep.base import KIND_TRANSLITERATE
from llb.rag.query_prep.pipeline import QueryPrep
from llb.rag.query_prep.restore import (
    TokenProvenance,
    VocabularyContext,
    select_restoration,
    surface_distance,
)
from llb.rag.query_prep.restore_policy import (
    DEFAULT_RESTORATION_POLICY,
    RANK_CONTEXT,
    RestorationPolicy,
)
from llb.rag.query_prep.typos import apply_typos, build_vocabulary

FIXTURE = Path(__file__).parents[2] / "fixtures" / "restoration_candidates.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))
SWEPT_POLICIES = {
    policy.label: policy for policy in policy_grid([0, 1], [3, 4, 5], ["morphology", "context"])
}


def _select(case: dict, policy: RestorationPolicy) -> str | None:
    provenance = case.get("provenance")
    known = case.get("known_words")
    chunks = case.get("context_chunks")
    context = VocabularyContext.build(chunks) if chunks else None
    anchors = context.anchors(case["context_query"]) if context is not None else ()
    return select_restoration(
        case["token"],
        [(distance, candidate) for distance, candidate in case["candidates"]],
        provenance=(
            TokenProvenance(provenance["noisy"], provenance["kind"]) if provenance else None
        ),
        known_word=set(known).__contains__ if known else None,
        context=context,
        anchors=anchors,
        policy=policy,
    )


def test_the_fixture_covers_exactly_the_settings_the_sweep_measures():
    assert set(CASES["settings"]) == set(SWEPT_POLICIES)
    assert CASES["settings"][0] == DEFAULT_RESTORATION_POLICY.label


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda case: case["id"])
def test_every_setting_makes_the_committed_selection(case):
    for label, expected in case["expected"].items():
        assert _select(case, SWEPT_POLICIES[label]) == expected, f"{case['id']} under {label}"


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda case: case["id"])
def test_each_case_isolates_one_constant(case):
    """A case must be decided by its own constant: every other setting matches the default."""
    default = case["expected"][DEFAULT_RESTORATION_POLICY.label]
    off_axis = [
        label
        for label, policy in SWEPT_POLICIES.items()
        if policy != DEFAULT_RESTORATION_POLICY
        and getattr(policy, case["constant"])
        == getattr(DEFAULT_RESTORATION_POLICY, case["constant"])
    ]
    assert off_axis
    for label in off_axis:
        assert case["expected"][label] == default


def test_surface_budget_bounds_the_reported_distance():
    typed = TokenProvenance("sut", KIND_TRANSLITERATE)
    assert surface_distance("суд", typed, 0) > 0
    assert surface_distance("суд", typed, 1) == 1


def test_the_policy_reaches_the_typos_step_through_the_pipeline():
    vocabulary = build_vocabulary(["слов зникло"])
    lenient = RestorationPolicy(ambiguous_token_max_chars=3)
    assert apply_typos("слово", vocabulary)[0] == "слов"
    assert (
        apply_typos("слово", vocabulary, policy=RestorationPolicy(ambiguous_token_max_chars=5))[0]
        == "слово"
    )
    pipeline = QueryPrep.build(
        ("normalize", "typos"), vocabulary=vocabulary, restoration_policy=lenient
    )
    assert pipeline.restoration_policy == lenient
    assert pipeline.process("Слово").processed == "слов"


def test_a_non_default_policy_requires_the_step_it_constrains():
    with pytest.raises(ValueError, match="restoration policy"):
        QueryPrep.build(
            ("normalize",), restoration_policy=RestorationPolicy(rank_order=RANK_CONTEXT)
        )


def test_a_policy_refuses_a_value_outside_its_measured_range():
    with pytest.raises(ValueError, match="surface_max_distance"):
        RestorationPolicy(surface_max_distance=3)
    with pytest.raises(ValueError, match="rank_order"):
        RestorationPolicy(rank_order="alphabetical")
