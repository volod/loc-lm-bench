"""Setting grid, noisy/clean alignment, and the per-edit precision labels."""

from llb.eval.restoration_sweep import policy_grid
from llb.eval.restoration_sweep_audit import (
    LABEL_CORRECT,
    LABEL_UNALIGNED,
    LABEL_WRONG,
    AuditCounts,
    CaseAlignment,
    audit_case,
)
from llb.rag.query_prep.base import KIND_TYPO, STEP_NORMALIZE, STEP_TYPOS, QueryEdit
from llb.rag.query_prep.restore_policy import DEFAULT_RESTORATION_POLICY

VOCABULARY = frozenset({"наказ", "герцогство", "засновано", "коли"})


def _edit(original: str, replacement: str, step: str = STEP_TYPOS) -> QueryEdit:
    return QueryEdit(step, KIND_TYPO, original=original, replacement=replacement)


def test_one_factor_grid_starts_at_the_default_and_varies_one_constant_each():
    grid = policy_grid([0, 1], [3, 4, 5], ["morphology", "context"])
    assert grid[0] == DEFAULT_RESTORATION_POLICY
    varied = [
        sum(
            getattr(policy, name) != getattr(DEFAULT_RESTORATION_POLICY, name)
            for name in ("surface_max_distance", "ambiguous_token_max_chars", "rank_order")
        )
        for policy in grid[1:]
    ]
    assert varied == [1, 1, 1, 1]
    assert len(set(grid)) == len(grid)


def test_full_grid_measures_the_product_with_the_default_first():
    grid = policy_grid([0, 1], [3, 4], ["morphology", "context"], full=True)
    assert grid[0] == DEFAULT_RESTORATION_POLICY
    assert len(grid) == 8
    assert len(set(grid)) == 8


def test_alignment_pairs_noisy_tokens_with_the_clean_tokens_they_came_from():
    alignment = CaseAlignment.build("коли засновано герцогство", "крли замновано герцогство")
    assert alignment.references["крли"] == "коли"
    assert alignment.opportunities(VOCABULARY) == (("крли", "коли"), ("замновано", "засновано"))


def test_alignment_refuses_to_judge_a_token_sequence_that_does_not_correspond():
    alignment = CaseAlignment.build("коли засновано герцогство", "крли замновано")
    assert alignment.pairs is None
    assert alignment.references == {}
    assert alignment.opportunities(VOCABULARY) == ()


def test_corrections_are_labeled_against_the_clean_token_the_noise_came_from():
    alignment = CaseAlignment.build("коли засновано герцогство", "крли замновано герцогство")
    records, counts = audit_case(
        setting="surface=0,short=4,rank=morphology",
        variant_class="keyboard_typos",
        item_id="q1",
        edits=[
            _edit("крли", "коли"),
            _edit("замновано", "наказ"),
            _edit("невідоме", "наказ"),
            _edit("герцогство", "герцогство", step=STEP_NORMALIZE),
        ],
        alignment=alignment,
        vocabulary=VOCABULARY,
    )
    assert [record.label for record in records] == [LABEL_CORRECT, LABEL_WRONG, LABEL_UNALIGNED]
    assert counts.corrections == 3
    assert (counts.correct, counts.wrong, counts.unaligned) == (1, 1, 1)
    assert counts.labeled == 2
    assert counts.wrong_share == 0.5
    # both noised tokens were restorable; only one of them was actually restored
    assert (counts.opportunities, counts.restored) == (2, 1)
    assert counts.restoration_recall == 0.5


def test_a_refused_restoration_is_a_missed_opportunity_rather_than_a_wrong_edit():
    alignment = CaseAlignment.build("коли засновано", "крли засновано")
    _, counts = audit_case(
        setting="surface=0,short=4,rank=morphology",
        variant_class="keyboard_typos",
        item_id="q1",
        edits=[],
        alignment=alignment,
        vocabulary=VOCABULARY,
    )
    assert (counts.corrections, counts.opportunities, counts.restored) == (0, 1, 0)
    assert counts.wrong_share == 0.0
    assert counts.restoration_recall == 0.0


def test_counts_sum_across_classes_for_the_pooled_row():
    left = AuditCounts(corrections=2, correct=1, wrong=1, opportunities=3, restored=1)
    right = AuditCounts(corrections=1, correct=1, opportunities=1, restored=1)
    total = left + right
    assert (total.corrections, total.correct, total.wrong) == (3, 2, 1)
    assert total.wrong_share == 1 / 3
    assert total.restoration_recall == 0.5
