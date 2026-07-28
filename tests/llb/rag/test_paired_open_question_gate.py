"""re-decide-the-relabeled-fusion-and-bakeoff-readings -- a withdrawn reading says what it needs.

The minimum-evidence gate turns a claim that rests on too few differing items into
`insufficient_evidence`. That is a statement about the ITEM SET, not about the difference, so a
recommendation an operator may still be acting on must not be left reading like a measured tie.
Every withdrawn row is therefore priced from its own discordance rate: `d` differing items out of
`n` extrapolate to the item count at which the reporting level becomes reachable at all.

Covered: the arithmetic and its two boundaries (already reachable, nothing differing), that the
floor moves with the reporting convention exactly as the bound it inverts does, the shared clause
every lane appends, the per-row column in the boundary table, and the recorded open questions whose
prices the current docs quote -- including the encoder row, which no committed goldset can reach.

Pure: value vectors and dict rows, so the whole vertical runs in the lightweight CI install.
"""

import pytest


from llb.rag.fusion_evidence.evidence_gate import (
    READING_FLAT,
    READING_INSUFFICIENT_EVIDENCE,
    evidence_gate_note,
    minimum_discordant_pairs,
    open_question_note,
    resolving_item_count,
)


from llb.rag.fusion_evidence.stability import (
    LOOSER_CONFIDENCE,
    TIGHTER_CONFIDENCE,
    boundary_table,
)


from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    bootstrap_index_sets,
)


from llb.rag.fusion_evidence.paired import (
    compared_pairs,
    evidence_gate_clause,
    paired_comparison,
    reading_of,
)


from _paired_open_question_helpers import (
    RESAMPLES,
    SEED,
    BOUND,
    GOLDSETS,
    _unanimous,
)


@pytest.mark.parametrize(("discordant", "pairs"), [(4, 35), (5, 35), (5, 250), (1, 7), (3, 1000)])
def test_the_floor_is_the_smallest_item_count_whose_rate_reaches_the_bound(
    discordant: int, pairs: int
):
    """Necessary AND smallest: one item fewer would still not price the reachable minimum."""
    required = resolving_item_count(discordant, pairs)
    rate = discordant / pairs
    assert required is not None
    assert required * rate >= BOUND
    assert (required - 1) * rate < BOUND


def test_the_recorded_open_questions_keep_the_prices_the_docs_quote():
    """The three shapes the re-decision states, so a doc number cannot drift from the rule."""
    assert resolving_item_count(4, 35) == 53  # the fusion sweeps' deciding span-identity row
    assert resolving_item_count(5, 35) == 42  # the routed answer-quality coverage row
    assert resolving_item_count(5, 250) == 300  # the bake-off's `e5-large` recall bar


def test_the_encoder_question_is_undecidable_on_every_committed_goldset():
    """The claim the bake-off section makes: no item set this repo has reaches that floor."""
    sizes = {
        path.parent.name: sum(1 for _ in path.open(encoding="utf-8"))
        for path in GOLDSETS.glob("*/goldset.jsonl")
    }
    assert sizes, "the committed goldsets are what bound the claim"
    assert max(sizes.values()) < resolving_item_count(5, 250)


def test_a_row_needing_nothing_and_a_row_pricing_nothing_are_both_unpriced():
    """`None` twice over, for opposite reasons -- neither may render as an item count."""
    assert resolving_item_count(BOUND, 40) is None  # already reachable
    assert resolving_item_count(BOUND + 3, 40) is None
    assert resolving_item_count(0, 40) is None  # no rate to extrapolate
    assert resolving_item_count(4, 0) is None


def test_the_floor_moves_with_the_reporting_convention():
    """It inverts the bound, so a tighter level costs items and a looser one refunds them."""
    looser = resolving_item_count(4, 35, LOOSER_CONFIDENCE)
    tighter = resolving_item_count(4, 35, TIGHTER_CONFIDENCE)
    # The trio the paired-uncertainty docs quote for the fusion sweeps' deciding row.
    assert (looser, resolving_item_count(4, 35), tighter) == (44, 53, 62)
    assert looser * (4 / 35) >= minimum_discordant_pairs(LOOSER_CONFIDENCE)
    assert tighter * (4 / 35) >= minimum_discordant_pairs(TIGHTER_CONFIDENCE)


def test_the_clause_states_the_question_and_its_price_together():
    marked = [("recall_at_k", 4, 35)]
    clause = open_question_note(marked)
    assert "OPEN QUESTION" in clause and "`recall_at_k` 4 of 35 -> 53 items" in clause
    assert clause.isascii()
    # Not a floor an operator may read as "run this many items and it is settled".
    assert "not a minimum-detectable-effect target" in clause
    # A reachable row leaves no open question at all.
    assert open_question_note([("recall_at_k", BOUND, 35)]) == ""


def test_no_withdrawn_reading_is_stated_without_what_would_settle_it():
    """The two halves are one clause: every insufficient-evidence sentence carries a price."""
    clause = evidence_gate_note([("recall_at_k", 4, 35), ("mrr", 2, 35)])
    assert "INSUFFICIENT EVIDENCE" in clause and "OPEN QUESTION" in clause
    assert "4 of 35 -> 53 items" in clause and "2 of 35 -> 105 items" in clause
    row = _unanimous(BOUND - 1)
    assert reading_of(row) == READING_INSUFFICIENT_EVIDENCE
    lane_clause = evidence_gate_clause([("recall_at_k", row)])
    assert "OPEN QUESTION" in lane_clause
    assert f"{BOUND - 1} of {compared_pairs(row)} -> 48 items" in lane_clause


def test_the_boundary_table_prices_a_thin_row_and_leaves_a_settled_one_alone():
    thin = _unanimous(BOUND - 1)["stability"]
    settled = _unanimous(BOUND + 6)["stability"]
    assert thin["pairs"] == settled["pairs"] == 40
    lines = boundary_table(
        [("thin", thin), ("settled", settled)],
        title="Where each reading sits",
        key_header="row",
        subject="the candidate",
    )
    text = "\n".join(lines)
    assert "n to reach" in text and text.isascii()
    thin_row = next(line for line in lines if line.startswith("| thin "))
    settled_row = next(line for line in lines if line.startswith("| settled "))
    assert thin_row.split("|")[-3].strip() == "48"
    assert settled_row.split("|")[-3].strip() == "-"


def test_a_flat_row_too_thin_to_have_shown_anything_is_priced_as_well():
    """`flat` on three differing items is not a measured tie, and the column says which it is."""
    n = 40
    candidate = [1.0, 1.0, 0.0] + [0.0] * (n - 3)
    baseline = [0.0, 0.0, 1.0] + [0.0] * (n - 3)
    row = paired_comparison(
        candidate, baseline, bootstrap_index_sets(n, RESAMPLES, SEED), DEFAULT_CONFIDENCE
    )
    assert row["stability"]["reading"] == READING_FLAT
    lines = boundary_table(
        [("thin flat", row["stability"])],
        title="Where each reading sits",
        key_header="row",
        subject="the candidate",
    )
    priced = next(line for line in lines if line.startswith("| thin flat "))
    assert priced.split("|")[-3].strip() == str(resolving_item_count(3, n))
    # The clause stays quiet, though: a verdict is not decided on a row that never cleared zero.
    assert evidence_gate_clause([("recall_at_k", row)]) == ""


def test_a_reading_recorded_before_the_pair_count_existed_prices_nothing():
    """An archived `stability` block carries no `pairs`; the column must say so, not guess."""
    archived = dict(_unanimous(BOUND - 1)["stability"])
    archived.pop("pairs")
    lines = boundary_table(
        [("archived", archived)],  # type: ignore[list-item]
        title="Where each reading sits",
        key_header="row",
        subject="the candidate",
    )
    assert next(line for line in lines if line.startswith("| archived ")).split("|")[
        -3
    ].strip() == ("-")


def test_an_archived_uncalibrated_boundary_table_keeps_its_columns_aligned():
    archived = dict(_unanimous(BOUND + 6)["stability"])
    archived.pop("randomization_p")
    archived.pop("randomization_method")
    archived.pop("randomization_samples")
    lines = boundary_table(
        [("archived", archived)],  # type: ignore[list-item]
        title="Where each reading sits",
        key_header="row",
        subject="the candidate",
    )
    header = next(line for line in lines if line.startswith("| row "))
    separator = lines[lines.index(header) + 1]
    row = next(line for line in lines if line.startswith("| archived "))
    assert header.count("|") == separator.count("|") == row.count("|")
