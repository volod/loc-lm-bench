"""The adoption study's contracts: the promoted policy field and the committed workload geometry.

The live run is a two-family GPU measurement, so what CI can own is everything around it: that the
trim strategy is a real `ContextPolicy` choice rather than a study parameter, that the committed
workload set still produces the regimes it declares, and that the arms are byte-identical exactly
where the fold fits. What the resulting numbers then LICENSE is
`agentic_summary_trim_verdict`.
"""

import pytest

from llb.bench.agentic.context_policy import (
    DEFAULT_SUMMARY_TRIM_STRATEGY,
    POLICY_COMPACT,
    SUMMARY_TRIM_HEAD_TAIL,
    SUMMARY_TRIM_PER_ENTRY_HEAD,
    SUMMARY_TRIM_STRATEGIES,
    ContextPolicy,
)
from llb.bench.policy_change.audit import AUDITABLE_FIELDS
from llb.bench.summary_trim.analysis import (
    analyze_summary_trim_runs,
    audit_default_change,
    family_eligibility,
)
from llb.bench.summary_trim.design import (
    ARMS,
    probe_workload,
    validate_summary_trim_design,
    workloads,
)
from llb.bench.summary_trim.adoption import ADOPT_INELIGIBLE
from llb.bench.context_policy.interleave import ORDER_ALTERNATING, ORDER_FIXED
from llb.bench.summary_trim.reading import WORKLOAD_UNCHANGED
from llb.bench.summary_trim.report import format_summary_trim_table
from llb.bench.summary_trim.workloads import workload_tasks


@pytest.fixture(scope="module")
def design(adoption_design: dict[str, object]) -> dict[str, object]:
    return adoption_design


def test_the_entry_aware_trim_is_the_shipped_default():
    """The adoption the study licensed, read off the dataclass an episode actually builds."""
    assert DEFAULT_SUMMARY_TRIM_STRATEGY == SUMMARY_TRIM_PER_ENTRY_HEAD
    assert ContextPolicy().summary_trim_strategy == SUMMARY_TRIM_PER_ENTRY_HEAD
    assert ContextPolicy(name=POLICY_COMPACT).summary_trim_strategy == SUMMARY_TRIM_PER_ENTRY_HEAD


def test_the_trim_strategy_is_a_validated_context_policy_choice():
    """Promoted, not a study parameter: it has a default, a vocabulary, and a refusal."""
    assert DEFAULT_SUMMARY_TRIM_STRATEGY in SUMMARY_TRIM_STRATEGIES
    assert set(SUMMARY_TRIM_STRATEGIES) == {SUMMARY_TRIM_HEAD_TAIL, SUMMARY_TRIM_PER_ENTRY_HEAD}
    assert (
        ContextPolicy(
            name=POLICY_COMPACT, summary_trim_strategy=SUMMARY_TRIM_HEAD_TAIL
        ).summary_trim_strategy
        == SUMMARY_TRIM_HEAD_TAIL
    )
    with pytest.raises(ValueError, match="unknown summary trim strategy"):
        ContextPolicy(name=POLICY_COMPACT, summary_trim_strategy="per_entry")


def test_the_audit_the_verdict_reads_runs_off_the_shipped_default():
    """The move is checked from the side the product is on, not the side the study started on."""
    from llb.bench.summary_trim.analysis import (
        RETIRED_SUMMARY_TRIM_STRATEGY,
        audit_default_change,
    )

    assert RETIRED_SUMMARY_TRIM_STRATEGY == SUMMARY_TRIM_HEAD_TAIL
    audit = audit_default_change()
    assert audit["shipped_default"] == DEFAULT_SUMMARY_TRIM_STRATEGY
    assert audit["change"].startswith(  # type: ignore[union-attr]
        f"summary_trim_strategy {DEFAULT_SUMMARY_TRIM_STRATEGY!r} -> "
    )
    # The pinned policy the replay runs under is the shipped one, and the reverse read reports the
    # same invariance the forward read did: every applicable cell, and no published arithmetic.
    assert audit["pinned_policy"]["summary_trim_strategy"] == DEFAULT_SUMMARY_TRIM_STRATEGY  # type: ignore[index]
    assert audit["invariant"] is True
    assert audit["n_invalidated"] == 0 and audit["n_prompt_invariant"] == audit["n_cells"]
    assert audit["affected_published_values"] == []


def test_the_promoted_field_is_audited_and_pinned_like_every_other_constant():
    """A policy field the audit cannot see is a field a default change could move silently."""
    assert "summary_trim_strategy" in AUDITABLE_FIELDS
    assert set(AUDITABLE_FIELDS) <= set(ContextPolicy.__dataclass_fields__)


def test_the_committed_design_still_produces_the_regimes_it_declares(design: dict[str, object]):
    """The whole design gate: every workload's fold count, offered span, elision, and both costs."""
    validate_summary_trim_design(design)


def test_a_fold_that_fits_renders_byte_identical_prompts(design: dict[str, object]):
    """The control workload's claim, measured: no elision means the strategies cannot differ."""
    held = design["held_fixed"]
    control = next(
        row
        for row in workloads(design)
        if row["expected"]["elided_chars"] == 0  # type: ignore[index]
    )
    measured = probe_workload(control, held)  # type: ignore[arg-type]
    assert measured[SUMMARY_TRIM_HEAD_TAIL] == measured[SUMMARY_TRIM_PER_ENTRY_HEAD]


def test_every_eliding_workload_spends_at_most_the_shipped_summary_bytes(
    design: dict[str, object],
):
    """The cost side, decided with no model: entry-aware never buys its recovery with bytes."""
    held = design["held_fixed"]
    for workload in workloads(design):
        measured = probe_workload(workload, held)  # type: ignore[arg-type]
        head_tail = measured[SUMMARY_TRIM_HEAD_TAIL]["compaction_prompt_chars"]  # type: ignore[index]
        entry_aware = measured[SUMMARY_TRIM_PER_ENTRY_HEAD]["compaction_prompt_chars"]  # type: ignore[index]
        assert entry_aware <= head_tail, workload["workload"]


def test_the_arms_offer_the_summarizer_the_identical_transcript(design: dict[str, object]):
    """What makes the comparison paired: the two arms only diverge AFTER the fold's input."""
    held = design["held_fixed"]
    for workload in workloads(design):
        measured = probe_workload(workload, held)  # type: ignore[arg-type]
        assert (
            measured[SUMMARY_TRIM_HEAD_TAIL]["summary_fold_input_chars"]  # type: ignore[index]
            == measured[SUMMARY_TRIM_PER_ENTRY_HEAD]["summary_fold_input_chars"]  # type: ignore[index]
        )


def test_a_run_over_the_declared_oracles_pairs_every_workload(design, oracle_family):
    """End to end over injected oracles: rows, pairing, eligibility, and a rendered table."""
    run = oracle_family("fixture")
    assert len(run.rows) == len(workloads(design)) * len(ARMS)
    eligible, reason = family_eligibility(design, run)
    assert eligible, reason
    analysis = analyze_summary_trim_runs(design, [run], audit=audit_default_change())
    readings = analysis["families"][0]["workloads"]  # type: ignore[index]
    assert [row["reading"] for row in readings] == [WORKLOAD_UNCHANGED] * len(workloads(design))
    assert "declared geometry" in format_summary_trim_table(analysis)
    # One qualified family is not two, and the verdict says so rather than reading the one it has.
    assert analysis["adoption_reading"] == ADOPT_INELIGIBLE


def test_the_repeatedly_folding_workload_is_where_the_strategies_cost_differently(
    design, oracle_family
):
    """The only workload whose two arms spend different bytes, and it spends FEWER."""
    run = oracle_family("fixture")
    analysis = analyze_summary_trim_runs(design, [run], audit=audit_default_change())
    deltas = {
        row["workload"]: row["d_summary_prompt_chars"]
        for row in analysis["families"][0]["workloads"]  # type: ignore[index]
    }
    assert deltas["repeated_fold"] < 0
    assert all(value == 0 for name, value in deltas.items() if name != "repeated_fold")


def test_the_design_declares_the_execution_order_rather_than_inheriting_it(
    design: dict[str, object],
):
    """A fixed schedule is not readable as evidence, so the design has to state a balanced one."""
    declared = design["arm_order"]
    assert declared["policy"] == ORDER_ALTERNATING  # type: ignore[index]
    assert declared["phase_by_family"] is True  # type: ignore[index]
    fixed = {**design, "arm_order": {"policy": ORDER_FIXED, "phase_by_family": True}}
    with pytest.raises(ValueError, match=ORDER_ALTERNATING):
        validate_summary_trim_design(fixed)
    with pytest.raises(ValueError, match="phase per family"):
        validate_summary_trim_design(
            {**design, "arm_order": {"policy": ORDER_ALTERNATING, "phase_by_family": False}}
        )


def test_a_run_balances_the_arm_order_across_every_workload(design, oracle_family):
    """Both arms of a task run adjacently, and neither arm holds the first position twice over."""
    run = oracle_family("fixture")
    n_tasks = sum(len(workload_tasks(workload)) for workload in workloads(design))
    assert len(run.schedule) == n_tasks * len(ARMS)
    firsts = [row for row in run.schedule if row["position"] == 1]
    held = [sum(1 for row in firsts if row["arm"] == arm) for arm in ARMS]
    assert abs(held[0] - held[1]) <= 1, held
    # Adjacent: consecutive schedule rows come in same-task pairs.
    pairs = list(zip(run.schedule[::2], run.schedule[1::2]))
    assert all(one["item_id"] == two["item_id"] for one, two in pairs)
    assert all(one["arm"] != two["arm"] for one, two in pairs)


def test_the_measured_reading_states_the_order_it_ran_under(design, oracle_family):
    """The order is carried into the reading, so a verdict can gate on it instead of assuming."""
    run = oracle_family("fixture")
    analysis = analyze_summary_trim_runs(design, [run], audit=audit_default_change())
    order = analysis["families"][0]["arm_order"]  # type: ignore[index]
    assert order["n_first_head_tail"] + order["n_first_per_entry_head"] == order["n_first"]
    assert order["n_first"] == order["n_second"]
    assert "arm order" in format_summary_trim_table(analysis)
