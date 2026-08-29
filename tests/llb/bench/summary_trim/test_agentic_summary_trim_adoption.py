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
    load_summary_trim_design,
    probe_workload,
    validate_summary_trim_design,
    workloads,
)
from llb.bench.summary_trim.adoption import ADOPT_INELIGIBLE
from llb.bench.context_policy.interleave import ORDER_ALTERNATING, ORDER_FIXED
from llb.bench.summary_trim.reading import WORKLOAD_UNCHANGED
from llb.bench.summary_trim.report import format_summary_trim_table
from llb.bench.summary_trim.run import run_summary_trim_family
from llb.bench.summary_trim.workloads import (
    build_workload_tasks,
    workload_oracle,
    workload_tasks,
)


@pytest.fixture(scope="module")
def design() -> dict[str, object]:
    return load_summary_trim_design()


# The step prompt carries this header only once the episode has a history to render, so a prompt
# WITHOUT it is the first step of a fresh episode.
_HISTORY_HEADER = "Виконані кроки:"


def _oracle_complete(design: dict[str, object]):
    """One fake `complete` that plays whichever declared task the loop is currently walking.

    An oracle is per EPISODE -- the aggregate-search one counts the queries it has issued -- and
    the runner walks each task twice back to back, once per arm, so a fresh oracle is taken when
    the prompt shows no history rather than when the task changes.
    """
    oracles = {
        record["prompt"][:60]: (workload, record)
        for workload in workloads(design)
        for record in build_workload_tasks(workload)
    }
    state: dict[str, object] = {"oracle": None}

    def complete(prompt: str) -> str:
        key = next((candidate for candidate in oracles if candidate in prompt), None)
        if key is not None and _HISTORY_HEADER not in prompt:
            state["oracle"] = workload_oracle(*oracles[key])
        oracle = state["oracle"]
        if oracle is None:
            return '{"name": "finish", "arguments": {"answer": ""}}'
        return oracle(prompt)  # type: ignore[operator]

    return complete


def _family(design: dict[str, object], name: str):
    return run_summary_trim_family(
        design,
        {"model_family": name, "model": name, "backend": "ollama"},
        complete=_oracle_complete(design),
    )


def test_the_trim_strategy_is_a_validated_context_policy_choice():
    """Promoted, not a study parameter: it has a default, a vocabulary, and a refusal."""
    assert DEFAULT_SUMMARY_TRIM_STRATEGY == SUMMARY_TRIM_HEAD_TAIL
    assert set(SUMMARY_TRIM_STRATEGIES) == {SUMMARY_TRIM_HEAD_TAIL, SUMMARY_TRIM_PER_ENTRY_HEAD}
    assert ContextPolicy().summary_trim_strategy == SUMMARY_TRIM_HEAD_TAIL
    assert (
        ContextPolicy(
            name=POLICY_COMPACT, summary_trim_strategy=SUMMARY_TRIM_PER_ENTRY_HEAD
        ).summary_trim_strategy
        == SUMMARY_TRIM_PER_ENTRY_HEAD
    )
    with pytest.raises(ValueError, match="unknown summary trim strategy"):
        ContextPolicy(name=POLICY_COMPACT, summary_trim_strategy="per_entry")


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


def test_a_run_over_the_declared_oracles_pairs_every_workload(design: dict[str, object]):
    """End to end over injected oracles: rows, pairing, eligibility, and a rendered table."""
    run = _family(design, "fixture")
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
    design: dict[str, object],
):
    """The only workload whose two arms spend different bytes, and it spends FEWER."""
    run = _family(design, "fixture")
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


def test_a_run_balances_the_arm_order_across_every_workload(design: dict[str, object]):
    """Both arms of a task run adjacently, and neither arm holds the first position twice over."""
    run = _family(design, "fixture")
    n_tasks = sum(len(workload_tasks(workload)) for workload in workloads(design))
    assert len(run.schedule) == n_tasks * len(ARMS)
    firsts = [row for row in run.schedule if row["position"] == 1]
    held = [sum(1 for row in firsts if row["arm"] == arm) for arm in ARMS]
    assert abs(held[0] - held[1]) <= 1, held
    # Adjacent: consecutive schedule rows come in same-task pairs.
    pairs = list(zip(run.schedule[::2], run.schedule[1::2]))
    assert all(one["item_id"] == two["item_id"] for one, two in pairs)
    assert all(one["arm"] != two["arm"] for one, two in pairs)


def test_the_measured_reading_states_the_order_it_ran_under(design: dict[str, object]):
    """The order is carried into the reading, so a verdict can gate on it instead of assuming."""
    run = _family(design, "fixture")
    analysis = analyze_summary_trim_runs(design, [run], audit=audit_default_change())
    order = analysis["families"][0]["arm_order"]  # type: ignore[index]
    assert order["n_first_head_tail"] + order["n_first_per_entry_head"] == order["n_first"]
    assert order["n_first"] == order["n_second"]
    assert "arm order" in format_summary_trim_table(analysis)
