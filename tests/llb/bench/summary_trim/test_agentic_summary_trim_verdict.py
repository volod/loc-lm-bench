"""How the adoption verdict reads a measured run: pairing semantics, and what each gate refuses.

Split from the design contracts because these are statements about the READING -- which cases pair,
which unpaired ones are about the treatment, and which of the four verdicts a given shape of
evidence licenses -- and they need no design, roster, or geometry to make.
"""

from llb.bench.summary_trim.adoption import (
    ADOPT_AS_DEFAULT,
    ADOPT_AS_OPTION,
    ADOPT_INCONCLUSIVE,
    ADOPT_REFUSE,
    adoption_reading,
)
from llb.bench.summary_trim.reading import (
    WORKLOAD_RECOVERS,
    WORKLOAD_UNCHANGED,
    WORKLOAD_UNPAIRED,
    workload_reading,
)


def _stub_family(name: str, *, middle: dict[str, int], workload_rows: list[dict[str, object]]):
    return {
        "model_family": name,
        "model": name,
        "workloads": workload_rows,
        "strata": {"head": {}, "middle": middle, "tail": {}},
    }


def _row(name: str, reading: str, *, summary_delta: int = 0) -> dict[str, object]:
    return {
        "workload": name,
        "reading": reading,
        "d_summary_prompt_chars": summary_delta,
        "n_unpaired_no_fold": 0,
        "n_unpaired_divergent_fold": 0,
    }


_WHOLE_MIDDLE = {
    "n_declared": 2,
    "n_pairs": 2,
    "entry_aware_wins": 2,
    "head_tail_wins": 0,
    "head_tail_completed": 0,
    "entry_aware_completed": 2,
}


def _clean(**kwargs):
    families = [
        _stub_family(name, middle=dict(_WHOLE_MIDDLE), workload_rows=[_row("a", WORKLOAD_RECOVERS)])
        for name in ("one", "two")
    ]
    return adoption_reading(
        families, required_families=2, audit_invariant=True, required_middle_pairs=2, **kwargs
    )


def test_adoption_recommends_a_default_change_only_with_a_clean_audit():
    """The audit is an INPUT to the verdict: a cell-retiring move is an option, not a default."""
    verdict, _ = _clean()
    assert verdict == ADOPT_AS_DEFAULT
    families = [
        _stub_family(name, middle=dict(_WHOLE_MIDDLE), workload_rows=[_row("a", WORKLOAD_RECOVERS)])
        for name in ("one", "two")
    ]
    verdict, reason = adoption_reading(
        families, required_families=2, audit_invariant=False, required_middle_pairs=2
    )
    assert verdict == ADOPT_AS_OPTION and "not cleared" in reason


def test_adoption_refuses_an_unfinished_middle_stratum():
    """The recovery the strategy exists for is a gate, not a column."""
    incomplete = {**_WHOLE_MIDDLE, "entry_aware_wins": 1, "entry_aware_completed": 1}
    families = [
        _stub_family("one", middle=dict(_WHOLE_MIDDLE), workload_rows=[]),
        _stub_family("two", middle=incomplete, workload_rows=[]),
    ]
    verdict, reason = adoption_reading(
        families, required_families=2, audit_invariant=True, required_middle_pairs=2
    )
    assert verdict == ADOPT_REFUSE and "unfinished" in reason


def test_adoption_refuses_a_workload_whose_folded_arms_diverged():
    """Both arms folded and still offered different bytes: not separable from the trim."""
    row = {**_row("a", WORKLOAD_UNPAIRED), "n_unpaired_divergent_fold": 1}
    families = [
        _stub_family(name, middle=dict(_WHOLE_MIDDLE), workload_rows=[row])
        for name in ("one", "two")
    ]
    verdict, reason = adoption_reading(
        families, required_families=2, audit_invariant=True, required_middle_pairs=2
    )
    assert verdict == ADOPT_REFUSE and "different bytes" in reason


def test_an_under_powered_middle_stratum_withholds_the_default_not_the_option():
    """Every pair it DID read recovered, so power withholds a default without refusing the option."""
    thin = {**_WHOLE_MIDDLE, "n_pairs": 1, "entry_aware_wins": 1, "entry_aware_completed": 1}
    families = [
        _stub_family(
            "one", middle=dict(_WHOLE_MIDDLE), workload_rows=[_row("a", WORKLOAD_RECOVERS)]
        ),
        _stub_family("two", middle=thin, workload_rows=[_row("a", WORKLOAD_RECOVERS)]),
    ]
    verdict, reason = adoption_reading(
        families, required_families=2, audit_invariant=True, required_middle_pairs=2
    )
    assert verdict == ADOPT_AS_OPTION and "only 1 of 2 declared middle cases" in reason


def test_a_middle_stratum_with_no_usable_pair_is_unreadable():
    """Nothing entered the regime under test, so the run says nothing in either direction."""
    empty = {**_WHOLE_MIDDLE, "n_pairs": 0, "entry_aware_wins": 0, "entry_aware_completed": 0}
    families = [
        _stub_family("one", middle=dict(_WHOLE_MIDDLE), workload_rows=[]),
        _stub_family("two", middle=empty, workload_rows=[]),
    ]
    verdict, reason = adoption_reading(
        families, required_families=2, audit_invariant=True, required_middle_pairs=2
    )
    assert verdict == ADOPT_INCONCLUSIVE and "middle evidence stratum" in reason


def test_a_case_that_never_folded_is_excluded_rather_than_charged_to_the_trim():
    """No fold means no trim ran, so the arms executed the identical program on that case."""

    def case(item: str, *, folds: int, offered: int, success: bool) -> dict[str, object]:
        return {
            "item_id": item,
            "success": success,
            "status": "ok",
            "measured_folds": folds,
            "first_fold_input_chars": offered,
            "summary_input_elided_chars": 10,
            "model_input_prompt_chars": 100,
            "summary_prompt_chars": 50,
        }

    baseline = {
        "workload": "w",
        "task_set_digest": "d",
        "completion": 1.0,
        "cases": [
            case("a", folds=1, offered=10, success=True),
            case("b", folds=1, offered=10, success=True),
        ],
    }
    candidate = {
        "workload": "w",
        "task_set_digest": "d",
        "completion": 0.5,
        "cases": [
            case("a", folds=1, offered=10, success=True),
            case("b", folds=0, offered=0, success=False),
        ],
    }
    reading = workload_reading(baseline, candidate)
    assert reading["n_pairs"] == 1
    assert reading["n_unpaired_no_fold"] == 1 and reading["n_unpaired_divergent_fold"] == 0
    # Excluded from the delta, but the completion RATE still carries the lost case.
    assert reading["reading"] == WORKLOAD_UNCHANGED
    assert reading["completion_delta"] == -0.5


def test_extra_summary_bytes_downgrade_a_default_change_to_an_option():
    """Recovering completion by spending more of the window is a different recommendation."""
    families = [
        _stub_family(
            name,
            middle=dict(_WHOLE_MIDDLE),
            workload_rows=[_row("a", WORKLOAD_RECOVERS, summary_delta=120)],
        )
        for name in ("one", "two")
    ]
    verdict, reason = adoption_reading(
        families, required_families=2, audit_invariant=True, required_middle_pairs=2
    )
    assert verdict == ADOPT_AS_OPTION and "more summary prompt bytes" in reason


def test_an_unpaired_case_is_named_rather_than_averaged():
    """A case whose arms folded different bytes is dropped from the delta and counted."""

    def case(item: str, *, offered: int, success: bool) -> dict[str, object]:
        return {
            "item_id": item,
            "success": success,
            "status": "ok",
            "first_fold_input_chars": offered,
            "summary_input_elided_chars": 10,
            "model_input_prompt_chars": 100,
            "summary_prompt_chars": 50,
            "measured_folds": 1,
        }

    baseline = {
        "workload": "w",
        "task_set_digest": "d",
        "completion": 0.5,
        "cases": [case("a", offered=10, success=False), case("b", offered=10, success=True)],
    }
    candidate = {
        "workload": "w",
        "task_set_digest": "d",
        "completion": 1.0,
        "cases": [case("a", offered=10, success=True), case("b", offered=99, success=True)],
    }
    reading = workload_reading(baseline, candidate)
    assert reading["n_pairs"] == 1 and reading["n_unpaired"] == 1
    assert reading["n_unpaired_divergent_fold"] == 1
    assert reading["reading"] == WORKLOAD_UNPAIRED
