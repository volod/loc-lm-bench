"""The CI gate: a shipped context-policy constant cannot drift away from its published evidence."""

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from llb.bench.agentic.context_policy import ContextPolicy
from llb.bench.policy_change.audit import (
    AUDITABLE_FIELDS,
    KIND_SURFACE,
)
from llb.bench.policy_change.geometry import load_audited_design, load_audited_designs
from llb.bench.policy_change.pin_gate import (
    DESIGNS_AGREE,
    DESIGNS_RESTATED,
    PINS_PATH,
    PolicyPins,
    check_policy_pins,
    load_policy_pins,
    shipped_policy_values,
)
from llb.bench.policy_change.pin_gate_report import format_pin_gate_report

ROOT = Path(__file__).resolve().parents[4]
PINS = ROOT / PINS_PATH
SURFACE = ROOT / "samples/benchmarks/agentic_compact_memory_boundary_surface_design.json"


def _pins() -> PolicyPins:
    return load_policy_pins(PINS)


def _only(*fields: str) -> PolicyPins:
    """The fixture narrowed to the named pins, so a synthetic-drift test audits only those."""
    pins = _pins()
    return replace(pins, pins={field: pins.pins[field] for field in fields})


def _surface() -> dict[str, dict[str, object]]:
    """One study, so a synthetic-drift test pays for six cells instead of twenty-two."""
    return {KIND_SURFACE: load_audited_design(SURFACE)}


def _drifted(field: str, value: Any) -> dict[str, Any]:
    return {**shipped_policy_values(), field: value}


def _write_pins(path: Path, **overrides: object) -> Path:
    raw = json.loads(PINS.read_text(encoding="utf-8"))
    for field, entry in overrides.items():
        if entry is None:
            del raw["pins"][field]
        else:
            raw["pins"][field] = entry
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


# --- the gate itself --------------------------------------------------------------------------


def test_every_shipped_constant_still_matches_the_value_its_evidence_was_measured_under():
    """THE gate. A red build here means a constant moved; the message says what it retired."""
    check = check_policy_pins(_pins(), load_audited_designs())
    if not check.ok:  # `fail` prints the re-run scope; `assert` would bury it under a PinCheck repr
        pytest.fail(format_pin_gate_report(check))
    assert "all 6 shipped context-policy constants match" in format_pin_gate_report(check)


def test_a_new_shipped_constant_cannot_be_added_without_a_pin():
    """The hole the gate closes is an unguarded constant, so the field sets must stay in lockstep."""
    constants = set(ContextPolicy.__dataclass_fields__) - {"name"}
    assert set(AUDITABLE_FIELDS) == constants == set(_pins().pins)


def test_the_pinned_values_are_the_shipped_ones_read_off_the_dataclass():
    shipped = shipped_policy_values()
    assert shipped == {field: pin.value for field, pin in _pins().pins.items()}
    assert shipped["observation_cap_chars"] == ContextPolicy().observation_cap_chars


def test_the_docs_the_pins_name_still_carry_those_sections():
    """A drift message points at where the numbers are published, so the anchors must resolve."""
    for anchor in _pins().published_in:
        path, _, slug = anchor.partition("#")
        headings = re.findall(r"^#+\s+(.+)$", (ROOT / path).read_text(encoding="utf-8"), re.M)
        assert slug in {_slug(heading) for heading in headings}, anchor


def _slug(heading: str) -> str:
    """GitHub's heading slug: lowercased, punctuation dropped, spaces hyphenated."""
    return re.sub(r"[^a-z0-9\s-]", "", heading.strip().lower()).replace(" ", "-")


# --- what a drift reports ---------------------------------------------------------------------


def test_a_drift_that_retires_published_cells_names_every_one_of_them():
    """The failure message is the re-run scope: which cells, which arms, where the change bites."""
    check = check_policy_pins(
        _only("observation_cap_chars"),
        _surface(),
        shipped=_drifted("observation_cap_chars", 1600),
    )

    assert not check.ok and check.drift is not None and len(check.drift.moves) == 1
    drift = check.drift
    move = drift.moves[0]
    assert (move.field, move.pinned, move.shipped) == ("observation_cap_chars", 800, 1600)
    assert drift.n_invalidated == drift.summary["n_cells"] > 0
    report = format_pin_gate_report(check)
    assert "observation_cap_chars: pinned 800 -> shipped 1600" in report
    assert "surface-d6-g12000: depth 6, guard 12000" in report
    assert "first divergent model call 2" in report
    assert "re-measure those cells" in report
    assert "extended-workflows/crossover-geometry.md#cap-fitting-boundary-surface" in report


def test_a_drift_that_retires_cells_lists_the_published_figures_those_cells_stand_under():
    """The operator gets the exact figures to restate, not only cell ids to re-measure."""
    check = check_policy_pins(
        _only("observation_cap_chars"),
        load_audited_designs(),
        shipped=_drifted("observation_cap_chars", 1600),
    )

    drift = check.drift
    assert drift is not None and drift.retired_figures
    forms = {(figure.study, figure.depth, figure.form) for figure in drift.retired_figures}
    assert ("compact_memory_boundary_surface", 10, "interpolated_guard") in forms
    assert ("compact_fold_step_crossover", 10, "fold_step_boundary") in forms
    assert ("compact_trigger_guard_collapse", 10, "portable_trigger_ratio") in forms
    report = format_pin_gate_report(check)
    assert "those cells retire these published figures:" in report
    assert "published value 21899.890064587056" in report
    assert "restate those figures in the docs" in report


def test_a_drift_the_audit_clears_on_cap_fitting_cells_still_fails_but_says_restating_is_free():
    """On the surface alone, `keep_last_n` still clears -- that study never runs the keep policy."""
    check = check_policy_pins(_only("keep_last_n"), _surface(), shipped=_drifted("keep_last_n", 1))

    assert not check.ok and check.drift is not None and check.drift.n_invalidated == 0
    assert check.drift.affected_published_values == ()
    report = format_pin_gate_report(check)
    assert "no published cell is invalidated" in report and "restating the pin is free" in report
    assert "re-measure those cells" not in report


def test_a_keep_last_n_drift_over_the_full_registry_names_the_keep_lanes():
    """Widening the audit registry widens the gate: keep is no longer a free restatement."""
    from llb.bench.policy_change.audit import KIND_CONSTANT_SWEEP, KIND_KEEP_LONG

    check = check_policy_pins(
        _only("keep_last_n"), load_audited_designs(), shipped=_drifted("keep_last_n", 1)
    )

    drift = check.drift
    assert drift is not None and drift.n_invalidated == 2
    assert set(cast(list[str], drift.summary["studies_invalidated"])) == {
        KIND_CONSTANT_SWEEP,
        KIND_KEEP_LONG,
    }
    report = format_pin_gate_report(check)
    assert "sweep-keep-shipped" in report and "keep-long-shipped" in report
    assert "re-measure those cells" in report
    assert "restating the pin is free" not in report


def test_a_cell_that_pins_the_drifted_field_itself_is_counted_out_of_the_scope():
    """The collapse study sweeps `compact_share`, so a share drift does not describe its cells."""
    check = check_policy_pins(
        _only("compact_share"), load_audited_designs(), shipped=_drifted("compact_share", 0.45)
    )

    drift = check.drift
    assert drift is not None
    assert drift.summary["n_not_applicable"] == 8 and drift.n_invalidated == 12
    report = format_pin_gate_report(check)
    assert "12 of 19 applicable published cells are invalidated" in report
    assert "8 further cell(s) pin compact_share as their own study axis" in report


def test_a_drift_names_published_bands_whose_registered_arithmetic_reads_the_field():
    """The cell replay and the registered-operation scope are two consumers of one pin move."""
    check = check_policy_pins(
        _only("compact_share"), load_audited_designs(), shipped=_drifted("compact_share", 0.45)
    )

    drift = check.drift
    assert drift is not None
    assert [value.depth for value in drift.affected_published_values] == [6, 10]
    assert {value.fields for value in drift.affected_published_values} == {("compact_share",)}
    assert {value.statement for value in drift.affected_published_values} == {
        "published band [0.85, 0.92]"
    }
    report = format_pin_gate_report(check)
    assert "registered arithmetic also moves these published values" in report
    assert "compact_trigger_guard_collapse depth 6 portable_trigger_ratio" in report
    assert "compact_trigger_guard_collapse depth 10 portable_trigger_ratio" in report
    assert "re-derive and restate those values" in report


def test_an_arithmetic_scope_means_a_cell_free_drift_is_not_free_to_repin():
    """No replayed cell moving does not clear a separately derived published value."""
    check = check_policy_pins(_only("compact_share"), {}, shipped=_drifted("compact_share", 0.45))

    drift = check.drift
    assert drift is not None and drift.n_invalidated == 0
    assert len(drift.affected_published_values) == 2
    report = format_pin_gate_report(check)
    assert "the registered-arithmetic scope follows" in report
    assert "restating the pin is free" not in report


def test_two_constants_that_drift_together_are_audited_as_one_change():
    """A commit that re-pins two constants moved BOTH, so one scope is computed between the two
    policies that really existed -- not two scopes against configurations that never shipped."""
    shipped = {**_drifted("observation_cap_chars", 1600), "compact_keep_recent": 2}
    check = check_policy_pins(
        _only("observation_cap_chars", "compact_keep_recent"), _surface(), shipped=shipped
    )

    drift = check.drift
    assert drift is not None and drift.is_compound and len(drift.moves) == 2
    assert drift.summary["baseline"] == {"observation_cap_chars": 800, "compact_keep_recent": 1}
    assert drift.summary["candidate"] == {"observation_cap_chars": 1600, "compact_keep_recent": 2}
    assert drift.n_invalidated == drift.summary["n_cells"] > 0

    report = format_pin_gate_report(check)
    assert "2 of 2 shipped context-policy constants no longer match" in report
    assert "2 constants moved together and are audited as ONE change" in report
    assert "observation_cap_chars: pinned 800 -> shipped 1600" in report
    assert "compact_keep_recent: pinned 1 -> shipped 2" in report
    # One re-run scope, and one "pinned because" per constant that moved.
    assert report.count("re-measure those cells") == 1
    assert "pinned because (observation_cap_chars):" in report
    assert "pinned because (compact_keep_recent):" in report


def test_a_compound_drift_a_cell_partly_owns_is_audited_on_the_rest_of_the_change():
    """The collapse study owns `compact_share`; the cap half of the change still describes it."""
    shipped = {**_drifted("compact_share", 0.45), "observation_cap_chars": 1600}
    check = check_policy_pins(
        _only("compact_share", "observation_cap_chars"), load_audited_designs(), shipped=shipped
    )

    drift = check.drift
    assert drift is not None
    assert drift.summary["n_not_applicable"] == 0
    assert drift.summary["n_partially_applicable"] == 8 and drift.n_invalidated == 23
    report = format_pin_gate_report(check)
    assert "8 cell(s) declare part of this change as their own study axis" in report


def test_a_pin_matching_its_shipped_value_costs_no_replay(monkeypatch: pytest.MonkeyPatch):
    """A clean build replays and walks nothing, which is what makes the gate free per CI run."""
    monkeypatch.setattr(
        "llb.bench.policy_change.pin_gate.audit_policy_change",
        lambda *args, **kwargs: pytest.fail("an undrifted pin must not replay any cell"),
    )
    monkeypatch.setattr(
        "llb.bench.policy_change.pin_gate.policy_affected_published_values",
        lambda *args, **kwargs: pytest.fail("an undrifted pin must not walk published arithmetic"),
    )
    check = check_policy_pins(_pins(), load_audited_designs())
    assert check.ok and check.drift is None


def test_a_drift_feeds_the_full_pinned_policy_into_the_replay(monkeypatch: pytest.MonkeyPatch):
    """Untouched fields come from the pins, so a restated held field cannot leave a stale baseline."""
    seen: dict[str, Any] = {}

    def capture(designs, change, *, pinned=None):
        seen["pinned"] = pinned
        from llb.bench.policy_change.audit import audit_policy_change as real

        return real(designs, change, pinned=pinned)

    monkeypatch.setattr("llb.bench.policy_change.pin_gate.audit_policy_change", capture)
    check = check_policy_pins(_pins(), _surface(), shipped=_drifted("compact_keep_recent", 2))
    assert check.drift is not None
    assert seen["pinned"] == {field: pin.value for field, pin in _pins().pins.items()}


# --- the pin's own claim about the committed designs -------------------------------------------


def test_the_pin_that_supersedes_a_design_must_say_so():
    """`summary_input_cap` ships `window`; the designs record the retired `trigger` bound."""
    pins = _pins()
    assert pins.pins["summary_input_cap"].designs == DESIGNS_RESTATED
    assert pins.pins["observation_cap_chars"].designs == DESIGNS_AGREE
    assert pins.pins["keep_last_n"].designs == DESIGNS_AGREE
    assert check_policy_pins(pins, load_audited_designs()).stale_claims == ()


def test_a_pin_that_claims_agreement_it_does_not_have_is_caught(tmp_path: Path):
    entry = {"value": "window", "designs": DESIGNS_AGREE, "note": "claims the designs agree"}
    pins = load_policy_pins(_write_pins(tmp_path / "pins.json", summary_input_cap=entry))
    check = check_policy_pins(pins, load_audited_designs())

    assert not check.ok and len(check.stale_claims) == 1
    claim = check.stale_claims[0]
    assert (claim.field, claim.declared, claim.supported) == (
        "summary_input_cap",
        DESIGNS_AGREE,
        DESIGNS_RESTATED,
    )
    report = format_pin_gate_report(check)
    assert "the committed studies support 'restated'" in report
    assert f"{KIND_SURFACE}='trigger'" in report


# --- the fixture the gate reads ----------------------------------------------------------------


def test_a_fixture_the_gate_could_not_act_on_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly one pin per auditable field"):
        load_policy_pins(_write_pins(tmp_path / "missing.json", keep_last_n=None))
    with pytest.raises(ValueError, match="must state a int value"):
        load_policy_pins(
            _write_pins(
                tmp_path / "typed.json",
                keep_last_n={"value": "3", "designs": "unstated", "note": "a string"},
            )
        )
    with pytest.raises(ValueError, match="must declare designs from"):
        load_policy_pins(
            _write_pins(
                tmp_path / "state.json",
                keep_last_n={"value": 3, "designs": "maybe", "note": "an unknown state"},
            )
        )
    with pytest.raises(ValueError, match="must carry a note"):
        load_policy_pins(
            _write_pins(
                tmp_path / "note.json",
                keep_last_n={"value": 3, "designs": "unstated", "note": "  "},
            )
        )
    with pytest.raises(ValueError, match="schema_version must be"):
        (tmp_path / "schema.json").write_text('{"pins": {}}', encoding="utf-8")
        load_policy_pins(tmp_path / "schema.json")
    with pytest.raises(ValueError, match="must name the doc sections"):
        raw = json.loads(PINS.read_text(encoding="utf-8")) | {"published_in": ["no-anchor.md"]}
        (tmp_path / "anchors.json").write_text(json.dumps(raw), encoding="utf-8")
        load_policy_pins(tmp_path / "anchors.json")
    with pytest.raises(ValueError, match="cannot read context-policy pins"):
        load_policy_pins(tmp_path / "absent.json")
