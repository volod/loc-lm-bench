"""Published-crossover restatement: the model-free bound audit and the restatement end to end.

Each published form's own row rule is exercised at its edges in
`test_agentic_memory_crossover_restatement_forms.py`; what runs here is the whole study over the
committed design -- the audit, the design contract, the substituted surface, the cap peaks, and the
reading the six published crossovers produce together.
"""

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import cast

import pytest

from llb.bench.agentic_memory_boundary_probe import oracle_controller
from llb.bench.agentic_memory_cap_audit import (
    VERDICT_INVARIANT,
    VERDICT_SENSITIVE,
    audit_design,
    audit_summary,
    sensitive_cell_ids,
)
from llb.bench.agentic_policy_change_audit import (
    AUDITED_KINDS,
    KIND_COLLAPSE,
    KIND_FOLD_STEP,
    KIND_SURFACE,
    declared_geometry,
    load_audited_design,
)
from llb.bench.agentic_memory_crossover_restatement import (
    analyze_restatement,
    audit_published_cells,
    run_sensitive_surface_cells,
)
from llb.bench.agentic_memory_crossover_restatement_design import (
    load_restatement_design,
    published_crossovers,
    validate_restatement_design,
)
from llb.bench.agentic_memory_crossover_restatement_reading import (
    BASIS_ALREADY_MEASURED,
    BASIS_DERIVED,
    BASIS_INVARIANT,
    BASIS_RESTATED,
    CRITERION_BAND,
    CRITERION_FOLD_STEP,
    FORM_PORTABLE_RATIO,
    PEAK_INVARIANT,
    PEAK_MOVED,
    PEAK_UNPUBLISHED,
    READING_INELIGIBLE,
    READING_MOVED,
    READING_UNCHANGED,
    STUDY_KIND,
)
from llb.bench.agentic_memory_crossover_restatement_report import (
    format_restatement_table,
    persist_restatement,
)

ROOT = Path(__file__).resolve().parents[3]
DESIGN_PATH = ROOT / "samples/benchmarks/agentic_compact_crossover_restatement_design.json"
SURFACE_DESIGN = ROOT / "samples/benchmarks/agentic_compact_memory_boundary_surface_design.json"

CONTROL_PASS = {
    "eligible": True,
    "model": "pinned",
    "completion": 1.0,
    "minimum_completion_rate": 0.75,
}
_MEMORY = re.compile(r"final_code=([A-Z0-9-]+)")

# The published surface, reduced to what the restatement reads: the measured cells and the peaks.
PUBLISHED_DELTAS = {
    "surface-d6-g12000": (6, 12000, -792.0, "compact_cheaper"),
    "surface-d6-g14000": (6, 14000, -125.85714285714286, "compact_cheaper"),
    "surface-d6-g15500": (6, 15500, 1054.5714285714287, "cap_cheaper"),
    "surface-d10-g14000": (10, 14000, -4458.285714285715, "compact_cheaper"),
    "surface-d10-g20000": (10, 20000, -2633.4285714285716, "compact_cheaper"),
    "surface-d10-g23000": (10, 23000, 1524.857142857143, "cap_cheaper"),
}


def _fake_model(prompt: str) -> str:
    """Perfect play, plus a fixed summary whenever the compact policy folds the transcript."""
    if "Стисло підсумуй" in prompt:
        return "[memory: final_code=" + (_MEMORY.findall(prompt) or [""])[0] + "] крок виконано"
    return oracle_controller(prompt)


def _sensitive_rows(
    design: dict[str, object], audit: dict[str, object], tmp_path: Path
) -> list[dict[str, object]]:
    """Re-measure exactly the cells the audit calls bound-sensitive, with no GPU."""
    return run_sensitive_surface_cells(
        design,
        audit,
        root=ROOT,
        model="fake",
        backend="fake",
        complete=_fake_model,
        data_dir=tmp_path,
    )


def _published_surface() -> dict[str, object]:
    return {
        "cap_peak_prompt_chars": {"6": 8374, "10": 11926},
        "cells": [
            {
                "cell_id": cell_id,
                "depth": depth,
                "max_prompt_chars": guard,
                "valid": True,
                "measured_side": side,
                "cost_evidence": {
                    "compact_minus_cap_total_model_input_tokens": {
                        "mean": delta,
                        "lo": delta - 20,
                        "hi": delta + 20,
                    }
                },
            }
            for cell_id, (depth, guard, delta, side) in PUBLISHED_DELTAS.items()
        ],
    }


# --- the model-free audit -------------------------------------------------------------------


def test_the_audit_reads_cell_geometry_out_of_every_committed_study_shape():
    """A flat grid, families, and per-depth ladders all reduce to the same geometry rows."""
    paths = {
        KIND_SURFACE: SURFACE_DESIGN,
        KIND_COLLAPSE: ROOT
        / "samples/benchmarks/agentic_compact_trigger_guard_collapse_design.json",
        KIND_FOLD_STEP: ROOT / "samples/benchmarks/agentic_compact_fold_step_crossover_design.json",
    }
    assert set(paths) == set(AUDITED_KINDS)
    for kind, path in paths.items():
        cells = declared_geometry(load_audited_design(path), kind)
        assert cells and all(
            cell["depth"] >= 3 and cell["max_prompt_chars"] > 0 and 0.0 < cell["compact_share"] <= 1
            for cell in cells
        ), kind
        assert len({cell["cell_id"] for cell in cells}) == len(cells), kind


def test_an_unknown_study_kind_has_no_known_cell_geometry():
    with pytest.raises(ValueError, match="not a readable geometry kind"):
        declared_geometry(load_audited_design(SURFACE_DESIGN), "compact_something_else")


def test_a_cell_that_elides_nothing_under_either_bound_needs_no_run():
    """The audit's whole point: bit-identical summarize input means a published cost that stands."""
    rows = audit_design(load_audited_design(SURFACE_DESIGN), study_kind=KIND_SURFACE)
    invariant = [row for row in rows if row["verdict"] == VERDICT_INVARIANT]
    sensitive = [row for row in rows if row["verdict"] == VERDICT_SENSITIVE]
    assert invariant and all(row["trigger_elided_chars"] == 0 for row in invariant)
    # The bound only ever trims; the shipped bound never elides where the retired one did not.
    assert all(row["window_elided_chars"] == 0 for row in rows)
    assert all(row["trigger_elided_chars"] > 0 for row in sensitive)
    assert sensitive_cell_ids(rows) == [row["cell_id"] for row in sensitive]


def test_the_depth_6_evidence_is_bound_invariant_end_to_end():
    """Depth 6 folds a transcript neither bound trims, so no depth-6 number can have moved."""
    audit = audit_published_cells(load_restatement_design(DESIGN_PATH), root=ROOT)
    depth_6 = [row for rows in audit["per_study"].values() for row in rows if row["depth"] == 6]
    assert depth_6 and all(row["verdict"] == VERDICT_INVARIANT for row in depth_6)
    summary = audit["summary"]
    assert summary["n_bound_invariant"] + summary["n_bound_sensitive"] == summary["n_cells"]
    assert all(cell["depth"] == 10 for cell in summary["sensitive"])


def test_the_audit_summary_names_every_sensitive_cell_with_its_geometry():
    rows = audit_design(load_audited_design(SURFACE_DESIGN), study_kind=KIND_SURFACE)
    summary = audit_summary({KIND_SURFACE: rows})
    for cell in summary["sensitive"]:
        assert cell["study_kind"] == KIND_SURFACE
        assert {"depth", "compact_share", "max_prompt_chars", "trigger_elided_chars"} <= set(cell)


# --- the design contract --------------------------------------------------------------------


def test_the_committed_design_names_studies_that_exist_and_publish_what_it_claims():
    design = load_restatement_design(DESIGN_PATH)
    validate_restatement_design(design, root=ROOT)
    crossovers = published_crossovers(design)
    assert len(crossovers) == 6  # two depths for each of the three audited studies
    assert all(row["fold_step"] >= 1 for row in crossovers)


def test_the_design_refuses_a_retired_bound_a_missing_study_and_an_untested_depth():
    design = load_restatement_design(DESIGN_PATH)

    retired = deepcopy(design)
    retired["shipped_summary_input_cap"] = "trigger"
    with pytest.raises(ValueError, match="the bound the runtime ships"):
        validate_restatement_design(retired, root=ROOT)

    missing = deepcopy(design)
    missing["audited_studies"][0]["design_path"] = "samples/benchmarks/does_not_exist.json"
    with pytest.raises(ValueError, match="does not exist"):
        validate_restatement_design(missing, root=ROOT)

    mismatched = deepcopy(design)
    mismatched["audited_studies"][0]["design_path"] = (
        "samples/benchmarks/agentic_compact_fold_step_crossover_design.json"
    )
    with pytest.raises(ValueError, match="declares study kind"):
        validate_restatement_design(mismatched, root=ROOT)

    untested = deepcopy(design)
    untested["audited_studies"][0]["published_crossovers"][0]["depth"] = 8
    with pytest.raises(ValueError, match="does not test"):
        validate_restatement_design(untested, root=ROOT)

    unstepped = deepcopy(design)
    unstepped["audited_studies"][0]["published_crossovers"][0]["fold_step"] = 0
    with pytest.raises(ValueError, match="must name the fold step"):
        validate_restatement_design(unstepped, root=ROOT)


def test_a_portable_ratio_crossover_must_publish_a_band_and_the_precision_it_is_quoted_to():
    design = load_restatement_design(DESIGN_PATH)
    collapse = next(
        study
        for study in cast(list[dict[str, object]], design["audited_studies"])
        if study["study_kind"] == KIND_COLLAPSE
    )
    assert all(
        crossover.get("value") is None
        for crossover in cast(list[dict[str, object]], collapse["published_crossovers"])
    )

    for mutation, match in (
        ({"published_band": [0.92, 0.85]}, "ascending pair"),
        ({"published_band": 0.85}, "ascending pair"),
        ({"published_band": [0.0, 0.92]}, "ascending pair"),
        ({"band_decimals": 0}, "band_decimals"),
        ({"band_decimals": None}, "band_decimals"),
    ):
        broken = deepcopy(design)
        for study in cast(list[dict[str, object]], broken["audited_studies"]):
            if study["study_kind"] == KIND_COLLAPSE:
                cast(list[dict[str, object]], study["published_crossovers"])[0].update(mutation)
        with pytest.raises(ValueError, match=match):
            validate_restatement_design(broken, root=ROOT)


# --- the restatement ------------------------------------------------------------------------


def test_only_the_bound_sensitive_surface_cells_are_re_run(tmp_path: Path):
    design = load_restatement_design(DESIGN_PATH)
    audit = audit_published_cells(design, root=ROOT)
    rows = _sensitive_rows(design, audit, tmp_path)
    sensitive = sensitive_cell_ids(audit["per_study"][KIND_SURFACE])
    assert [row["restated_cell_id"] for row in rows] == sensitive
    # The published grid has six cells; re-running all of them is exactly what the audit avoids.
    assert 0 < len(rows) < 6


def test_the_restatement_re_interpolates_and_checks_the_fold_step_it_names(tmp_path: Path):
    design = load_restatement_design(DESIGN_PATH)
    audit = audit_published_cells(design, root=ROOT)
    rows = _sensitive_rows(design, audit, tmp_path)
    analysis = analyze_restatement(
        design, audit, CONTROL_PASS, _published_surface(), rows, root=ROOT
    )
    crossovers = {(row["study_kind"], row["depth"]): row for row in analysis["crossovers"]}

    # Depth 6 needed no run anywhere: every contributing cell is bit-identical under both bounds.
    # (The collapse's derived ratio is the exception, and has its own test below: its value comes
    # from the surface's guard rather than from its own cells.)
    for kind in (KIND_SURFACE, KIND_FOLD_STEP):
        assert crossovers[(kind, 6)]["basis"] == BASIS_INVARIANT
        assert crossovers[(kind, 6)]["restated_value"] is None
        assert crossovers[(kind, 6)]["invariance_holds"] is True

    # The depth-10 surface crossover is the one that is actually re-interpolated.
    surface = crossovers[(KIND_SURFACE, 10)]
    assert surface["basis"] == BASIS_RESTATED
    assert surface["restated_value"] is not None
    assert surface["restated_bracket"] == [20000, 23000]
    # It must still land inside the fold step it was published in -- that is the invariance rule.
    assert surface["invariance_criterion"] == CRITERION_FOLD_STEP
    assert surface["invariance_holds"] is True
    low, high = surface["fold_step_guard_interval"]
    assert low <= surface["restated_value"] < high

    # The fold-step boundary at depth 10 is not re-derived here; the cap study already measured it.
    assert crossovers[(KIND_FOLD_STEP, 10)]["basis"] == BASIS_ALREADY_MEASURED
    assert analysis["restatement_reading"] == READING_UNCHANGED
    assert analysis["changes_shipped_default"] is False
    assert any(
        "stated for `summary_input_cap=window`" in line for line in analysis["operator_lines"]
    )

    table = format_restatement_table(analysis)
    paths = persist_restatement(
        design, analysis, data_dir=tmp_path, table=table, tokens_per_s=0.0, mirror=lambda *_: None
    )
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["config"]["study_kind"] == STUDY_KIND
    assert "model-free bound audit" in table and "published crossovers" in table


def test_a_crossover_that_leaves_its_fold_step_is_reported_as_moved(tmp_path: Path):
    """The invariance rule is the fold step, so only a step change withdraws a published number."""
    design = load_restatement_design(DESIGN_PATH)
    audit = audit_published_cells(design, root=ROOT)
    rows = _sensitive_rows(design, audit, tmp_path)
    # Drive the re-measured cell's cost far enough that the interpolated crossing leaves step 10.
    moved = deepcopy(rows)
    for row in moved:
        row["compact_mean_total_model_input_tokens"] = (
            row["cap_mean_total_model_input_tokens"] + 1e5
        )
        row["paired"]["total_model_input_tokens"]["delta"] = {"mean": 1e5, "lo": 1e5, "hi": 1e5}
    analysis = analyze_restatement(
        design, audit, CONTROL_PASS, _published_surface(), moved, root=ROOT
    )
    surface = next(
        row
        for row in analysis["crossovers"]
        if row["study_kind"] == KIND_SURFACE and row["depth"] == 10
    )
    assert surface["invariance_holds"] is False
    assert analysis["restatement_reading"] == READING_MOVED
    assert any("re-derive the routing rule" in line for line in analysis["operator_lines"])


# --- the derived portable ratio ---------------------------------------------------------------


def test_the_portable_ratio_is_restated_from_the_guard_it_is_derived_from(tmp_path: Path):
    """The collapse's own cells cannot restate it -- the surface's guard is what moves under it."""
    design = load_restatement_design(DESIGN_PATH)
    audit = audit_published_cells(design, root=ROOT)
    rows = _sensitive_rows(design, audit, tmp_path)
    analysis = analyze_restatement(
        design, audit, CONTROL_PASS, _published_surface(), rows, root=ROOT
    )
    surfaces = {cast(int, row["depth"]): row for row in analysis["restated_depth_surface"]}
    portable = {
        cast(int, row["depth"]): row
        for row in analysis["crossovers"]
        if row["form"] == FORM_PORTABLE_RATIO
    }
    assert set(portable) == {6, 10}
    for depth, row in portable.items():
        # Every collapse cell is bound-invariant, and the ratio is restated all the same.
        assert row["n_bound_sensitive_cells"] == 0
        assert row["basis"] == BASIS_DERIVED
        assert row["invariance_criterion"] == CRITERION_BAND
        # Trigger over cap peak, both read off the SAME restated surface row.
        guard = cast(float, surfaces[depth]["crossover_max_prompt_chars"])
        peak = cast(int, surfaces[depth]["cap_peak_prompt_chars"])
        assert row["derived_from_guard_chars"] == guard
        assert row["restated_cap_peak_prompt_chars"] == peak
        assert row["restated_trigger_chars"] == int(guard) // 2  # compact_share = 0.5
        assert row["restated_value"] == pytest.approx(row["restated_trigger_chars"] / peak)
        assert row["invariance_holds"] is True

    # The re-measured depth-10 cell removes a discount that flattered compact, so the guard and the
    # ratio it is derived from both come DOWN, and the published band still covers it at 2 decimals.
    assert portable[10]["restated_value"] < 10950 / 11926
    assert round(cast(float, portable[10]["restated_value"]), 2) == 0.92
    assert round(cast(float, portable[6]["restated_value"]), 2) == 0.85
    assert analysis["restatement_reading"] == READING_UNCHANGED
    assert any(
        "the portable trigger ratio is 0.917x" in line
        and "inside the published 0.85-0.92x band" in line
        for line in analysis["operator_lines"]
    )
    assert "0.85-0.92x" in format_restatement_table(analysis)


# --- the peak the restated ratio is stated against --------------------------------------------


def test_the_restated_ratio_is_stated_against_the_peak_of_the_geometry_that_measured_it(
    tmp_path: Path,
):
    """The guard and its peak come from ONE prompt sequence, and the published peak is a row."""
    design = load_restatement_design(DESIGN_PATH)
    audit = audit_published_cells(design, root=ROOT)
    rows = _sensitive_rows(design, audit, tmp_path)
    analysis = analyze_restatement(
        design, audit, CONTROL_PASS, _published_surface(), rows, root=ROOT
    )
    surfaces = {cast(int, row["depth"]): row for row in analysis["restated_depth_surface"]}
    peaks = {cast(int, row["depth"]): row for row in analysis["restated_cap_peaks"]}
    assert set(peaks) == set(surfaces) == {6, 10}
    for depth, peak in peaks.items():
        # The committed design's geometry still measures the peaks the surface published, so the
        # published table is unmoved -- but the ratio now rests on the measured one either way.
        assert peak["reading"] == PEAK_INVARIANT
        assert peak["cap_peak_delta_chars"] == 0
        assert (
            peak["measured_cap_peak_prompt_chars"]
            == peak["published_cap_peak_prompt_chars"]
            == surfaces[depth]["cap_peak_prompt_chars"]
        )
        assert surfaces[depth]["crossover_guard_ratio"] == pytest.approx(
            cast(float, surfaces[depth]["crossover_max_prompt_chars"])
            / cast(int, peak["measured_cap_peak_prompt_chars"])
        )


def test_a_moved_task_world_is_named_as_a_moved_peak_not_a_rescaled_ratio(tmp_path: Path):
    """A peak the geometry retired must not silently rescale a freshly interpolated guard.

    The fold-step check cannot see this: the drift here leaves the step ladder alone, so every
    published crossover still names its own fold step and the restatement reads UNCHANGED. That is
    exactly why the peak needs a row of its own -- dividing the new guard by the old peak would have
    published a rescaled ratio with nothing in the run saying the two came from different worlds.
    """
    design = load_restatement_design(DESIGN_PATH)
    audit = audit_published_cells(design, root=ROOT)
    rows = _sensitive_rows(design, audit, tmp_path)
    control = analyze_restatement(
        design, audit, CONTROL_PASS, _published_surface(), rows, root=ROOT
    )

    retired = _published_surface()
    cast(dict[str, object], retired["cap_peak_prompt_chars"])["10"] = 9000
    analysis = analyze_restatement(design, audit, CONTROL_PASS, retired, rows, root=ROOT)

    moved = next(row for row in analysis["restated_cap_peaks"] if row["depth"] == 10)
    assert moved["reading"] == PEAK_MOVED
    assert moved["published_cap_peak_prompt_chars"] == 9000
    assert moved["measured_cap_peak_prompt_chars"] == 11926
    assert moved["cap_peak_delta_chars"] == 2926
    # The ratio is the control's: it is divided by the measured peak, never by the published one.
    surfaces = {cast(int, row["depth"]): row for row in analysis["restated_depth_surface"]}
    controls = {cast(int, row["depth"]): row for row in control["restated_depth_surface"]}
    assert surfaces[10]["crossover_guard_ratio"] == controls[10]["crossover_guard_ratio"]
    assert next(row for row in analysis["restated_cap_peaks"] if row["depth"] == 6)["reading"] == (
        PEAK_INVARIANT
    )

    assert analysis["restatement_reading"] == READING_UNCHANGED
    assert any(
        "the cap peak moved 9000 -> 11926 chars (+2926)" in line
        for line in analysis["operator_lines"]
    )
    assert "cap peak" in format_restatement_table(analysis)


def test_a_depth_the_published_surface_states_no_peak_for_is_named_rather_than_missing(
    tmp_path: Path,
):
    """A published aggregate without the depth's peak is a gap to report, not a lookup failure."""
    design = load_restatement_design(DESIGN_PATH)
    audit = audit_published_cells(design, root=ROOT)
    rows = _sensitive_rows(design, audit, tmp_path)
    unpublished = _published_surface()
    del cast(dict[str, object], unpublished["cap_peak_prompt_chars"])["10"]
    analysis = analyze_restatement(design, audit, CONTROL_PASS, unpublished, rows, root=ROOT)

    row = next(row for row in analysis["restated_cap_peaks"] if row["depth"] == 10)
    assert row["reading"] == PEAK_UNPUBLISHED
    assert row["published_cap_peak_prompt_chars"] is None
    assert row["cap_peak_delta_chars"] is None
    assert row["measured_cap_peak_prompt_chars"] == 11926
    assert any("states no cap peak here" in line for line in analysis["operator_lines"])


def test_an_ineligible_family_supports_no_restatement_line():
    design = load_restatement_design(DESIGN_PATH)
    audit = audit_published_cells(design, root=ROOT)
    analysis = analyze_restatement(
        design, audit, {**CONTROL_PASS, "eligible": False}, _published_surface(), [], root=ROOT
    )
    assert analysis["restatement_reading"] == READING_INELIGIBLE
    assert analysis["restated_cells"] == [] and analysis["restated_depth_surface"] == []
    assert analysis["restated_cap_peaks"] == []
    assert analysis["operator_lines"] == [
        f"[{READING_INELIGIBLE}] no restatement line is supported"
    ]
