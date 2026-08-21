"""Per-form restatement rows, built directly from a synthetic restated surface.

`crossover_row` is the one place where a COMMITTED published number meets a freshly measured
geometry, and each published form meets it differently. These call it with a hand-built surface row
so a form's own rule -- the band a derived ratio is read against, the ladder a guard is placed back
on -- is exercised at its edges without a run to reach them. The end-to-end restatement over the
committed design lives in `test_agentic_memory_crossover_restatement.py`.
"""

from copy import deepcopy
import json
from pathlib import Path
from typing import cast

import pytest

from llb.bench.memory.boundary.crossover import (
    READING_BRACKETED,
    READING_COMPACT_ACROSS_GRID,
)
from llb.bench.memory.crossover_restatement.run import audit_published_cells
from llb.bench.memory.crossover_restatement.design import (
    audited_designs,
    load_restatement_design,
    published_crossovers,
)
from llb.bench.memory.crossover_restatement.forms import crossover_row
from llb.bench.memory.crossover_restatement.reading import (
    BASIS_RESTATED,
    BASIS_UNRESTATED_DERIVATION,
    FORM_INTERPOLATED,
    READING_DERIVED_UNRESTATED,
    READING_MOVED,
    operator_lines,
    restatement_reading,
)
from llb.bench.memory.crossover_restatement.report import persist_restatement
from llb.bench.policy_change.audit import KIND_COLLAPSE, KIND_SURFACE

ROOT = Path(__file__).resolve().parents[4]
DESIGN_PATH = ROOT / "samples/benchmarks/agentic_compact_crossover_restatement_design.json"

# The depth-10 cap peak the committed geometry measures, and the bracket the surface published.
DEPTH_10_CAP_PEAK = 11926
DEPTH_10_BRACKET = [20000, 23000]


def _restatement_inputs() -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """The audited designs and the model-free audit, both cheap enough to rebuild per test."""
    design = load_restatement_design(DESIGN_PATH)
    return audited_designs(design, root=ROOT), audit_published_cells(design, root=ROOT)


def _published(study_kind: str, depth: int) -> dict[str, object]:
    return next(
        row
        for row in published_crossovers(load_restatement_design(DESIGN_PATH))
        if row["study_kind"] == study_kind and row["depth"] == depth
    )


def _surface(guard: float, *, cap_peak: int = DEPTH_10_CAP_PEAK) -> list[dict[str, object]]:
    return [
        {
            "depth": 10,
            "reading": READING_BRACKETED,
            "crossover_max_prompt_chars": guard,
            "cap_peak_prompt_chars": cap_peak,
            "crossover_guard_ratio": guard / cap_peak,
            "bracket": DEPTH_10_BRACKET,
        }
    ]


def test_a_portable_ratio_that_leaves_its_published_band_withdraws_the_number():
    """The band is the criterion this form has, so a ratio outside it is a moved crossover."""
    designs, audit = _restatement_inputs()
    published = _published(KIND_COLLAPSE, 10)

    # A guard whose trigger lands under the band's lower edge even after rounding.
    low = crossover_row(published, designs, audit, _surface(19000.0))
    assert low["restated_value"] == pytest.approx(9500 / DEPTH_10_CAP_PEAK)
    assert low["invariance_holds"] is False
    reading, reason = restatement_reading(True, [low], 1)
    assert reading == READING_MOVED
    assert "portable ratio 0.797x outside the 0.85-0.92x band" in reason

    # The band's edges are the ROUNDED ratios of the tested depths, so a ratio a hair under the
    # lower edge is inside the band as published -- that is why the design states the precision.
    edge = crossover_row(published, designs, audit, _surface(20270.0))
    assert cast(float, edge["restated_value"]) < 0.85
    assert edge["invariance_holds"] is True


def test_a_ratio_with_an_unrestated_source_is_unresolved_and_fails_the_persisted_metric(
    tmp_path: Path,
):
    """The ratio's invariant cells cannot silently substitute for its declared source guard."""
    designs, audit = _restatement_inputs()
    surfaces = [{"depth": 10, "reading": READING_COMPACT_ACROSS_GRID}]
    row = crossover_row(_published(KIND_COLLAPSE, 10), designs, audit, surfaces)

    assert row["restated_value"] is None
    assert row["invariance_holds"] is None
    assert row["basis"] == BASIS_UNRESTATED_DERIVATION
    assert row["derived_from_study_kind"] == KIND_SURFACE
    assert row["derived_from_depth"] == 10
    assert row["derived_from_form"] == FORM_INTERPOLATED

    reading, reason = restatement_reading(True, [row], 0)
    assert reading == READING_DERIVED_UNRESTATED
    assert "compact_trigger_guard_collapse depth 10 portable_trigger_ratio" in reason
    assert "compact_memory_boundary_surface depth 10 interpolated_guard" in reason
    lines = operator_lines([row], [], reading, "window")
    assert any("portable trigger ratio was not restated" in line for line in lines)
    assert any("do not read its invariant contributing cells" in line for line in lines)

    analysis = {"crossovers": [row], "restatement_reading": reading}
    paths = persist_restatement(
        load_restatement_design(DESIGN_PATH),
        analysis,
        data_dir=tmp_path,
        table="synthetic unresolved derivation",
        tokens_per_s=0.0,
        mirror=lambda *_: None,
    )
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["metrics"]["objective_score"] == 0.0
    assert manifest["metrics"]["reliability"] == 0.0


def test_a_published_fold_step_the_moved_geometry_no_longer_has_names_the_published_row():
    """The restatement's step comes from a COMMITTED artifact, so the two can describe task worlds.

    Every other ladder caller reads a step it measured itself. This one interpolates a fresh guard
    against a fresh sequence and then asks where the PUBLISHED step's interval was, so a step the
    geometry no longer offers is not an argument error -- it is the drift the restatement exists to
    catch, and it used to surface as the interval arithmetic's bare "outside an N-step sequence".
    """
    designs, audit = _restatement_inputs()
    published = _published(KIND_SURFACE, 10)
    surfaces = _surface(21899.890064587056)

    # The control: on the geometry the number was published against, the interval is the one the
    # restated guard has to stay inside.
    row = crossover_row(published, designs, audit, surfaces)
    low, high = row["fold_step_guard_interval"]
    assert row["basis"] == BASIS_RESTATED and low <= row["restated_value"] < high

    # Move the task world so the depth-10 walk ends four steps before the published fold step 10.
    moved = deepcopy(designs)
    cast(dict[str, object], moved[KIND_SURFACE]["held_fixed"])["max_steps_margin"] = -4
    with pytest.raises(ValueError, match="crossover at depth 10 is stated at fold step 10"):
        crossover_row(published, moved, audit, surfaces)
