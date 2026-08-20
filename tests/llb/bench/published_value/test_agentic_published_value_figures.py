"""Published figures keyed by study and cell id -- what the pin gate prints when cells move."""

from pathlib import Path

import pytest

from llb.bench.memory.crossover_restatement.design import (
    load_restatement_design,
    published_crossovers,
)
from llb.bench.published_value.figures import (
    published_figures_for_cells,
    validate_published_cell_ids,
)
from llb.bench.published_value.fixture import load_provenance_fixture
from llb.core.paths import PROJECT_ROOT

RESTATEMENT = PROJECT_ROOT / "samples/benchmarks/agentic_compact_crossover_restatement_design.json"


def test_every_committed_figure_names_exactly_the_cells_its_artifact_measured():
    """The cell list is evidence, not decoration: it must match the committed aggregate."""
    values = published_crossovers(load_restatement_design(RESTATEMENT))
    validate_published_cell_ids(values, root=PROJECT_ROOT)
    assert all(value.get("cell_ids") for value in values)


def test_an_artifact_the_evidence_does_not_carry_is_refused_before_any_cell_is_read(tmp_path: Path):
    """The path resolution check: a figure cannot retire cells against an absent aggregate."""
    values = published_crossovers(load_restatement_design(RESTATEMENT))
    broken = [{**values[0], "provenance": {**values[0]["provenance"], "artifact": "no/such.json"}}]
    with pytest.raises(ValueError, match="no committed copy of 'no/such.json'"):
        validate_published_cell_ids(broken, root=PROJECT_ROOT)


def test_stale_or_narrowed_cell_ids_are_refused():
    """Hand-editing the list away from the aggregate is the second-mapping failure this closes."""
    values = published_crossovers(load_restatement_design(RESTATEMENT))
    narrowed = [{**values[0], "cell_ids": list(values[0]["cell_ids"])[:1]}]
    with pytest.raises(ValueError, match="must be exactly the cells"):
        validate_published_cell_ids(narrowed, root=PROJECT_ROOT)


def test_invalidated_cells_retire_their_figures_and_derived_consequences():
    """A surface cell retires its interpolated guard and the portable ratio that rests on it."""
    figures = published_figures_for_cells(
        [
            {
                "study_kind": "compact_memory_boundary_surface",
                "cell_id": "surface-d10-g23000",
            }
        ]
    )
    forms = {(figure.study, figure.depth, figure.form) for figure in figures}
    assert ("compact_memory_boundary_surface", 10, "interpolated_guard") in forms
    assert ("compact_trigger_guard_collapse", 10, "portable_trigger_ratio") in forms
    assert any("21899.890064587056" in figure.statement for figure in figures)
    assert any("surface-d10-g23000" in figure.named() for figure in figures)


def test_cells_that_no_figure_names_retire_nothing():
    assert (
        published_figures_for_cells(
            [{"study_kind": "compact_memory_boundary_surface", "cell_id": "not-a-published-cell"}]
        )
        == ()
    )


def test_the_shipped_aggregates_still_resolve_every_declared_cell_id():
    """Each cell_id is reachable through the same pointer walk a figure field uses."""
    fixture = load_provenance_fixture(PROJECT_ROOT)
    for value in published_crossovers(load_restatement_design(RESTATEMENT)):
        payload = fixture[value["provenance"]["artifact"]].payload
        for cell_id in value["cell_ids"]:
            from llb.bench.published_value.pointer import read_field

            assert read_field(payload, f"cells[cell_id={cell_id}].cell_id", where="test") == cell_id
