"""What a proposed generation swap invalidates, read off the repo's own committed evidence.

The repo-level tests below run against the real register and the real committed evidence, so the
acceptance they state is the one an operator gets: the aggregates and published values measured on
`mistral-small3.1:24b` are listed for a Mistral swap and for no other family's. The synthetic root
covers the shapes the repo does not currently hold -- an empty evidence tree, an unreadable surface,
and an aggregate naming a model the register cannot place.
"""

import json
import os
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llb.backends.invalidation import (
    ADOPTION,
    BASELINE_TABLES,
    COMMITTED_AGGREGATES,
    PUBLISHED_VALUES,
    ROLLBACK,
    UNORDERED,
    render_json,
    render_text,
    report_invalidation,
    swap_direction,
)
from llb.backends.invalidation.surfaces import BASELINE_DOC_ROOT
from llb.backends.roster import load_register
from llb.bench.published_value.fixture import write_provenance_fixture
from llb.bench.published_value.registry import PUBLISHED_VALUE_DESIGNS
from llb.core.paths import PROJECT_ROOT
from llb.main import app

ROSTER = PROJECT_ROOT / "samples" / "configs" / "models_uk.yaml"
MISTRAL_TAG = "mistral-small3.1:24b"
UNPLACED_TAG = "llama3.2:3b"
ARTIFACT = "study/20260101T000000.000000Z-abcdef012345/analysis.json"

TABLE = """# Baseline

| model | served artifact | tok/s |
| --- | --- | ---: |
| `qwen3.8-27b` | `qwen3.8:27b` | 10.38 |
"""


@pytest.fixture
def register():
    return load_register(ROSTER)


def _aggregate(root: Path, model: str) -> None:
    payload = {"study_id": "synthetic", "held_fixed": {"model": model, "backend": "ollama"}}
    write_provenance_fixture(root, {ARTIFACT: json.dumps(payload).encode("utf-8")})


def _designs(root: Path) -> None:
    for design in PUBLISHED_VALUE_DESIGNS.values():
        target = root / design.design_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / design.design_path, target)


@pytest.fixture
def synthetic(tmp_path):
    """A repo-shaped root holding one aggregate, the registered designs, and no baseline tables."""
    (tmp_path / BASELINE_DOC_ROOT).mkdir(parents=True)
    _aggregate(tmp_path, MISTRAL_TAG)
    _designs(tmp_path)
    return tmp_path


# --- the repo's own evidence -------------------------------------------------------------


def test_a_committed_aggregate_measured_on_the_outgoing_generation_is_listed(register) -> None:
    report = report_invalidation(register, "mistral", "4", root=PROJECT_ROOT)

    aggregates = report.by_surface(COMMITTED_AGGREGATES)
    assert aggregates
    assert all(record.recorded == MISTRAL_TAG for record in aggregates)
    assert all(record.detail == "held_fixed.model" for record in aggregates)
    assert all(record.location.endswith(".json") for record in aggregates)


def test_a_swap_of_an_unrelated_family_lists_none_of_them(register) -> None:
    """The whole point of resolving through the register: a Gemma swap is not a Mistral re-run."""
    report = report_invalidation(register, "gemma", "5", root=PROJECT_ROOT)

    assert report.by_surface(COMMITTED_AGGREGATES) == ()
    assert report.by_surface(PUBLISHED_VALUES) == ()
    assert all(record.resolved.family_id == "gemma" for record in report.invalidated)
    assert report.scanned > len(report.invalidated)


def test_every_published_value_is_listed_rather_than_its_design(register) -> None:
    report = report_invalidation(register, "mistral", "4", root=PROJECT_ROOT)

    values = report.by_surface(PUBLISHED_VALUES)
    assert len(values) > len({record.location for record in values})
    assert all("study_kind=" in record.detail for record in values)


def test_a_baseline_row_is_located_by_file_and_line(register) -> None:
    report = report_invalidation(register, "qwen", "4", root=PROJECT_ROOT)

    rows = report.by_surface(BASELINE_TABLES)
    assert rows
    for record in rows:
        path, _, line = record.location.rpartition(":")
        assert (PROJECT_ROOT / path).is_file()
        assert int(line) > 0


def test_one_baseline_row_is_reported_once_even_when_two_columns_name_it(register) -> None:
    report = report_invalidation(register, "mistral", "4", root=PROJECT_ROOT)

    rows = report.by_surface(BASELINE_TABLES)
    assert len(rows) == len({(record.location, record.resolved.model_name) for record in rows})


def test_the_cost_is_reported_in_roster_entries_as_well_as_records(register) -> None:
    """A record count sizes the edit; the entry count sizes the run that has to be re-taken."""
    report = report_invalidation(register, "mistral", "4", root=PROJECT_ROOT)

    assert report.entries == ("mistral-small-3.1-24b",)
    assert len(report.invalidated) > len(report.entries)
    assert "mistral-small-3.1-24b" in render_text(report)
    assert json.loads(render_json(report))["entries"] == list(report.entries)


def test_a_family_carrying_several_models_names_all_of_them(register) -> None:
    report = report_invalidation(register, "gemma", "5", root=PROJECT_ROOT)

    assert len(report.entries) > 1
    assert all(entry.startswith("gemma-4") for entry in report.entries)


def test_the_report_names_the_surfaces_it_walked(register) -> None:
    """A walk that read nothing and a walk that found nothing must not print the same."""
    report = report_invalidation(register, "mistral", "4", root=PROJECT_ROOT)

    assert [reading.surface for reading in report.readings] == [
        COMMITTED_AGGREGATES,
        PUBLISHED_VALUES,
        BASELINE_TABLES,
    ]
    assert report.unread == ()
    text = render_text(report)
    assert all(reading.describe in text for reading in report.readings)
    assert json.loads(render_json(report))["scanned"] == report.scanned


# --- the swap itself ---------------------------------------------------------------------


def test_a_swap_to_the_generation_already_carried_is_refused(register) -> None:
    with pytest.raises(ValueError, match="already carries generation `3.1`"):
        report_invalidation(register, "mistral", "3.1", root=PROJECT_ROOT)


def test_an_unregistered_family_is_refused_with_the_ones_that_exist(register) -> None:
    with pytest.raises(ValueError, match="no such family in the register"):
        report_invalidation(register, "llama", "4", root=PROJECT_ROOT)


def test_a_rollback_costs_what_an_adoption_costs(register) -> None:
    forward = report_invalidation(register, "qwen", "4", root=PROJECT_ROOT)
    back = report_invalidation(register, "qwen", "3.6", root=PROJECT_ROOT)

    assert (forward.direction, back.direction) == (ADOPTION, ROLLBACK)
    assert forward.invalidated == back.invalidated


def test_a_generation_pair_nothing_orders_is_reported_as_unordered() -> None:
    assert swap_direction("3.1", "release-x") == UNORDERED
    assert swap_direction("v0.1.2", "v0.1.3") == ADOPTION


# --- the shapes the repo does not currently hold ------------------------------------------


def test_nothing_affected_is_a_sentence_beside_the_surfaces_that_were_read(
    register, synthetic
) -> None:
    report = report_invalidation(register, "gemma", "5", root=synthetic)

    assert report.invalidated == ()
    assert report.unread == ()
    assert report.scanned > 0
    assert "nothing published or committed was measured" in render_text(report)


def test_a_recorded_model_the_register_cannot_place_is_reported(register, tmp_path) -> None:
    (tmp_path / BASELINE_DOC_ROOT).mkdir(parents=True)
    _aggregate(tmp_path, UNPLACED_TAG)
    _designs(tmp_path)

    report = report_invalidation(register, "mistral", "4", root=tmp_path)

    assert [record.recorded for record in report.unresolved] == [UNPLACED_TAG]
    assert "cannot place" in render_text(report)


def test_an_unreadable_surface_becomes_a_stated_reason_rather_than_a_lost_report(
    register, tmp_path
) -> None:
    """Losing a surface silently understates a swap in the direction that makes it look cheap."""
    (tmp_path / BASELINE_DOC_ROOT).mkdir(parents=True)
    _designs(tmp_path)

    report = report_invalidation(register, "mistral", "4", root=tmp_path)

    assert [reading.surface for reading in report.unread] == [COMMITTED_AGGREGATES]
    assert "UNREAD" in render_text(report)


def test_a_baseline_table_under_the_docs_root_is_read_from_any_root(register, synthetic) -> None:
    table = synthetic / BASELINE_DOC_ROOT / "telemetry.md"
    table.write_text(TABLE, encoding="utf-8")

    report = report_invalidation(register, "qwen", "4", root=synthetic)

    assert [record.location for record in report.by_surface(BASELINE_TABLES)] == [
        f"{BASELINE_DOC_ROOT.as_posix()}/telemetry.md:5"
    ]


def test_a_relative_root_still_locates_a_baseline_row_inside_the_repo(register, synthetic) -> None:
    """The doc locations are repo-relative, so the root a caller hands in must not change them."""
    (synthetic / BASELINE_DOC_ROOT / "telemetry.md").write_text(TABLE, encoding="utf-8")
    relative = Path(os.path.relpath(synthetic, Path.cwd()))

    report = report_invalidation(register, "qwen", "4", root=relative)

    assert [record.location for record in report.by_surface(BASELINE_TABLES)] == [
        f"{BASELINE_DOC_ROOT.as_posix()}/telemetry.md:5"
    ]


# --- the command -------------------------------------------------------------------------


def test_the_command_reports_and_strict_exits_non_zero_when_a_swap_costs_something() -> None:
    runner = CliRunner()

    reported = runner.invoke(app, ["report-generation-invalidation", "mistral", "4"])
    strict = runner.invoke(app, ["report-generation-invalidation", "mistral", "4", "--strict"])

    assert reported.exit_code == 0
    assert MISTRAL_TAG in reported.stdout
    assert strict.exit_code == 1


def test_the_command_refuses_an_unknown_family() -> None:
    result = CliRunner().invoke(app, ["report-generation-invalidation", "llama", "4"])

    assert result.exit_code == 2
