"""What a refresh says about the evidence it just committed.

Regenerating the committed aggregates and resolving the published values out of them are the same
question asked of the same registry, and for a while only the first half was answered at the refresh:
an operator who re-ran a study, re-committed its aggregates, and read a clean `provenance manifest
-> ...` learned on some LATER `make ci` that the design's numbers are now the old run's -- at the
point where the fix is a design edit and the run context is gone.

These tests drive the half that closes it: the refresh validates what it just wrote, names the
values that no longer resolve, and exits non-zero -- while leaving the write in place, because a
refresh after a re-run is exactly the repair flow and a rollback would leave the operator unable to
commit the new evidence at all.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llb.bench.agentic_memory_crossover_restatement_design import (
    load_restatement_design,
    published_crossovers,
)
from llb.bench.agentic_published_value_fixture import (
    committed_aggregate_path,
    load_provenance_fixture,
)
from llb.bench.agentic_published_value_provenance import provenance_pair
from llb.bench.agentic_published_value_registry import (
    KIND_CROSSOVER_RESTATEMENT,
    PublishedValueDesign,
    refresh_committed_evidence_and_report,
    report_published_designs,
    validate_published_designs,
)
from llb.cli.bench.category_agentic_memory_crossover_restatement import (
    EXIT_UNRESOLVED_PUBLISHED_VALUES,
)
from llb.main import app

ROOT = Path(__file__).resolve().parents[3]
DESIGN_PATH = ROOT / "samples/benchmarks/agentic_compact_crossover_restatement_design.json"
SURFACE_ARTIFACT = (
    "agentic-compact-memory-boundary-surface/"
    "20260802T154634.305722Z-c668820b6c4d/boundary-surface-analysis.json"
)
FIRST = "first-study/run/analysis.json"
SECOND = "second-study/run/analysis.json"


def _accept(_design_path: Path, *, root: Path, data_dir: Path | None) -> None:
    """A validation that resolves every published value, for the designs a test is not about."""


def _refuse(reason: str):
    """A validation standing for a design whose published numbers the evidence no longer states."""

    def validate(_design_path: Path, *, root: Path, data_dir: Path | None) -> None:
        raise ValueError(reason)

    return validate


def _design(
    design_root: Path, name: str, artifacts: list[str], *, validate: object = _accept
) -> PublishedValueDesign:
    path = design_root / "samples/benchmarks" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"study_kind": name}), encoding="utf-8")
    return PublishedValueDesign(
        design_path=f"samples/benchmarks/{name}.json",
        published_values=lambda _path: [],
        cited_artifacts=lambda _path: list(artifacts),
        validate_published_values=validate,
    )


def _register(monkeypatch, designs: dict[str, PublishedValueDesign]) -> None:
    monkeypatch.setattr(
        "llb.bench.agentic_published_value_registry.PUBLISHED_VALUE_DESIGNS", designs
    )


def _host(data_dir: Path, *artifacts: str) -> Path:
    for artifact in artifacts:
        path = data_dir / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"value": artifact}), encoding="utf-8")
    return data_dir


# --- the walk collects rather than stopping -----------------------------------------------------


def test_the_walk_reports_every_design_whose_published_values_do_not_resolve(tmp_path, monkeypatch):
    """One name per re-run of the refresh is the loop this replaces, so the walk finishes."""
    _register(
        monkeypatch,
        {
            "first": _design(tmp_path, "first", [FIRST], validate=_refuse("first depth 6 moved")),
            "second": _design(
                tmp_path, "second", [SECOND], validate=_refuse("second depth 10 moved")
            ),
        },
    )

    report = report_published_designs(root=tmp_path)
    assert report.walked == ("first", "second")
    assert [unresolved.kind for unresolved in report.unresolved] == ["first", "second"]
    assert [unresolved.reason for unresolved in report.unresolved] == [
        "first depth 6 moved",
        "second depth 10 moved",
    ]


def test_a_clean_walk_names_the_designs_it_spoke_for(tmp_path, monkeypatch):
    """A walk that checked nothing reads exactly like one that checked everything, so it counts."""
    _register(monkeypatch, {"first": _design(tmp_path, "first", [FIRST])})

    report = report_published_designs(root=tmp_path)
    assert report.walked == ("first",)
    assert report.unresolved == ()


def test_the_ci_refusal_names_every_unresolved_design_rather_than_the_first(tmp_path, monkeypatch):
    """The refusing form is the collecting one plus a raise, so CI reads the same list."""
    _register(
        monkeypatch,
        {
            "first": _design(tmp_path, "first", [FIRST], validate=_refuse("first depth 6 moved")),
            "second": _design(
                tmp_path, "second", [SECOND], validate=_refuse("second depth 10 moved")
            ),
        },
    )

    with pytest.raises(ValueError) as excinfo:
        validate_published_designs(root=tmp_path)
    assert "first depth 6 moved" in str(excinfo.value)
    assert "second depth 10 moved" in str(excinfo.value)


# --- the refresh answers its own question -------------------------------------------------------


def test_the_refresh_reports_unresolved_values_and_still_commits_what_it_wrote(
    tmp_path, monkeypatch
):
    """Reported, never rolled back: the new aggregates are what the operator must commit first."""
    _register(
        monkeypatch,
        {
            "first": _design(tmp_path, "first", [FIRST]),
            "second": _design(
                tmp_path, "second", [SECOND], validate=_refuse("second depth 10 moved")
            ),
        },
    )
    data_dir = _host(tmp_path / "data", FIRST, SECOND)

    manifest, report = refresh_committed_evidence_and_report(root=tmp_path, data_dir=data_dir)

    assert set(load_provenance_fixture(tmp_path)) == {FIRST, SECOND}
    assert manifest.is_file()
    assert report.walked == ("first", "second")
    assert [unresolved.named() for unresolved in report.unresolved] == [
        "second (samples/benchmarks/second.json): second depth 10 moved"
    ]


def test_a_refresh_whose_designs_all_resolve_reports_no_unresolved_value(tmp_path, monkeypatch):
    """The ordinary refresh, which must still say so rather than only echoing a manifest path."""
    _register(monkeypatch, {"first": _design(tmp_path, "first", [FIRST])})
    data_dir = _host(tmp_path / "data", FIRST)

    _manifest, report = refresh_committed_evidence_and_report(root=tmp_path, data_dir=data_dir)
    assert report.walked == ("first",)
    assert report.unresolved == ()


# --- the case this exists for: a re-run moved a published value ---------------------------------


def test_a_re_run_that_moved_a_published_guard_is_named_by_the_refresh_that_commits_it(tmp_path):
    """The real design against a host whose boundary-surface run now measures a different guard.

    The committed evidence is regenerated into a throwaway copy of the repo's design files, which is
    what the operator's repair flow does to the tracked ones: the moved aggregate lands in the
    manifest, and the refresh says which published number that aggregate no longer states instead of
    deferring it to `make ci`.
    """
    root = _design_copy(tmp_path)
    data_dir = _moved_host(tmp_path, depth=10, guard=21525.5)

    manifest, report = refresh_committed_evidence_and_report(root=root, data_dir=data_dir)

    committed = json.loads(
        committed_aggregate_path(root, SURFACE_ARTIFACT).read_text(encoding="utf-8")
    )
    assert manifest.is_file()
    assert committed["depth_surface"][1]["crossover_max_prompt_chars"] == 21525.5
    assert report.walked == (KIND_CROSSOVER_RESTATEMENT,)
    assert len(report.unresolved) == 1
    reason = report.unresolved[0].reason
    assert "compact_memory_boundary_surface depth 10" in reason
    assert "21899.890064587056" in reason and "21525.5" in reason


def test_the_refresh_command_exits_non_zero_and_names_the_value_it_no_longer_states(monkeypatch):
    """The user-visible outcome: `make bench-agentic-published-provenance` answers, and fails."""
    from llb.bench.agentic_published_value_registry import PublishedValueReport, UnresolvedDesign

    report = PublishedValueReport(
        walked=(KIND_CROSSOVER_RESTATEMENT,),
        unresolved=(
            UnresolvedDesign(
                kind=KIND_CROSSOVER_RESTATEMENT,
                design_path="samples/benchmarks/agentic_compact_crossover_restatement_design.json",
                reason="compact_memory_boundary_surface depth 10: the design publishes 21899.89",
            ),
        ),
    )
    monkeypatch.setattr(
        "llb.bench.agentic_published_value_registry.refresh_committed_evidence_and_report",
        lambda **_kwargs: (Path("manifest.json"), report),
    )

    result = CliRunner().invoke(
        app,
        [
            "bench-agentic-context-compact-crossover-restatement",
            "--design",
            str(DESIGN_PATH),
            "--refresh-provenance",
        ],
    )

    assert result.exit_code == EXIT_UNRESOLVED_PUBLISHED_VALUES
    assert "provenance manifest -> manifest.json" in result.output
    assert "compact_memory_boundary_surface depth 10" in result.output
    assert "the write stands" in result.output


def test_the_refresh_command_says_so_when_every_published_value_still_resolves(monkeypatch):
    """A clean refresh states what it checked, so silence is not what a pass looks like."""
    from llb.bench.agentic_published_value_registry import PublishedValueReport

    monkeypatch.setattr(
        "llb.bench.agentic_published_value_registry.refresh_committed_evidence_and_report",
        lambda **_kwargs: (
            Path("manifest.json"),
            PublishedValueReport(walked=(KIND_CROSSOVER_RESTATEMENT,), unresolved=()),
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "bench-agentic-context-compact-crossover-restatement",
            "--design",
            str(DESIGN_PATH),
            "--refresh-provenance",
        ],
    )

    assert result.exit_code == 0
    assert "every published value resolves out of the evidence just committed" in result.output
    assert f"walked: {KIND_CROSSOVER_RESTATEMENT}" in result.output


def _design_copy(tmp_path: Path) -> Path:
    """A throwaway repo root carrying the committed design files, but none of the evidence.

    Copied rather than pointed at: the refresh WRITES, and a test that regenerated the tracked
    fixture it was checking would report a pass by having repaired it.
    """
    root = tmp_path / "repo"
    benchmarks = root / "samples/benchmarks"
    benchmarks.mkdir(parents=True)
    for design in sorted((ROOT / "samples/benchmarks").glob("*.json")):
        (benchmarks / design.name).write_bytes(design.read_bytes())
    return root


def _moved_host(tmp_path: Path, *, depth: int, guard: float) -> Path:
    """A DATA_DIR holding the repo's committed aggregates with one published guard re-measured.

    Built from the committed copies rather than from this host's runs, so the case is reproducible
    on a machine that never ran the studies -- the copies ARE the run's bytes, so moving one field
    is exactly what a re-run of that study would hand the refresh.
    """
    data_dir = tmp_path / "data"
    for artifact in load_provenance_fixture(ROOT):
        payload = json.loads(committed_aggregate_path(ROOT, artifact).read_text(encoding="utf-8"))
        if artifact == SURFACE_ARTIFACT:
            for row in payload["depth_surface"]:
                if int(row["depth"]) == depth:
                    row["crossover_max_prompt_chars"] = guard
        path = data_dir / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return data_dir


def test_the_moved_host_helper_cites_the_artifact_the_committed_design_publishes_from():
    """The fixture above is only the right case while the design still resolves through it."""
    cited = {
        provenance_pair(crossover["provenance"], where="test")[0]
        for crossover in published_crossovers(load_restatement_design(DESIGN_PATH))
    }
    assert SURFACE_ARTIFACT in cited
