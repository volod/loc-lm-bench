"""The committed evidence is the union over every design that publishes resolvable values.

The prune that keeps the committed tree tracking its citations is correct exactly while ONE study
resolves published values through it. These tests drive the rule that keeps it correct for the
second one: the refresh walks a registry of designs, commits the union of what they cite, and
refuses to write a set that would retire a registered design's aggregates -- because that deletion
would otherwise read as a clean prune rather than as evidence loss.
"""

import json
from pathlib import Path

import pytest

from llb.bench.agentic_published_value_fixture import (
    committed_aggregate_path,
    load_provenance_fixture,
    write_provenance_fixture,
)
from llb.bench.agentic_published_value_registry import (
    KIND_CROSSOVER_RESTATEMENT,
    PUBLISHED_VALUE_DESIGNS,
    PublishedValueDesign,
    published_citations,
    refresh_committed_evidence,
)

FIRST = "first-study/run/analysis.json"
SECOND = "second-study/run/analysis.json"


def _design(design_root: Path, name: str, artifacts: list[str]) -> PublishedValueDesign:
    """A registered design standing for a study, with the artifacts its published values cite."""
    path = design_root / "samples/benchmarks" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"study_kind": name}), encoding="utf-8")
    return PublishedValueDesign(
        design_path=f"samples/benchmarks/{name}.json",
        cited_artifacts=lambda _path: list(artifacts),
    )


def _register(monkeypatch, designs: dict[str, PublishedValueDesign]) -> None:
    monkeypatch.setattr(
        "llb.bench.agentic_published_value_registry.PUBLISHED_VALUE_DESIGNS", designs
    )


def _host(data_dir: Path, *artifacts: str) -> Path:
    """A DATA_DIR carrying one run artifact per name, each with its own bytes."""
    for artifact in artifacts:
        path = data_dir / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"value": artifact}), encoding="utf-8")
    return data_dir


def _two_studies(tmp_path, monkeypatch) -> Path:
    """Two registered designs, one artifact each, with the evidence regenerated from both."""
    _register(
        monkeypatch,
        {
            "first": _design(tmp_path, "first", [FIRST]),
            "second": _design(tmp_path, "second", [SECOND]),
        },
    )
    data_dir = _host(tmp_path / "data", FIRST, SECOND)
    refresh_committed_evidence(root=tmp_path, data_dir=data_dir)
    return data_dir


def test_the_refresh_commits_the_union_over_every_registered_design(tmp_path, monkeypatch):
    """Two studies, one shared tree: refreshing through the registry carries both studies' bytes."""
    data_dir = _two_studies(tmp_path, monkeypatch)

    assert set(load_provenance_fixture(tmp_path)) == {FIRST, SECOND}
    for artifact in (FIRST, SECOND):
        assert (
            committed_aggregate_path(tmp_path, artifact).read_bytes()
            == (data_dir / artifact).read_bytes()
        )


def test_a_refresh_that_would_retire_another_registered_design_s_evidence_is_refused(
    tmp_path, monkeypatch
):
    """The landmine, stated directly: writing one design's citations must not prune the other's.

    The pruning refresh cannot tell evidence loss from housekeeping, so the write seam refuses any
    set that omits an artifact the registry says is still cited, whatever built that set.
    """
    _two_studies(tmp_path, monkeypatch)
    citations = published_citations(tmp_path)

    with pytest.raises(ValueError, match=r"would drop the committed copy of .*second-study"):
        write_provenance_fixture(tmp_path, {FIRST: b'{"value": "only mine"}'}, cited_by=citations)
    assert set(load_provenance_fixture(tmp_path)) == {FIRST, SECOND}


def test_a_host_missing_one_registered_design_s_run_refuses_rather_than_pruning_it(
    tmp_path, monkeypatch
):
    """A partial host is the ordinary way a study's evidence would have been silently retired."""
    _two_studies(tmp_path, monkeypatch)
    partial = _host(tmp_path / "partial", FIRST)

    with pytest.raises(ValueError, match=r"^second: the run artifact second-study"):
        refresh_committed_evidence(root=tmp_path, data_dir=partial)
    assert set(load_provenance_fixture(tmp_path)) == {FIRST, SECOND}


def test_an_artifact_two_designs_cite_is_committed_once_and_names_both(tmp_path, monkeypatch):
    """Studies that publish out of the same aggregate share one copy, not two."""
    _register(
        monkeypatch,
        {
            "first": _design(tmp_path, "first", [FIRST]),
            "second": _design(tmp_path, "second", [FIRST, SECOND]),
        },
    )
    data_dir = _host(tmp_path / "data", FIRST, SECOND)
    refresh_committed_evidence(root=tmp_path, data_dir=data_dir)

    assert published_citations(tmp_path) == {FIRST: ["first", "second"], SECOND: ["second"]}
    assert set(load_provenance_fixture(tmp_path)) == {FIRST, SECOND}


def test_a_design_the_registry_names_but_the_repo_does_not_carry_refuses_the_refresh(
    tmp_path, monkeypatch
):
    """An unreadable design's citations are unknown, and unknown is not the same as none."""
    _two_studies(tmp_path, monkeypatch)
    (tmp_path / "samples/benchmarks/second.json").unlink()

    with pytest.raises(ValueError, match="does not exist"):
        refresh_committed_evidence(root=tmp_path, data_dir=tmp_path / "data")
    assert set(load_provenance_fixture(tmp_path)) == {FIRST, SECOND}


def test_retiring_a_design_from_the_registry_is_what_prunes_its_evidence(tmp_path, monkeypatch):
    """The prune still works -- it is now driven by the registry rather than by one refresh call."""
    data_dir = _two_studies(tmp_path, monkeypatch)
    _register(monkeypatch, {"first": _design(tmp_path, "first", [FIRST])})
    refresh_committed_evidence(root=tmp_path, data_dir=data_dir)

    assert set(load_provenance_fixture(tmp_path)) == {FIRST}
    assert not committed_aggregate_path(tmp_path, SECOND).exists()


# --- the committed registry ---------------------------------------------------------------------


def test_the_shipped_registry_names_the_restatement_design_the_repo_carries():
    """The one design that publishes resolvable values today, and where the repo keeps it."""
    from llb.core.paths import PROJECT_ROOT

    assert set(PUBLISHED_VALUE_DESIGNS) == {KIND_CROSSOVER_RESTATEMENT}
    for design in PUBLISHED_VALUE_DESIGNS.values():
        assert (PROJECT_ROOT / design.design_path).is_file()
