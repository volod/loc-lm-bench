"""Every published crossover, resolved back to the aggregate that measured it.

The fold-step annotation check catches a transcription slip only when it is large enough to leave
the published step, which no realistic slip is: a dropped digit in `14159.929807575942` lands well
inside step 6's `[13136, 14912)`. These tests drive the rule that closes that -- each published
value is read back out of the repo's own verbatim copy of its run aggregate, and a design that
states anything else is refused before a single cell runs.
"""

from copy import deepcopy
import json
from pathlib import Path
from typing import cast

import pytest

from llb.bench.agentic_memory_crossover_restatement_design import (
    load_restatement_design,
    published_crossovers,
    validate_restatement_design,
)
from llb.bench.agentic_memory_crossover_restatement_provenance import (
    validate_published_provenance,
)
from llb.bench.agentic_memory_crossover_restatement_reading import (
    FORM_FOLD_STEP,
    FORM_INTERPOLATED,
    FORM_PORTABLE_RATIO,
)
from llb.bench.agentic_published_value_fixture import (
    MAX_AGGREGATE_BYTES,
    MAX_COMMITTED_BYTES,
    PROVENANCE_FIXTURE,
    committed_aggregate_path,
    committed_evidence_bytes,
    load_provenance_fixture,
    write_provenance_fixture,
)
from llb.bench.agentic_published_value_derivation import (
    DERIVED_FROM,
    ValueKey,
    required_derivation,
)
from llb.bench.agentic_published_value_operations import (
    OPERATION,
    OPERATION_TRIGGER_OVER_OWN_CAP_PEAK,
)
from llb.bench.agentic_published_value_provenance import provenance_pair
from llb.bench.agentic_published_value_registry import (
    published_citations,
    refresh_committed_evidence,
)
from llb.bench.agentic_policy_change_audit import KIND_COLLAPSE, KIND_FOLD_STEP, KIND_SURFACE
from llb.core.paths import resolve_data_dir

ROOT = Path(__file__).resolve().parents[3]
DESIGN_PATH = ROOT / "samples/benchmarks/agentic_compact_crossover_restatement_design.json"


def _apply(design: dict[str, object], kind: str, depth: int, **fields: object) -> dict[str, object]:
    """Override one published crossover's fields in place, and hand the design back."""
    for study in cast(list[dict[str, object]], design["audited_studies"]):
        if study["study_kind"] != kind:
            continue
        for crossover in cast(list[dict[str, object]], study["published_crossovers"]):
            if int(cast(int, crossover["depth"])) == depth:
                crossover.update(fields)
    return design


def _mutate(kind: str, depth: int, **fields: object) -> dict[str, object]:
    """The committed design with one published crossover's fields overridden."""
    return _apply(deepcopy(load_restatement_design(DESIGN_PATH)), kind, depth, **fields)


def _both_bands(**fields: object) -> dict[str, object]:
    """The band is published identically at both depths, so a band test must move both rows."""
    design = _mutate(KIND_COLLAPSE, 6, **fields)
    return _apply(design, KIND_COLLAPSE, 10, **fields)


def _drop_provenance(kind: str, depth: int) -> dict[str, object]:
    design = _mutate(kind, depth)
    for study in cast(list[dict[str, object]], design["audited_studies"]):
        for crossover in cast(list[dict[str, object]], study["published_crossovers"]):
            if study["study_kind"] == kind and int(cast(int, crossover["depth"])) == depth:
                crossover.pop("provenance")
    return design


# --- the committed design ---------------------------------------------------------------------


def test_every_committed_crossover_resolves_out_of_the_aggregate_it_cites():
    """The positive control, and the one CI runs: six published numbers, no run on the host."""
    design = load_restatement_design(DESIGN_PATH)
    validate_restatement_design(design, root=ROOT)

    committed = load_provenance_fixture(ROOT)
    for crossover in published_crossovers(design):
        artifact, _field = provenance_pair(crossover["provenance"], where="test")
        assert artifact in committed


def test_every_committed_derived_value_declares_the_edge_and_the_arithmetic_over_it():
    """Both halves read off the design now, so the design has to state both.

    Asserted in both directions: each band names the guard it is a quotient of AND the registered
    operation that divides it, and no MEASURED crossover names either -- a measurement that declared
    a source would be marked not-judged for a number it does not rest on.
    """
    rows = published_crossovers(load_restatement_design(DESIGN_PATH))
    derived = [row for row in rows if row["form"] == FORM_PORTABLE_RATIO]
    assert len(derived) == 2
    for row in derived:
        derivation = required_derivation(row)
        assert derivation.source_of_form(FORM_INTERPOLATED) == ValueKey(
            study_kind=KIND_SURFACE, depth=int(cast(int, row["depth"])), form=FORM_INTERPOLATED
        )
        assert derivation.operation.name == OPERATION_TRIGGER_OVER_OWN_CAP_PEAK
    measured = [row for row in rows if row["form"] != FORM_PORTABLE_RATIO]
    assert all(row.get(DERIVED_FROM) is None and row.get(OPERATION) is None for row in measured)


def test_the_committed_evidence_is_the_union_over_every_registered_design():
    """A copy nothing resolves through is an aggregate every registered design stopped citing.

    Read off the registry rather than off this one design, because the tree is shared: the day a
    second study adopts the resolver, "what the restatement cites" stops being the right set and a
    refresh that used it would prune the newcomer's evidence.
    """
    assert set(load_provenance_fixture(ROOT)) == set(published_citations(ROOT))
    assert _cited_artifacts() <= set(published_citations(ROOT))


def test_the_committed_evidence_stays_inside_the_growth_budget():
    """The evidence is bounded on purpose: it grows with every study that adopts the seam.

    The budget is what makes committing the bytes the right trade against a signed manifest. A study
    whose analysis artifact outgrows it publishes its values out of a narrower one instead.
    """
    total = committed_evidence_bytes(ROOT)
    assert 0 < total <= MAX_COMMITTED_BYTES
    for artifact in published_citations(ROOT):
        assert committed_aggregate_path(ROOT, artifact).stat().st_size <= MAX_AGGREGATE_BYTES


@pytest.mark.parametrize(
    ("kind", "depth", "value"),
    [
        # The slip the annotation rule cannot see: still inside fold step 6's guard interval.
        (KIND_SURFACE, 6, 14159.92980757594),
        (KIND_SURFACE, 10, 21899.890064587),
    ],
)
def test_a_transcribed_interpolated_guard_is_refused_however_small_the_slip(kind, depth, value):
    with pytest.raises(ValueError, match="transcribed rather than resolved"):
        validate_restatement_design(_mutate(kind, depth, value=value), root=ROOT)


def test_a_fold_step_boundary_is_resolved_against_its_own_study_s_ladder_artifact(tmp_path):
    """The ladder rule pins this value too, so resolution ties it to the RUN as well as the geometry.

    They answer different questions: the ladder says the number is the one this geometry implies,
    and the aggregate says it is the one the study that published it actually recorded. A run whose
    recorded boundary is not the one the geometry implies is drift the ladder rule cannot see.
    """
    design = load_restatement_design(DESIGN_PATH)
    validate_published_provenance(published_crossovers(design), root=ROOT)

    with pytest.raises(ValueError, match="transcribed rather than resolved"):
        _validate_against(design, _re_run(ladder={6: 14913}), tmp_path)


@pytest.mark.parametrize("band", [[0.84, 0.92], [0.85, 0.93], [0.80, 0.99]])
def test_the_published_band_edges_are_re_derived_from_the_two_aggregates_that_produced_them(band):
    """The band is the one form no aggregate states, so its edges are resolved rather than read.

    The collapse measured the cap PEAK; the surface measured the guard. The band's edges are the
    rounded quotients of those, at the depths it was published across, through the runtime's own
    trigger arithmetic -- the same arithmetic the restatement applies to a restated guard. A wider
    band is refused as squarely as a wrong one: an edge nothing derives is an edge nobody measured.
    """
    with pytest.raises(ValueError, match="the band was transcribed rather than resolved"):
        validate_restatement_design(_both_bands(published_band=band), root=ROOT)


def test_a_band_stated_at_a_precision_the_quotients_do_not_reach_is_refused():
    """`band_decimals` is the reading, so widening it changes what the edges must resolve to.

    At three decimals the same two quotients are 0.845x and 0.918x, so the published two-decimal
    edges stop being what the aggregates derive.
    """
    with pytest.raises(ValueError, match="derive 0.845-0.918x"):
        validate_restatement_design(_both_bands(band_decimals=3), root=ROOT)


def test_a_crossover_with_no_provenance_at_all_is_refused():
    """The default has to be refusal: a value nothing points at is exactly the old state."""
    with pytest.raises(ValueError, match="must carry a `provenance` object"):
        validate_restatement_design(_drop_provenance(KIND_SURFACE, 10), root=ROOT)


def test_a_crossover_citing_an_aggregate_the_repo_does_not_carry_is_refused():
    """The evidence was garbage-collected, so nothing on a CI host can resolve the number."""
    design = _mutate(
        KIND_SURFACE,
        6,
        provenance={"artifact": "gone/run/analysis.json", "field": "depth_surface[depth=6].x"},
    )
    with pytest.raises(ValueError, match="no committed copy of 'gone/run/analysis.json'"):
        validate_restatement_design(design, root=ROOT)


def test_a_band_declaring_a_source_the_design_does_not_publish_is_refused_before_any_read():
    """The ratio divides a guard it NAMES, so dropping that row leaves the declaration dangling.

    Fail-fast alongside the other shape refusals rather than collected: a value pointing at a number
    the design does not publish is a design that never said what its band rests on, and that is not
    evidence which moved.
    """
    rows = [
        row
        for row in published_crossovers(load_restatement_design(DESIGN_PATH))
        if not (row["study_kind"] == KIND_SURFACE and row["depth"] == 6)
    ]
    with pytest.raises(ValueError, match="which this design does not publish"):
        validate_published_provenance(rows, root=ROOT)


# --- one refusal per re-run, not one name per re-run --------------------------------------------


def test_a_re_run_that_moved_several_published_values_names_every_one_of_them(tmp_path):
    """The loop this ends: restate one number, re-run the check, meet the next one, six times over.

    A re-run that moved three of this study's published numbers is three design edits, so the walk
    resolves every crossover and refuses once with all three named, in the order the design publishes
    them rather than in the order the resolution happens to reach them.
    """
    design = load_restatement_design(DESIGN_PATH)
    moved = _re_run(surface={10: 21525.5}, ladder={6: 14913, 10: 22017})

    with pytest.raises(ValueError) as excinfo:
        _validate_against(design, moved, tmp_path)

    message = str(excinfo.value)
    assert "3/6 published values do not resolve" in message
    named = [
        message.index(f"{KIND_SURFACE} depth 10"),
        message.index(f"{KIND_FOLD_STEP} depth 6"),
        message.index(f"{KIND_FOLD_STEP} depth 10"),
    ]
    assert named == sorted(named)
    assert message.count("transcribed rather than resolved") == 3


def test_a_moved_guard_is_named_as_the_cause_and_its_derived_band_is_not_named_again(tmp_path):
    """The band is a QUOTIENT of the guard, so reporting both names one moved measurement twice.

    Worse than noise: the second name sends the operator to restate a band that nothing here can
    evaluate, since its own edges are re-derived from the very guard that no longer resolves. The
    cause is named, and the band is marked not-judged against it.
    """
    design = load_restatement_design(DESIGN_PATH)

    with pytest.raises(ValueError) as excinfo:
        _validate_against(design, _re_run(surface={10: 21525.5}), tmp_path)

    message = str(excinfo.value)
    assert "1/6 published values do not resolve" in message
    assert f"{KIND_SURFACE} depth 10" in message
    assert f"[not judged] {KIND_COLLAPSE} depth 10" in message
    assert "the band was transcribed rather than resolved" not in message


def test_a_malformed_crossover_is_refused_before_any_published_value_is_read(tmp_path):
    """Shape refusals stay fail-fast: a design that never said what states a number has none to name.

    Checked against evidence that has ALSO moved, because the point is which message survives -- the
    defect is the design's shape, and it must not arrive underneath a list of values that could not
    be checked because of it.
    """
    design = _drop_provenance(KIND_SURFACE, 10)

    with pytest.raises(ValueError) as excinfo:
        _validate_against(design, _re_run(ladder={6: 14913}), tmp_path)

    message = str(excinfo.value)
    assert "must carry a `provenance` object" in message
    assert "do not resolve out of the evidence" not in message
    assert "transcribed rather than resolved" not in message


def test_the_committed_evidence_is_this_host_s_own_run_artifacts_byte_for_byte(tmp_path):
    """On a host with the runs, the committed copy is checked against the file rather than carried.

    This is the half of the check a CI host cannot run, and the half that makes the other half mean
    anything: bytes copied from some other run digest to something no cited artifact has.
    """
    data_dir = _host_data_dir()
    design = load_restatement_design(DESIGN_PATH)
    validate_published_provenance(published_crossovers(design), root=ROOT, data_dir=data_dir)

    artifact = _cited(FORM_INTERPOLATED)
    from_another_run = _committed_bytes()
    from_another_run[artifact] = from_another_run[artifact] + b"\n"
    with pytest.raises(ValueError, match="came from a different run"):
        _validate_against(design, from_another_run, tmp_path, data_dir=data_dir)


# --- regenerating the committed evidence --------------------------------------------------------


def test_the_committed_evidence_is_what_the_run_artifacts_still_hold(tmp_path):
    """On a host that ran the studies, regeneration must be a no-op -- otherwise the copy drifted.

    Regenerated into a throwaway root rather than over the committed one: a test that rewrites a
    tracked fixture reports a pass by having repaired what it was checking.
    """
    data_dir = _host_data_dir()
    regenerated = refresh_committed_evidence(root=tmp_path, data_dir=data_dir, design_root=ROOT)
    assert regenerated.read_text(encoding="utf-8") == (ROOT / PROVENANCE_FIXTURE).read_text(
        encoding="utf-8"
    )
    for artifact in _cited_artifacts():
        assert (
            committed_aggregate_path(tmp_path, artifact).read_bytes()
            == committed_aggregate_path(ROOT, artifact).read_bytes()
        )


def test_regeneration_refuses_a_host_that_does_not_have_the_run(tmp_path):
    """Better a named refusal than a fixture silently regenerated with an artifact left out."""
    with pytest.raises(ValueError, match="is not under DATA_DIR on this host"):
        refresh_committed_evidence(root=tmp_path, data_dir=tmp_path / "empty", design_root=ROOT)


def test_regeneration_commits_each_cited_artifact_once_and_verbatim(tmp_path):
    """Six crossovers over three aggregates become three copies, each the run's own bytes.

    Verbatim including the parts no crossover cites: an excerpt shaped by hand is what a reader
    would have to take on faith, and it is the whole reason the file itself is carried.
    """
    data_dir = _fake_host(tmp_path)
    assert set(load_provenance_fixture(tmp_path)) == set(_FAKE_ARTIFACTS)
    for artifact in _FAKE_ARTIFACTS:
        committed = committed_aggregate_path(tmp_path, artifact).read_bytes()
        assert committed == (data_dir / artifact).read_bytes()
        assert b"kept in the committed copy" in committed


def test_a_regenerated_copy_pins_the_file_it_was_taken_from(tmp_path):
    """Bytes and pin come out of one read, so an edited artifact re-pins rather than drifts."""
    data_dir = _fake_host(tmp_path)
    pinned = _pins(tmp_path)

    surface = data_dir / _cited(FORM_INTERPOLATED)
    surface.write_text(surface.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    refresh_committed_evidence(root=tmp_path, data_dir=data_dir, design_root=ROOT)
    repinned = _pins(tmp_path)
    assert repinned[_cited(FORM_INTERPOLATED)] != pinned[_cited(FORM_INTERPOLATED)]
    assert repinned[_cited(FORM_FOLD_STEP)] == pinned[_cited(FORM_FOLD_STEP)]


def _fake_host(tmp_path: Path) -> Path:
    """A DATA_DIR carrying one fake artifact per cited aggregate, with the evidence regenerated."""
    data_dir = tmp_path / "data"
    for artifact, payload in _FAKE_ARTIFACTS.items():
        (data_dir / artifact).parent.mkdir(parents=True, exist_ok=True)
        (data_dir / artifact).write_text(json.dumps(payload), encoding="utf-8")
    refresh_committed_evidence(root=tmp_path, data_dir=data_dir, design_root=ROOT)
    return data_dir


def _pins(root: Path) -> dict[str, str]:
    return {artifact: entry.digest for artifact, entry in load_provenance_fixture(root).items()}


def _committed_bytes(root: Path = ROOT) -> dict[str, bytes]:
    """The committed aggregates as bytes, ready to be mutated into a throwaway root."""
    return {
        artifact: committed_aggregate_path(root, artifact).read_bytes()
        for artifact in load_provenance_fixture(root)
    }


def _re_run(
    *, surface: dict[int, float] | None = None, ladder: dict[int, int] | None = None
) -> dict[str, bytes]:
    """The committed aggregates with named per-depth values re-measured, as a re-run would leave them.

    Built by moving fields of the committed copies rather than by running anything: the copies ARE
    the runs' bytes, so a moved field is exactly what the study would hand a refresh on a host that
    re-ran it -- and the case reproduces on a machine that never ran the studies at all.
    """
    moved = _committed_bytes()
    if surface:
        moved[_cited(FORM_INTERPOLATED)] = _moved_rows(
            moved[_cited(FORM_INTERPOLATED)],
            "depth_surface",
            {depth: {"crossover_max_prompt_chars": guard} for depth, guard in surface.items()},
        )
    if ladder:
        moved[_cited(FORM_FOLD_STEP)] = _moved_rows(
            moved[_cited(FORM_FOLD_STEP)],
            "depth_ladders",
            {depth: {"boundary": {"guard_boundary_chars": edge}} for depth, edge in ladder.items()},
        )
    return moved


def _moved_rows(payload: bytes, rows_key: str, moved: dict[int, dict[str, object]]) -> bytes:
    """One aggregate's per-depth rows, with the named depths' fields replaced."""
    analysis = json.loads(payload)
    for row in cast(list[dict[str, object]], analysis[rows_key]):
        fields = moved.get(int(cast(int, row["depth"])))
        if fields is None:
            continue
        for name, value in fields.items():
            if isinstance(value, dict):
                cast(dict[str, object], row[name]).update(cast(dict[str, object], value))
            else:
                row[name] = value
    return json.dumps(analysis).encode("utf-8")


def _validate_against(
    design: dict[str, object],
    aggregates: dict[str, bytes],
    tmp: Path,
    *,
    data_dir: Path | None = None,
) -> None:
    """Validate the design's provenance against a mutated set of committed aggregates."""
    write_provenance_fixture(tmp, aggregates)
    validate_published_provenance(published_crossovers(design), root=tmp, data_dir=data_dir)


def _host_data_dir() -> Path:
    """This host's DATA_DIR, or a skip when it never ran the studies the registry cites."""
    data_dir = resolve_data_dir()
    missing = [
        artifact for artifact in published_citations(ROOT) if not (data_dir / artifact).is_file()
    ]
    if missing:
        pytest.skip(f"this host never ran {missing[0]}")
    return data_dir


def _cited_artifacts() -> set[str]:
    """Every run artifact the committed design's published crossovers point at."""
    return {
        provenance_pair(crossover["provenance"], where="test")[0]
        for crossover in published_crossovers(load_restatement_design(DESIGN_PATH))
    }


def _cited(form: str) -> str:
    """The artifact the committed design cites for one published form."""
    return next(
        provenance_pair(row["provenance"], where="test")[0]
        for row in published_crossovers(load_restatement_design(DESIGN_PATH))
        if row["form"] == form
    )


# The three published forms read three differently shaped aggregates -- a list of per-depth surface
# rows, a mapping keyed by depth, and a list of ladders with a nested boundary. A fake set of all
# three proves the regeneration handles each shape, and the `unrelated` key proves it commits the
# artifact rather than an excerpt of it.
_FAKE_ARTIFACTS: dict[str, dict[str, object]] = {
    _cited(FORM_INTERPOLATED): {
        "depth_surface": [
            {"depth": 6, "crossover_max_prompt_chars": 14159.929807575942},
            {"depth": 10, "crossover_max_prompt_chars": 21899.890064587056},
        ],
        "unrelated": "kept in the committed copy",
    },
    _cited(FORM_PORTABLE_RATIO): {
        "cap_peak_prompt_chars": {"6": 8374, "10": 11926},
        "unrelated": "kept in the committed copy",
    },
    _cited(FORM_FOLD_STEP): {
        "depth_ladders": [
            {"depth": 6, "boundary": {"guard_boundary_chars": 14912}},
            {"depth": 10, "boundary": {"guard_boundary_chars": 22016}},
        ],
        "unrelated": "kept in the committed copy",
    },
}
