"""The pointer walk and the two-source read behind resolving a published number.

This file owns the ARITHMETIC -- how a field pointer addresses a row of a run aggregate, how a
committed slice is cut out of one, and what the resolver does when the host still has the artifact
the slice was cut from. Nothing here loads a study design; the committed crossovers are resolved
against their own aggregates in `test_agentic_memory_crossover_restatement_provenance.py`, so a
failure here names the pointer rather than the study that happened to use it.
"""

import json
from pathlib import Path

import pytest

from llb.bench.agentic_published_value_provenance import (
    FIXTURE_SCHEMA_VERSION,
    PROVENANCE_FIXTURE,
    PublishedValueResolver,
    load_provenance_fixture,
    merge_field_slice,
    provenance_pair,
    read_field,
    write_provenance_fixture,
)

ARTIFACT = "study/run/analysis.json"
AGGREGATE: dict[str, object] = {
    "depth_surface": [
        {"depth": 6, "crossover_max_prompt_chars": 14159.929807575942, "bracket": [14000, 15500]},
        {"depth": 10, "crossover_max_prompt_chars": 21899.890064587056, "bracket": [20000, 23000]},
    ],
    "depth_ladders": [{"depth": 6, "boundary": {"guard_boundary_chars": 14912, "to_fold_step": 7}}],
    "cap_peak_prompt_chars": {"6": 8374, "10": 11926},
    "reading": "crossover_bracketed",
}


def _host(tmp_path: Path, aggregates: dict[str, dict[str, object]]) -> Path:
    """A project root carrying just the committed provenance fixture."""
    (tmp_path / PROVENANCE_FIXTURE.parent).mkdir(parents=True, exist_ok=True)
    write_provenance_fixture(tmp_path, aggregates)
    return tmp_path


def _slice(*fields: str) -> dict[str, object]:
    cut: dict[str, object] = {}
    for field in fields:
        merge_field_slice(cut, AGGREGATE, field)
    return cut


# --- the field pointer ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("reading", "crossover_bracketed"),
        ("cap_peak_prompt_chars.6", 8374),
        ("depth_surface[depth=10].crossover_max_prompt_chars", 21899.890064587056),
        ("depth_ladders[depth=6].boundary.guard_boundary_chars", 14912),
    ],
)
def test_a_field_pointer_addresses_a_dotted_path_a_row_and_a_nested_field(field, expected):
    """Every shape the agentic aggregates key their per-depth rows by is reachable."""
    assert read_field(AGGREGATE, field, where="test") == expected


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("missing", "reaches no 'missing'"),
        ("depth_surface[depth=7].crossover_max_prompt_chars", "with depth=7"),
        ("cap_peak_prompt_chars[depth=6].x", "with depth=6"),
        ("depth_surface[depth=6].absent", "reaches no 'absent'"),
    ],
)
def test_a_pointer_that_addresses_nothing_names_the_segment_that_failed(field, match):
    """A refusal about the whole path leaves an operator diffing two long strings."""
    with pytest.raises(ValueError, match=match):
        read_field(AGGREGATE, field, where="test")


def test_the_row_selector_reads_a_list_and_never_a_dict_of_the_same_name():
    """`cap_peak_prompt_chars` is keyed by depth as a MAPPING, so the two forms cannot be confused."""
    assert read_field(AGGREGATE, "cap_peak_prompt_chars.10", where="test") == 11926
    with pytest.raises(ValueError, match="which the artifact does not carry"):
        read_field(AGGREGATE, "cap_peak_prompt_chars[depth=10].x", where="test")


# --- the committed slice ----------------------------------------------------------------------


def test_a_slice_keeps_the_artifact_s_shape_so_one_walk_reads_both():
    """The slice is read by the SAME pointer as the artifact, which is why it is nested, not flat."""
    field = "depth_surface[depth=6].crossover_max_prompt_chars"
    cut = _slice(field)
    assert cut == {
        "depth_surface": [{"depth": 6, "crossover_max_prompt_chars": 14159.929807575942}]
    }
    assert read_field(cut, field, where="test") == read_field(AGGREGATE, field, where="test")


def test_two_pointers_into_one_list_merge_into_one_slice():
    """Two depths of the same aggregate are two rows of one list, not two copies of the list."""
    cut = _slice(
        "depth_surface[depth=6].crossover_max_prompt_chars",
        "depth_surface[depth=10].bracket",
        "depth_ladders[depth=6].boundary.guard_boundary_chars",
    )
    assert [row["depth"] for row in cut["depth_surface"]] == [6, 10]
    assert cut["depth_surface"][1] == {"depth": 10, "bracket": [20000, 23000]}
    assert cut["depth_ladders"] == [{"depth": 6, "boundary": {"guard_boundary_chars": 14912}}]


def test_a_pointer_ending_on_a_row_selector_cuts_no_slice():
    """A selector picks a row; the value being resolved is a FIELD of it, so the path must go on."""
    with pytest.raises(ValueError, match="ends on a row selector"):
        _slice("depth_surface[depth=6]")


def test_the_fixture_round_trips_and_declares_its_schema(tmp_path):
    root = _host(tmp_path, {ARTIFACT: _slice("cap_peak_prompt_chars.6")})
    payload = json.loads((root / PROVENANCE_FIXTURE).read_text(encoding="utf-8"))
    assert payload["schema_version"] == FIXTURE_SCHEMA_VERSION
    assert load_provenance_fixture(root) == {ARTIFACT: {"cap_peak_prompt_chars": {"6": 8374}}}


def test_a_missing_or_mis_versioned_fixture_is_refused(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        load_provenance_fixture(tmp_path)
    root = _host(tmp_path, {})
    (root / PROVENANCE_FIXTURE).write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version 1"):
        load_provenance_fixture(root)


# --- the two-source read ----------------------------------------------------------------------


def test_a_value_resolves_from_the_committed_slice_with_no_run_on_the_host(tmp_path):
    """CI is the case with no DATA_DIR at all, so the slice has to be sufficient on its own."""
    root = _host(tmp_path, {ARTIFACT: _slice("depth_surface[depth=6].crossover_max_prompt_chars")})
    resolver = PublishedValueResolver(root=root)
    pair = {"artifact": ARTIFACT, "field": "depth_surface[depth=6].crossover_max_prompt_chars"}
    assert resolver.resolve(pair, where="test") == 14159.929807575942


def test_the_run_artifact_is_read_as_a_check_on_the_slice_and_a_drift_is_refused(tmp_path):
    """The slice is generated, so the only way it disagrees with its artifact is by going stale."""
    field = "depth_surface[depth=6].crossover_max_prompt_chars"
    root = _host(tmp_path, {ARTIFACT: _slice(field)})
    data_dir = tmp_path / "data"
    (data_dir / ARTIFACT).parent.mkdir(parents=True)
    (data_dir / ARTIFACT).write_text(json.dumps(AGGREGATE), encoding="utf-8")
    pair = {"artifact": ARTIFACT, "field": field}
    assert PublishedValueResolver(root=root, data_dir=data_dir).resolve(pair, where="t") == (
        14159.929807575942
    )

    moved = json.loads(json.dumps(AGGREGATE))
    moved["depth_surface"][0]["crossover_max_prompt_chars"] = 14161.0
    (data_dir / ARTIFACT).write_text(json.dumps(moved), encoding="utf-8")
    with pytest.raises(ValueError, match="the slice is stale"):
        PublishedValueResolver(root=root, data_dir=data_dir).resolve(pair, where="t")


def test_a_host_without_the_run_still_validates_rather_than_skipping(tmp_path):
    """A missing artifact is an ordinary host, not a licence to stop checking the published value."""
    field = "cap_peak_prompt_chars.6"
    root = _host(tmp_path, {ARTIFACT: _slice(field)})
    resolver = PublishedValueResolver(root=root, data_dir=tmp_path / "empty")
    assert resolver.resolve({"artifact": ARTIFACT, "field": field}, where="t") == 8374


def test_an_artifact_with_no_committed_slice_is_refused_as_unresolvable(tmp_path):
    """The evidence was garbage-collected or never committed; either way nothing resolves it."""
    root = _host(tmp_path, {ARTIFACT: _slice("reading")})
    with pytest.raises(ValueError, match="no committed slice of 'other/run.json'"):
        PublishedValueResolver(root=root).resolve(
            {"artifact": "other/run.json", "field": "reading"}, where="t"
        )


def test_a_field_that_resolves_to_something_other_than_a_number_is_refused(tmp_path):
    """A published value is a number; a pointer landing on a verdict string is a mis-aimed pointer."""
    root = _host(tmp_path, {ARTIFACT: _slice("reading")})
    with pytest.raises(ValueError, match="which is not a number"):
        PublishedValueResolver(root=root).resolve(
            {"artifact": ARTIFACT, "field": "reading"}, where="t"
        )


@pytest.mark.parametrize(
    ("provenance", "match"),
    [
        (None, "must carry a `provenance` object"),
        ({"field": "reading"}, "DATA_DIR-relative run artifact"),
        ({"artifact": "/abs/run.json", "field": "reading"}, "must stay inside DATA_DIR"),
        ({"artifact": "../run.json", "field": "reading"}, "must stay inside DATA_DIR"),
        ({"artifact": ARTIFACT}, "must be the field pointer"),
    ],
)
def test_a_provenance_pair_that_names_no_artifact_or_no_field_is_refused(provenance, match):
    with pytest.raises(ValueError, match=match):
        provenance_pair(provenance, where="test")
