"""The pointer walk behind resolving a published number, and the slice it cuts.

This file owns the ARITHMETIC -- how a field pointer addresses a row of a run aggregate and how a
committed slice is cut out of one. The two-source read and the artifact pin live in
`test_agentic_published_value_provenance.py`; the committed crossovers are resolved against their
own aggregates in `test_agentic_memory_crossover_restatement_provenance.py`, so a failure here names
the pointer rather than the study that happened to use it.
"""

import pytest

from llb.bench.agentic_published_value_pointer import merge_field_slice, read_field

AGGREGATE: dict[str, object] = {
    "depth_surface": [
        {"depth": 6, "crossover_max_prompt_chars": 14159.929807575942, "bracket": [14000, 15500]},
        {"depth": 10, "crossover_max_prompt_chars": 21899.890064587056, "bracket": [20000, 23000]},
    ],
    "depth_ladders": [{"depth": 6, "boundary": {"guard_boundary_chars": 14912, "to_fold_step": 7}}],
    "cap_peak_prompt_chars": {"6": 8374, "10": 11926},
    "reading": "crossover_bracketed",
}


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
