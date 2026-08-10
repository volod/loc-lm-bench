"""The pointer walk behind resolving a published number.

This file owns the ARITHMETIC -- how a field pointer addresses a row of a run aggregate. The
committed evidence and the two-source read live in `test_agentic_published_value_provenance.py`; the
committed crossovers are resolved against their own aggregates in
`test_agentic_memory_crossover_restatement_provenance.py`, so a failure here names the pointer
rather than the study that happened to use it.
"""

import pytest

from llb.bench.agentic_published_value_pointer import read_field

AGGREGATE: dict[str, object] = {
    "depth_surface": [
        {"depth": 6, "crossover_max_prompt_chars": 14159.929807575942, "bracket": [14000, 15500]},
        {"depth": 10, "crossover_max_prompt_chars": 21899.890064587056, "bracket": [20000, 23000]},
    ],
    "depth_ladders": [{"depth": 6, "boundary": {"guard_boundary_chars": 14912, "to_fold_step": 7}}],
    "cells": [
        {"cell_id": "surface-d10-g23000", "depth": 10, "measured_side": "cap_cheaper"},
        {"cell_id": "surface-d6-g14000", "depth": 6, "measured_side": "compact_cheaper"},
        {"cell_id": "collapse-d6-s0.4-g17500", "depth": 6, "measured_side": "compact_cheaper"},
    ],
    "cap_peak_prompt_chars": {"6": 8374, "10": 11926},
    "reading": "crossover_bracketed",
}


# --- the field pointer ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("reading", "crossover_bracketed"),
        ("cap_peak_prompt_chars.6", 8374),
        ("depth_surface[depth=10].crossover_max_prompt_chars", 21899.890064587056),
        ("depth_ladders[depth=6].boundary.guard_boundary_chars", 14912),
        ("cells[cell_id=surface-d10-g23000].measured_side", "cap_cheaper"),
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
        ("cells[cell_id=surface-missing].measured_side", "with cell_id=surface-missing"),
    ],
)
def test_a_pointer_that_addresses_nothing_names_the_segment_that_failed(field, match):
    """A refusal about the whole path leaves an operator diffing two long strings."""
    with pytest.raises(ValueError, match=match):
        read_field(AGGREGATE, field, where="test")


def test_a_row_selector_keeps_dots_that_belong_to_the_key_value():
    """Cell ids embed compact_share (`s0.4`); splitting on every dot would invent a broken segment."""
    assert (
        read_field(AGGREGATE, "cells[cell_id=collapse-d6-s0.4-g17500].cell_id", where="test")
        == "collapse-d6-s0.4-g17500"
    )


def test_the_row_selector_reads_a_list_and_never_a_dict_of_the_same_name():
    """`cap_peak_prompt_chars` is keyed by depth as a MAPPING, so the two forms cannot be confused."""
    assert read_field(AGGREGATE, "cap_peak_prompt_chars.10", where="test") == 11926
    with pytest.raises(ValueError, match="which the artifact does not carry"):
        read_field(AGGREGATE, "cap_peak_prompt_chars[depth=10].x", where="test")
