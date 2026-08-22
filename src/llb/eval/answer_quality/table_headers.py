"""The table-header dimension of the answer-quality lane (table-header-context-restoration).

Prompt-side header restoration is not a retrieval knob: it changes what the model READS and, by
construction, nothing the retrieval metrics can see. So it cannot be diagnosed by a retrieval
sweep at all -- the only reading that exists is the same retrieval row scored twice, with the step
off and on, over one item set.

That is exactly the shape the answer-quality lane already compares, so the step rides in the lane
LABEL (`vector+headers`) like the retrieval budget does (`llb.eval.answer_quality.budgets`): a cell
stays a plain lane everywhere downstream -- one run bundle, one row in every table, one entry in
the item ledger -- and the label still parses back into the config that produced it.
"""

from llb.eval.answer_quality.models import LaneSpec

# Suffix marking the lane whose prompt restores table headers. `+` cannot occur in a sweep row
# label (`vector`, `graph/<strategy>`, `fused/<strategy>@<weight>/d<depth>/i<identity>/r<ratio>`),
# and it matches the `<row>+rerank` twin `compare-retrieval` already prints for a knob that is not
# part of the row itself.
HEADER_MARKER = "+headers"


def header_label(row: str) -> str:
    """`vector` -> `vector+headers`."""
    return f"{row}{HEADER_MARKER}"


def split_header_label(label: str) -> tuple[str, bool]:
    """`vector+headers` -> `("vector", True)`; a label without the suffix keeps False."""
    if label.endswith(HEADER_MARKER):
        row = label[: -len(HEADER_MARKER)]
        if not row:
            raise ValueError(f"lane label {label!r} is only the {HEADER_MARKER!r} suffix")
        return row, True
    return label, False


def header_lanes(lanes: list[LaneSpec]) -> list[LaneSpec]:
    """Every lane twinned with its header-restoring copy, row-major so the pair sits together.

    `lanes[0]` with the step OFF stays first, so the comparison baseline remains the shipped
    prompt and every restored cell is read against it.
    """
    return [
        twin
        for lane in lanes
        for twin in (
            lane,
            lane._replace(label=header_label(lane.label), restore_table_headers=True),
        )
    ]


__all__ = ["HEADER_MARKER", "header_label", "header_lanes", "split_header_label"]
