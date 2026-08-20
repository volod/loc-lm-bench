"""Answering a coupling by REPLAY -- the independent check on what the band arithmetic derived.

The coupling enumeration says only `compact_share` x `summary_input_cap` separates the compound
policy-change reading from the per-field union, and it says so from three inequalities per pair. An
inequality can be wrong about the loop it claims to describe, so the same question is asked the
other way here: replay a grid of geometries, read both ways, and report every cell where they
actually disagree. The arithmetic and the scan must agree -- inside a solved band and nowhere else.

The scan is deterministic and modelless (an oracle controller and a fixed summary reply), so its
answer is a fact about the shipped loop rather than a sample of it. It is also the expensive
direction: one cell costs three audits (the compound plus one per field), so a wide grid belongs in
a `slow` test or a one-off run, and `make ci` scans a few guards per pair. The value sweep asks the
same grid at every candidate in `FIELD_CANDIDATE_GRID`, so a silent pair is silent across the values
a commit could ship rather than only the neighbour `FIELD_MOVES` names.
"""

from typing import Any, Mapping, Sequence, cast

from llb.bench.policy_change.audit import PolicyChange
from llb.bench.policy_change.interaction.run import (
    READING_COMPOUND,
    READING_PER_FIELD,
    audit_interaction_cell,
)
from llb.bench.policy_change.interaction.couplings import Coupling, held_baseline
from llb.bench.policy_change.interaction.terms import (
    FIELD_CAP,
    FIELD_HEAD,
    FIELD_SHARE,
    shipped_policy_value,
)

# What `agentic_policy_change_replay._policy` reads off the cell / `held_fixed` rather than off the
# shipped dataclass. Every other field falls back to the shipped default on BOTH arms.
HELD_POLICY_FIELDS = (FIELD_CAP, FIELD_HEAD, FIELD_SHARE)


def scan_cell(
    change: PolicyChange, *, depth: int, guard: int, held: Mapping[str, Any]
) -> dict[str, object]:
    """Read one (depth, guard) both ways, as a cell the interaction audit accepts."""
    cell = {
        "cell_id": f"scan-d{depth}-g{guard}",
        "depth": depth,
        "compact_share": held[FIELD_SHARE],
        "max_prompt_chars": guard,
        "pinned_fields": [],
    }
    return audit_interaction_cell(cell, dict(held), change)


def scan_separating_cells(
    change: PolicyChange,
    *,
    depths: list[int],
    guards: list[int],
    held: Mapping[str, Any],
) -> list[dict[str, object]]:
    """Every geometry in the grid where the two readings disagree -- usually none."""
    check_baseline_is_the_replayed_one(change, held)
    return [
        row
        for depth in depths
        for guard in guards
        if (row := scan_cell(change, depth=depth, guard=guard, held=held))["separates"]
    ]


def scan_couplings_across_candidate_values(
    couplings: Sequence[Coupling],
    *,
    depths: list[int],
    guards: list[int],
    held: Mapping[str, Any],
) -> dict[str, object]:
    """Replay every coupling at every candidate-value change; record which pairs stay silent.

    A pair is silent across the grid when no (candidate combo, depth, guard) separates. Hits name
    the candidate values that did separate, so a band pair that opens at one share and vanishes at
    another is visible as a hit list rather than as a false "always separates".
    """
    silent_pairs: list[tuple[str, ...]] = []
    hits: list[dict[str, object]] = []
    for coupling in couplings:
        geometry = held_baseline(coupling, held)
        pair_hit = False
        for change in coupling.candidate_changes():
            rows = scan_separating_cells(change, depths=depths, guards=guards, held=geometry)
            if not rows:
                continue
            pair_hit = True
            hits.append(
                {
                    "fields": coupling.fields,
                    "label": coupling.label,
                    "candidate": dict(change.candidate),
                    "change": change,
                    "rows": rows,
                }
            )
        if not pair_hit:
            silent_pairs.append(coupling.fields)
    return {"silent_pairs": silent_pairs, "hits": hits}


def check_baseline_is_the_replayed_one(change: PolicyChange, held: Mapping[str, Any]) -> None:
    """Refuse a scan whose per-field arm would replay a policy the change does not name.

    The per-field reading audits each field alone and takes the others from the cell, `held_fixed`,
    and the shipped defaults. Those sources are the change's BASELINE only when they agree with it;
    if they do not, the union is a reading of a configuration nobody shipped and nobody published,
    and its agreement or disagreement with the compound says nothing about the guarantee.
    """
    for field in change.fields:
        replayed = held[field] if field in HELD_POLICY_FIELDS else shipped_policy_value(field)
        if replayed != change.baseline[field]:
            raise ValueError(
                f"the per-field arm would replay {field}={replayed!r} where the change baselines it "
                f"at {change.baseline[field]!r}; restate held_fixed at the baseline first"
            )


def format_scan_report(change: PolicyChange, rows: list[dict[str, object]]) -> str:
    """What the scan found, in the shape a failing assertion can print."""
    header = f"[scan] {change.label}"
    if not rows:
        return f"{header}\n  no geometry in the grid separates the two readings"
    return "\n".join(
        [
            header,
            *(
                f"  {row['cell_id']}: separates on {row['separates_on']}"
                f" (compound {_verdict(row, READING_COMPOUND)},"
                f" union {_verdict(row, READING_PER_FIELD)})"
                for row in rows
            ),
        ]
    )


def format_value_sweep_report(result: Mapping[str, object]) -> str:
    """Silent pairs and every candidate that separated, for a failing value-sweep assertion."""
    silent = cast(list[tuple[str, ...]], result["silent_pairs"])
    hits = cast(list[dict[str, object]], result["hits"])
    lines = [
        f"[value-sweep] {len(silent)} pair(s) silent across the candidate grid",
        *(f"  silent: {' x '.join(fields)}" for fields in silent),
    ]
    if not hits:
        lines.append("  no separating candidate in the grid")
        return "\n".join(lines)
    lines.append(f"  {len(hits)} separating candidate(s):")
    for hit in hits:
        candidate = cast(dict[str, object], hit["candidate"])
        named = ", ".join(f"{field}={value!r}" for field, value in candidate.items())
        rows = cast(list[dict[str, object]], hit["rows"])
        lines.append(f"  {hit['label']} @ {named}: {len(rows)} cell(s)")
        lines.extend(f"    {row['cell_id']}: separates on {row['separates_on']}" for row in rows)
    return "\n".join(lines)


def _verdict(row: dict[str, object], reading: str) -> str:
    """One reading's verdict out of a scanned row, for the report line."""
    return cast(dict[str, str], row[reading])["verdict"]
