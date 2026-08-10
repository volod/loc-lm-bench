"""Every pair of auditable policy fields, and what the band arithmetic answers about it.

Two constants COUPLE when one decides what the other MEANS, and the compound policy-change
guarantee is only tested where a coupling actually separates the compound reading from the
per-field union. One pair does that today (`compact_share` x `summary_input_cap`), and a guarantee
resting on one pair through one coupling is a guarantee that retires with the next refactor: drop
the `trigger` bound -- already not the shipped default -- and the only counterexample goes with it.

So the question "is there a second pair?" is answered here for ALL of them rather than left open.
Each of the `C(6, 2) = 15` pairs of `AUDITABLE_FIELDS` states its mechanism and hands the solver its
own three conditions (`agentic_policy_change_interaction_conditions`), and the solver -- not this
file -- says whether a guard satisfies them:

  - `band`        -- a separating guard band exists and is solved per fold step. One pair.
  - `no_geometry` -- the fields DO couple, and the three conditions are contradictory at every fold
    step of every depth, so no guard separates them.
  - `independent` -- neither field's runtime meaning is a function of the other at all.

The `no_geometry` answers are arithmetic, not a scan; `agentic_policy_change_interaction_scan`
replays a grid of geometries as the independent check on them.
"""

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping

from llb.bench.agentic_policy_change_audit import AUDITABLE_FIELDS, PolicyChange
from llb.bench.agentic_policy_change_interaction_cap import observation_cap_conditions
from llb.bench.agentic_policy_change_interaction_conditions import (
    head_share_conditions,
    inert_field_conditions,
    keep_recent_conditions,
    share_bound_conditions,
)
from llb.bench.agentic_policy_change_interaction_terms import (
    FIELD_BOUND,
    FIELD_CAP,
    FIELD_HEAD,
    FIELD_KEEP_LAST,
    FIELD_KEEP_RECENT,
    FIELD_SHARE,
    ConditionsFn,
)

# What the band arithmetic answers about one pair.
ANSWER_BAND = "band"
ANSWER_NO_GEOMETRY = "no_geometry"
ANSWER_INDEPENDENT = "independent"

# One concrete move per field: the shipped value, and a neighbour a commit could plausibly ship.
# A pair's example change is the two fields' moves made together, which is how the scan asks the
# replay the same question the conditions answer by arithmetic -- and for `compact_share` x
# `summary_input_cap` it is exactly the change the committed interaction fixture is built around.
FIELD_MOVES: dict[str, tuple[Any, Any]] = {
    FIELD_CAP: (800, 1600),
    FIELD_HEAD: (0.6, 0.5),
    FIELD_KEEP_LAST: (3, 1),
    FIELD_SHARE: (0.5, 0.48),
    FIELD_KEEP_RECENT: (1, 2),
    FIELD_BOUND: ("window", "trigger"),
}

# Candidates the value sweep asks about, per field. The first entry IS the FIELD_MOVES neighbour so
# the single-move fixture and the grid stay one table; further entries cover the other direction (or
# another plausible ship) so a silent pair is silent across a value space, not one neighbour. Share
# 0.55 is the instructive second point: the band that opens for 0.5 -> 0.48 vanishes for 0.5 -> 0.55
# because only a lower share can elide where the baseline did not.
FIELD_CANDIDATE_GRID: dict[str, tuple[Any, ...]] = {
    FIELD_CAP: (1600, 400),
    FIELD_HEAD: (0.5, 0.7),
    FIELD_KEEP_LAST: (1, 5),
    FIELD_SHARE: (0.48, 0.55),
    FIELD_KEEP_RECENT: (2, 3),
    FIELD_BOUND: ("trigger",),
}

_MECHANISM_SHARE_BOUND = (
    "under the `trigger` bound the summarize call's input cap IS `compact_share * max_prompt_chars`,"
    " so the share moves the bound's own value"
)
_MECHANISM_CAP_SHARE = (
    "the cap decides every prompt size, and the prompt size is what crosses the share's trigger"
)
_MECHANISM_CAP_BOUND = (
    "the cap decides which step folds, and the fold step decides what the bound is asked to fit"
)
_MECHANISM_CAP_KEEP_RECENT = (
    "the cap decides which step folds, and the keep decides what that fold leaves live"
)
_MECHANISM_CAP_HEAD = (
    "the head share splits the span the cap trims away, so the cap decides whether the head share"
    " means anything at all"
)
_MECHANISM_SHARE_KEEP_RECENT = (
    "the share decides which step folds, and the keep decides how much of the transcript that fold"
    " leaves live behind it"
)
_MECHANISM_KEEP_RECENT_BOUND = (
    "the keep decides how large a transcript the fold offers, and the bound decides whether that"
    " transcript is elided"
)
_MECHANISM_HEAD = (
    "the head share moves bytes inside a trimmed observation and never its length, which is the only"
    " thing another field could read"
)
_MECHANISM_KEEP_LAST = (
    "`keep_last_n` parameterizes neither audited arm (`observation_cap`, `compact`), so no other"
    " field's meaning can depend on it"
)


@dataclass(frozen=True, slots=True)
class Coupling:
    """One pair of auditable fields: how they couple, and the conditions a separating guard needs."""

    fields: tuple[str, str]
    answer: str
    mechanism: str
    conditions: ConditionsFn

    @property
    def separates(self) -> bool:
        return self.answer == ANSWER_BAND

    @property
    def label(self) -> str:
        return f"{self.fields[0]} x {self.fields[1]}"

    def describe(self) -> str:
        return f"{self.label} [{self.answer}]: {self.mechanism}"

    def example_change(self) -> PolicyChange:
        """The concrete two-field change this pair is asked about, from `FIELD_MOVES`."""
        return PolicyChange(
            baseline={field: FIELD_MOVES[field][0] for field in self.fields},
            candidate={field: FIELD_MOVES[field][1] for field in self.fields},
        )

    def candidate_changes(self) -> list[PolicyChange]:
        """Every baseline-to-candidate change the value sweep asks about for this pair.

        Baselines stay the shipped values in `FIELD_MOVES`; candidates are the cartesian product of
        `FIELD_CANDIDATE_GRID` for the two fields. A one-field no-op (candidate equals baseline) is
        dropped so the scan never asks about a change that does not move both fields.
        """
        baselines = {field: FIELD_MOVES[field][0] for field in self.fields}
        grids = [FIELD_CANDIDATE_GRID[field] for field in self.fields]
        changes: list[PolicyChange] = []
        for combo in product(*grids):
            candidate = dict(zip(self.fields, combo, strict=True))
            if any(candidate[field] == baselines[field] for field in self.fields):
                continue
            changes.append(PolicyChange(baseline=baselines, candidate=candidate))
        return changes


def _index(couplings: list[Coupling]) -> dict[tuple[str, ...], Coupling]:
    """Key every pair by its fields in `AUDITABLE_FIELDS` order -- the order `PolicyChange` reports."""
    return {coupling.fields: coupling for coupling in couplings}


COUPLINGS: dict[tuple[str, ...], Coupling] = _index(
    [
        Coupling(
            (FIELD_SHARE, FIELD_BOUND), ANSWER_BAND, _MECHANISM_SHARE_BOUND, share_bound_conditions
        ),
        Coupling(
            (FIELD_CAP, FIELD_SHARE),
            ANSWER_NO_GEOMETRY,
            _MECHANISM_CAP_SHARE,
            observation_cap_conditions,
        ),
        Coupling(
            (FIELD_CAP, FIELD_BOUND),
            ANSWER_NO_GEOMETRY,
            _MECHANISM_CAP_BOUND,
            observation_cap_conditions,
        ),
        Coupling(
            (FIELD_CAP, FIELD_KEEP_RECENT),
            ANSWER_NO_GEOMETRY,
            _MECHANISM_CAP_KEEP_RECENT,
            keep_recent_conditions,
        ),
        Coupling(
            (FIELD_CAP, FIELD_HEAD), ANSWER_NO_GEOMETRY, _MECHANISM_CAP_HEAD, head_share_conditions
        ),
        Coupling(
            (FIELD_SHARE, FIELD_KEEP_RECENT),
            ANSWER_NO_GEOMETRY,
            _MECHANISM_SHARE_KEEP_RECENT,
            keep_recent_conditions,
        ),
        Coupling(
            (FIELD_KEEP_RECENT, FIELD_BOUND),
            ANSWER_NO_GEOMETRY,
            _MECHANISM_KEEP_RECENT_BOUND,
            keep_recent_conditions,
        ),
        *(
            Coupling(pair, ANSWER_INDEPENDENT, _MECHANISM_HEAD, head_share_conditions)
            for pair in (
                (FIELD_HEAD, FIELD_SHARE),
                (FIELD_HEAD, FIELD_KEEP_RECENT),
                (FIELD_HEAD, FIELD_BOUND),
            )
        ),
        *(
            Coupling(pair, ANSWER_INDEPENDENT, _MECHANISM_KEEP_LAST, inert_field_conditions)
            for pair in (
                (FIELD_CAP, FIELD_KEEP_LAST),
                (FIELD_HEAD, FIELD_KEEP_LAST),
                (FIELD_KEEP_LAST, FIELD_SHARE),
                (FIELD_KEEP_LAST, FIELD_KEEP_RECENT),
                (FIELD_KEEP_LAST, FIELD_BOUND),
            )
        ),
    ]
)


def coupling_for(fields: tuple[str, ...]) -> Coupling:
    """The coupling a two-field change is about, or a refusal naming what the solver can answer."""
    if fields not in COUPLINGS:
        raise ValueError(
            f"a separating band is only defined for a PAIR of auditable policy fields in "
            f"{AUDITABLE_FIELDS} order, got {fields}"
        )
    return COUPLINGS[fields]


def separating_couplings() -> list[Coupling]:
    """Every pair that DOES separate the two readings -- the set the compound guarantee rests on."""
    return [coupling for coupling in COUPLINGS.values() if coupling.separates]


def held_baseline(coupling: Coupling, held: Mapping[str, Any]) -> dict[str, Any]:
    """`held_fixed` restated at this coupling's baseline, so the per-field union audits that policy.

    The per-field reading replays each field alone with the others taken from the cell, the design's
    `held_fixed`, and the shipped defaults. That is only the change's baseline policy when those
    sources AGREE with the baseline -- otherwise the union is a reading of a configuration the change
    never names, and a separation would say nothing about the compound guarantee.
    """
    change = coupling.example_change()
    return {
        **dict(held),
        **{field: change.baseline[field] for field in coupling.fields if field in held},
    }
