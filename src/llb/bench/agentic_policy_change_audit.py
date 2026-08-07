"""Which published agent evidence a context-policy change invalidates -- decided with no model.

Changing one policy constant silently invalidates an unknown share of the repo's published agentic
numbers, and the honest answer without an audit is "re-run everything" or "hope". Neither is needed,
because a context policy is a PURE FUNCTION of the deterministic tool world: fix the geometry and the
controller, and the exact sequence of prompts an episode sends is decided before any model runs.

So the invariance test is the strongest one available and needs no interpretation: replay every
published cell under both values of the changed field with an ORACLE controller, record every prompt
each replay sends, and compare the sequences byte for byte. Identical sequences mean the published
cost cannot have moved; the first differing step says where the change bites.

Two properties make the replay legitimate as a statement about a REAL run:

  - the summarize call is answered with a FIXED summary, so the replay is deterministic. That only
    ever hides downstream divergence, never invents it: if the summarize prompts are identical then
    a temperature-0 model returns the same summary, so the later controller prompts are identical
    too. "All prompts identical under the replay" therefore implies "all prompts identical under the
    served model" -- the direction the invariance claim needs.
  - both arms of a published cell are replayed (`observation_cap` and `compact`), because a cell's
    published number is a compact-minus-cap delta and a change that moves either arm moves it.

A change is a set of FIELDS, not one field. A commit that re-pins two constants moves them together,
so auditing each on its own would replay two configurations that never shipped and report two re-run
scopes for one act; `PolicyChange` carries every moved field and the audit answers with one verdict.

This module owns the geometry extraction shared with the summarize-bound audit
(`agentic_memory_cap_audit`), which is one USE of this mechanism rather than a second one.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from llb.bench.agentic_policy_change_replay import AUDITED_POLICIES, arm_comparison

# How each committed study nests its cells under its own design root.
KIND_SURFACE = "compact_memory_boundary_surface"
KIND_COLLAPSE = "compact_trigger_guard_collapse"
KIND_FOLD_STEP = "compact_fold_step_crossover"
AUDITED_KINDS = (KIND_SURFACE, KIND_COLLAPSE, KIND_FOLD_STEP)
# A FIXTURE kind, not a published study: the interaction geometry that separates the compound
# verdict from the per-field one (`agentic_policy_change_interaction`). It states its cells flat at
# the design root and publishes no number, so it is readable as geometry but is deliberately absent
# from `AUDITED_DESIGN_PATHS` -- nothing it declares is evidence a constant change can invalidate.
KIND_INTERACTION = "policy_change_interaction"
GEOMETRY_KINDS = (*AUDITED_KINDS, KIND_INTERACTION)

# The committed studies whose published numbers an agent policy change can invalidate. One registry,
# because the CLI audit and the CI pin gate must never walk different evidence.
AUDITED_DESIGN_PATHS: dict[str, str] = {
    KIND_SURFACE: "samples/benchmarks/agentic_compact_memory_boundary_surface_design.json",
    KIND_COLLAPSE: "samples/benchmarks/agentic_compact_trigger_guard_collapse_design.json",
    KIND_FOLD_STEP: "samples/benchmarks/agentic_compact_fold_step_crossover_design.json",
}

VERDICT_INVARIANT = "prompt_invariant"
VERDICT_CHANGED = "prompts_change"
# A cell that declares the audited field ITSELF is not describable by the change: the field is that
# study's own axis there (the trigger collapse sweeps `compact_share` cell by cell), so replaying it
# at another value measures a different cell rather than the published one. A value inherited from
# `held_fixed` is not the same thing -- that is the study's inherited setting, and asking whether its
# number would hold at another value is exactly the counterfactual this audit answers.
VERDICT_NOT_APPLICABLE = "cell_pins_the_field"

# The `ContextPolicy` fields a published number can rest on, and how a CLI string becomes one.
POLICY_FIELD_TYPES: dict[str, type] = {
    "observation_cap_chars": int,
    "observation_head_share": float,
    "keep_last_n": int,
    "compact_share": float,
    "compact_keep_recent": int,
    "summary_input_cap": str,
}
AUDITABLE_FIELDS: tuple[str, ...] = tuple(POLICY_FIELD_TYPES)


def coerce_policy_value(field: str, value: str) -> Any:
    """Turn one CLI-supplied value into the type its policy field actually takes."""
    if field not in POLICY_FIELD_TYPES:
        raise ValueError(
            f"{field!r} is not an auditable policy field; choose from {AUDITABLE_FIELDS}"
        )
    return POLICY_FIELD_TYPES[field](value)


@dataclass(frozen=True, slots=True)
class PolicyChange:
    """The whole change being audited: every moved field, and the two policies it moves between.

    One change, not one per field. A commit that re-pins `observation_cap_chars` and
    `compact_keep_recent` together has ONE baseline (what the published cells were measured under)
    and ONE candidate (what the new build ships); replaying either field on its own would put a
    configuration that never existed on both sides of the comparison.
    """

    baseline: Mapping[str, Any]
    candidate: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.baseline:
            raise ValueError("a policy change must move at least one auditable field")
        if set(self.baseline) != set(self.candidate):
            raise ValueError(
                "a policy change must state a baseline and a candidate for the same fields, got "
                f"{sorted(self.baseline)} and {sorted(self.candidate)}"
            )
        # Over the stated keys, not over `fields`, which only knows the auditable ones.
        for field in sorted(self.baseline):
            if field not in POLICY_FIELD_TYPES:
                raise ValueError(
                    f"{field!r} is not an auditable policy field; choose from {AUDITABLE_FIELDS}"
                )
            if self.baseline[field] == self.candidate[field]:
                raise ValueError(
                    f"auditing {field!r} needs two different values, "
                    f"got {self.baseline[field]!r} twice"
                )

    @classmethod
    def of(cls, field: str, baseline: Any, candidate: Any) -> "PolicyChange":
        """The one-field change -- the common case, and what the CLI builds per `--field`."""
        return cls(baseline={field: baseline}, candidate={field: candidate})

    @property
    def fields(self) -> tuple[str, ...]:
        """Every moved field, in the order the shipped dataclass declares them."""
        return tuple(field for field in POLICY_FIELD_TYPES if field in self.baseline)

    @property
    def is_compound(self) -> bool:
        return len(self.baseline) > 1

    @property
    def label(self) -> str:
        return ", ".join(
            f"{field} {self.baseline[field]!r} -> {self.candidate[field]!r}"
            for field in self.fields
        )

    def without(self, fields: Collection[str]) -> "PolicyChange | None":
        """The same change minus the named fields, or None when nothing is left to audit."""
        kept = [field for field in self.fields if field not in set(fields)]
        if not kept:
            return None
        return PolicyChange(
            baseline={field: self.baseline[field] for field in kept},
            candidate={field: self.candidate[field] for field in kept},
        )


def audit_policy_change(
    designs: dict[str, dict[str, object]], change: PolicyChange
) -> dict[str, list[dict[str, object]]]:
    """Replay every published cell of every design under both policies and compare its prompts."""
    return {
        kind: [
            audit_cell_prompts(cell, _held_fixed(design, kind), change)
            for cell in declared_geometry(design, kind)
        ]
        for kind, design in designs.items()
    }


def audit_cell_prompts(
    cell: dict[str, object],
    held: dict[str, object],
    change: PolicyChange,
) -> dict[str, object]:
    """One cell's verdict: do both arms send the identical prompts under both policies?

    A cell that declares one of the moved fields itself keeps its own value for that field and is
    audited on the rest of the change; a cell that declares ALL of them is not described by the
    change at all.
    """
    pinned = [field for field in change.fields if field in cast(list[str], cell["pinned_fields"])]
    described = change.without(pinned)
    row = {
        **cell,
        "policy_fields": list(change.fields),
        "baseline": dict(change.baseline),
        "candidate": dict(change.candidate),
        "not_applicable_fields": pinned,
    }
    if described is None:
        return {
            **row,
            "arms": {},
            "changed_arms": [],
            "refused_prompt_only": False,
            "refused_prompt_bytes_only": False,
            "candidate_overflows": [],
            "first_divergent_step": None,
            "verdict": VERDICT_NOT_APPLICABLE,
        }
    arms = {
        name: arm_comparison(name, cell, held, described.baseline, described.candidate)
        for name in AUDITED_POLICIES
    }
    changed = sorted(name for name, arm in arms.items() if not arm["identical"])
    return {
        **row,
        "arms": arms,
        "changed_arms": changed,
        # The change moved only the prompt the guard REFUSED: every arm sent the same bytes and
        # ended on a differently sized overflow. Nothing a model saw moved, and the published
        # number still did, because the episode ended on the prompt that moved.
        "refused_prompt_only": bool(changed)
        and all(arms[name]["sent_identical"] for name in changed),
        # The refusal the two sides priced identically and still wrote differently: invisible to
        # anything that compares the refused prompt by size.
        "refused_prompt_bytes_only": any(
            arms[name]["refused_prompt_moved_bytes_only"] for name in changed
        ),
        # Stronger than "the prompts move": under the candidate the cell no longer FITS its
        # published guard, so re-running it needs a new guard rather than a new measurement.
        "candidate_overflows": sorted(
            name
            for name, arm in arms.items()
            if arm["candidate_refused_tasks"] and not arm["baseline_refused_tasks"]
        ),
        "first_divergent_step": min(
            (
                cast(int, arm["first_divergent_step"])
                for arm in arms.values()
                if arm["first_divergent_step"] is not None
            ),
            default=None,
        ),
        "verdict": VERDICT_CHANGED if changed else VERDICT_INVARIANT,
    }


def declared_geometry(design: dict[str, object], study_kind: str) -> list[dict[str, object]]:
    """Every declared cell's `(cell_id, depth, compact_share, max_prompt_chars)`, in design order.

    The three studies nest cells differently -- a flat grid, families, and per-depth step ladders --
    so the shape is read per kind rather than guessed at.
    """
    held = _held_fixed(design, study_kind)
    # The collapse study SWEEPS the share, so it states one per cell and holds none fixed.
    default_share = held.get("compact_share")
    # `(depth, group)`: the surface states depth per cell, the others state it on the group.
    groups: list[tuple[int | None, dict[str, object]]]
    if study_kind == KIND_SURFACE:
        groups = [(None, cast(dict[str, object], design["surface"]))]
    elif study_kind == KIND_INTERACTION:
        groups = [(None, design)]
    elif study_kind == KIND_COLLAPSE:
        groups = [
            (int(cast(int, family["depth"])), family)
            for family in cast(list[dict[str, object]], design["families"])
        ]
    else:
        groups = [
            (int(cast(int, ladder["depth"])), step)
            for ladder in cast(list[dict[str, object]], design["ladders"])
            for step in cast(list[dict[str, object]], ladder["steps"])
        ]
    return [
        {
            "cell_id": cast(str, cell["cell_id"]),
            "depth": int(cast(int, cell.get("depth", depth))),
            "compact_share": _share(cell, default_share),
            "max_prompt_chars": int(cast(int, cell["max_prompt_chars"])),
            "pinned_fields": [name for name in POLICY_FIELD_TYPES if name in cell],
        }
        for depth, group in groups
        for cell in cast(list[dict[str, object]], group["cells"])
    ]


def load_audited_design(path: Path | str) -> dict[str, object]:
    """Load one committed study design for auditing (the studies' own strict JSON loader)."""
    from llb.bench.agentic_memory_transfer import load_transfer_design

    return load_transfer_design(path)


def load_audited_designs() -> dict[str, dict[str, object]]:
    """Every committed study the audit walks, keyed by its study kind."""
    from llb.core.paths import PROJECT_ROOT

    return {
        kind: load_audited_design(PROJECT_ROOT / path)
        for kind, path in AUDITED_DESIGN_PATHS.items()
    }


def _share(cell: dict[str, object], default: object) -> float:
    share = cell.get("compact_share", default)
    if share is None:
        raise ValueError(f"cell {cell.get('cell_id')!r} states no compact_share and none is held")
    return float(cast(float, share))


def _held_fixed(design: dict[str, object], study_kind: str) -> dict[str, object]:
    if study_kind not in GEOMETRY_KINDS:
        raise ValueError(
            f"{study_kind!r} is not a readable geometry kind; choose from {GEOMETRY_KINDS}"
        )
    return cast(dict[str, object], design["held_fixed"])
