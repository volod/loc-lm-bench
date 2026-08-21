"""Which published agent evidence a context-policy change invalidates -- decided with no model.

Changing one policy constant silently invalidates an unknown share of the repo's published agentic
numbers, and the honest answer without an audit is "re-run everything" or "hope". Neither is needed,
because a context policy is a PURE FUNCTION of the deterministic tool world: fix the geometry and the
controller, and the exact sequence of prompts an episode sends is decided before any model runs.

So the invariance test is the strongest one available and needs no interpretation: replay every
published cell under both values of the changed field with an ORACLE controller, record every prompt
each replay sends, and compare the sequences byte for byte. Identical sequences mean the published
cost cannot have moved; the first differing step says where the change bites.

The registry covers the three cap-fitting memory studies plus the constant-sweep, keep-long, and
harness-comparison lanes that rest on the same constants -- each with its own geometry reader and
task builder -- so an "invalidates nothing" verdict is a statement about the whole agentic evidence
base, not one family of studies.

Two properties make the replay legitimate as a statement about a REAL run:

  - the summarize call is answered with a FIXED summary, so the replay is deterministic. That only
    ever hides downstream divergence, never invents it: if the summarize prompts are identical then
    a temperature-0 model returns the same summary, so the later controller prompts are identical
    too. "All prompts identical under the replay" therefore implies "all prompts identical under the
    served model" -- the direction the invariance claim needs.
  - both arms of a published cell are replayed when the cell's number is a compact-minus-cap delta;
    other lanes declare the policy arm(s) they were measured under.

A change is a set of FIELDS, not one field. A commit that re-pins two constants moves them together,
so auditing each on its own would replay two configurations that never shipped and report two re-run
scopes for one act; `PolicyChange` carries every moved field and the audit answers with one verdict.

This module owns the geometry extraction shared with the summarize-bound audit
(`agentic_memory_cap_audit`), which is one USE of this mechanism rather than a second one.
"""

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any, cast

from llb.bench.agentic.model import AgenticTask
from llb.bench.policy_change.replay import AUDITED_POLICIES, arm_comparison
from llb.bench.common import LLMComplete

# How a caller names the walk a replay compares over: given one task and the study's held geometry,
# return the controller that plays it. `replay_controller_for` is the default implementation.
ReplayController = Callable[[AgenticTask, dict[str, object]], LLMComplete]

# How each committed study nests its cells under its own design root.
KIND_SURFACE = "compact_memory_boundary_surface"
KIND_COLLAPSE = "compact_trigger_guard_collapse"
KIND_FOLD_STEP = "compact_fold_step_crossover"
# Non-cap-fitting lanes that rest on the same ContextPolicy constants: the constant sweep, the
# keep-long transcript lane, and the harness-comparison rows. Each needs its own geometry reader
# and a task builder other than the memory-chain one (see `agentic_policy_change_tasks`).
KIND_CONSTANT_SWEEP = "context_policy_constant_sweep"
KIND_KEEP_LONG = "context_policy_keep_long"
KIND_HARNESS = "agentic_harness_comparison"
CAP_FITTING_KINDS = (KIND_SURFACE, KIND_COLLAPSE, KIND_FOLD_STEP)
AUDITED_KINDS = (
    *CAP_FITTING_KINDS,
    KIND_CONSTANT_SWEEP,
    KIND_KEEP_LONG,
    KIND_HARNESS,
)
# A FIXTURE kind, not a published study: the interaction geometry that separates the compound
# verdict from the per-field one (`agentic_policy_change_interaction`). It states its cells flat at
# the design root and publishes no number, so it is readable as geometry but is deliberately absent
# from `AUDITED_DESIGN_PATHS` -- nothing it declares is evidence a constant change can invalidate.
KIND_INTERACTION = "policy_change_interaction"
# The other FIXTURE kind: the repeatedly folding geometry that bounds how far an invariance verdict
# read under perfect play carries (`agentic_memory_two_fold_fixture`). Its cells are deliberately
# NOT cap-fitting -- a cap-fitting guard cannot fold twice -- so it publishes no number either and
# is likewise absent from `AUDITED_DESIGN_PATHS`.
KIND_TWO_FOLD = "compact_two_fold_geometry"
GEOMETRY_KINDS = (*AUDITED_KINDS, KIND_INTERACTION, KIND_TWO_FOLD)

# The committed studies whose published numbers an agent policy change can invalidate. One registry,
# because the CLI audit and the CI pin gate must never walk different evidence.
AUDITED_DESIGN_PATHS: dict[str, str] = {
    KIND_SURFACE: "samples/benchmarks/agentic_compact_memory_boundary_surface_design.json",
    KIND_COLLAPSE: "samples/benchmarks/agentic_compact_trigger_guard_collapse_design.json",
    KIND_FOLD_STEP: "samples/benchmarks/agentic_compact_fold_step_crossover_design.json",
    KIND_CONSTANT_SWEEP: "samples/benchmarks/agentic_context_policy_constant_sweep_design.json",
    KIND_KEEP_LONG: "samples/benchmarks/agentic_context_policy_keep_long_design.json",
    KIND_HARNESS: "samples/benchmarks/agentic_harness_comparison_design.json",
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
    designs: dict[str, dict[str, object]],
    change: PolicyChange,
    *,
    pinned: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Replay every published cell of every design under both policies and compare its prompts.

    `pinned`, when supplied, is the full pinned policy the published numbers stand under. The pin
    gate always passes it so untouched fields replay the pinned values rather than a design's
    possibly-stale `held_fixed`; a hand-run CLI audit that omits it keeps the design fallback.
    """
    from llb.bench.policy_change.geometry import declared_geometry, held_fixed

    return {
        kind: [
            audit_cell_prompts(cell, held_fixed(design, kind), change, pinned=pinned)
            for cell in declared_geometry(design, kind)
        ]
        for kind, design in designs.items()
    }


def audit_cell_prompts(
    cell: dict[str, object],
    held: dict[str, object],
    change: PolicyChange,
    *,
    pinned: Mapping[str, Any] | None = None,
    controller: ReplayController | None = None,
) -> dict[str, object]:
    """One cell's verdict: do both arms send the identical prompts under both policies?

    A cell that declares one of the moved fields itself keeps its own value for that field and is
    audited on the rest of the change; a cell that declares ALL of them is not described by the
    change at all.

    `controller` chooses the walk the two policies are compared over, defaulting to each study's
    own oracle. The verdict is only ever as broad as that walk, which is why a caller can ask for
    the same comparison on the longest transcript the step budget allows.
    """
    cell_pins = [
        field for field in change.fields if field in cast(list[str], cell["pinned_fields"])
    ]
    described = change.without(cell_pins)
    row = {
        **cell,
        "policy_fields": list(change.fields),
        "baseline": dict(change.baseline),
        "candidate": dict(change.candidate),
        "not_applicable_fields": cell_pins,
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
    policies = cast(list[str], cell.get("policies", list(AUDITED_POLICIES)))
    arms = {
        name: arm_comparison(
            name,
            cell,
            held,
            described.baseline,
            described.candidate,
            pinned=pinned,
            controller=controller,
        )
        for name in policies
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
