"""The loop-policy grid: which cells a sweep runs, and which grids are refusable.

Separated from the sweep itself so the product of the four axes -- and the baseline every grid
must contain -- can be built and checked with no model, no budget, and no run directory.
"""

from llb.bench.agentic.loop_policy import (
    DEFAULT_REPEAT_FEEDBACK,
    MALFORMED_POLICIES,
    REPEATED_CALL_POLICIES,
    REPEATED_NOOP,
    REPEAT_FEEDBACK_VARIANTS,
    LoopPolicy,
)
from llb.bench.loop_policy.report import (
    BASELINE_MAX_STEPS,
    BASELINE_POLICY,
    LoopPolicyCell,
)


def _check_known(values: list[str], known: tuple[str, ...] | list[str], subject: str) -> None:
    """Refuse a grid axis naming something no policy implements, listing what it could have named."""
    unknown = [name for name in values if name not in known]
    if unknown:
        raise SystemExit(f"unknown {subject}: {unknown}; choose from {list(known)}")


def _expanded_cells(
    max_steps: list[int],
    malformed_policies: list[str],
    repeated_call_policies: list[str],
    feedback_variants: list[str],
) -> list[LoopPolicyCell]:
    """The full product, deduplicated in declared order.

    The feedback axis only varies under `noop`: it is the text a repeated call is answered WITH, so
    a grid that is not answering repeated calls has exactly one meaningful value for it.
    """
    return [
        LoopPolicyCell(steps, LoopPolicy(malformed, repeated, feedback))
        for steps in dict.fromkeys(max_steps)
        for malformed in dict.fromkeys(malformed_policies)
        for repeated in dict.fromkeys(repeated_call_policies)
        for feedback in (
            dict.fromkeys(feedback_variants)
            if repeated == REPEATED_NOOP
            else [DEFAULT_REPEAT_FEEDBACK]
        )
    ]


def policy_grid(
    max_steps: list[int],
    malformed_policies: list[str],
    repeated_call_policies: list[str],
    repeated_feedback_variants: list[str] | None = None,
) -> list[LoopPolicyCell]:
    """Validate and expand the grid, requiring its exact legacy baseline."""
    if not max_steps or any(value < 1 for value in max_steps):
        raise SystemExit("agent max steps must be a non-empty list of positive integers")
    _check_known(malformed_policies, MALFORMED_POLICIES, "malformed-call policies")
    _check_known(repeated_call_policies, REPEATED_CALL_POLICIES, "repeated-call policies")
    feedback_variants = repeated_feedback_variants or [DEFAULT_REPEAT_FEEDBACK]
    _check_known(feedback_variants, REPEAT_FEEDBACK_VARIANTS, "repeat-feedback variants")
    cells = _expanded_cells(
        max_steps, malformed_policies, repeated_call_policies, feedback_variants
    )
    if not any(cell.is_baseline for cell in cells):
        raise SystemExit(
            f"grid must include baseline max_steps={BASELINE_MAX_STEPS}, "
            f"malformed={BASELINE_POLICY.malformed_call}, "
            f"repeated={BASELINE_POLICY.repeated_call}"
        )
    return cells
