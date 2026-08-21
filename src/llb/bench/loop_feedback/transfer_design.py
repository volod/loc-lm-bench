"""The prospective contract a neutral repeat-feedback study is validated against.

Everything here runs BEFORE any model call: the study identity, the freshness of its task ledger,
the single-family roster, the immutable notice wording, the seed grid, and the adoption thresholds.
`agentic_loop_feedback_transfer` reads the runs those settings produced.
"""

import re
from typing import cast

from llb.bench.agentic.loop_policy import (
    DEFAULT_REPEAT_FEEDBACK,
    REPEATED_NOOP_OBSERVATIONS,
    REPEAT_FEEDBACK_GEMMA_PROGRESS,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.context_policy.run import task_set_digest
from llb.bench.agentic.design_fields import (
    as_float,
    as_int,
    as_ints,
    as_mapping,
    as_rows,
    as_str,
    as_strs,
)
from llb.bench.loop_feedback.run import validate_repeat_feedback_design
from llb.bench.loop_policy.grid import policy_grid

STUDY_KIND = "repeat_feedback_task_family_transfer"
MODEL_FAMILY = "gemma"
EXPECTED_HYPOTHESIS = (
    "Completed-state feedback will make Gemma advance after a suppressed repeat in at least "
    "three task families while preserving material completion and paired cost bounds."
)
FORBIDDEN_NOTICE_TERMS = (
    "answer",
    "calculator",
    "database",
    "file",
    "mutation",
    "read",
    "search",
    "tool",
    "write",
)


def candidate_variant_of(design: dict[str, object]) -> str:
    """The registered notice this study is comparing the current feedback against."""
    return cast(str, design["candidate_feedback_variant"])


def validate_feedback_transfer_design(design: dict[str, object], tasks: list[AgenticTask]) -> None:
    """Refuse inference unless the neutral notice, fresh ledger, seeds, and gates are fixed."""
    validate_neutral_feedback_design(
        design,
        tasks,
        study_kind=STUDY_KIND,
        hypothesis=EXPECTED_HYPOTHESIS,
        candidate_variant=REPEAT_FEEDBACK_GEMMA_PROGRESS,
    )


def _check_transfer_identity(
    design: dict[str, object], tasks: list[AgenticTask], study_kind: str, hypothesis: str
) -> None:
    """The study, its immutable hypothesis, and a task ledger fresh against every prior one."""
    if design.get("study_kind") != study_kind:
        raise ValueError(f"study_kind must be {study_kind}")
    if design.get("hypothesis") != hypothesis:
        raise ValueError("transfer hypothesis does not match the immutable prospective hypothesis")
    reference = as_mapping(design, "reference")
    digest = task_set_digest(tasks)
    if reference.get("task_set_digest") != digest:
        raise ValueError("transfer task digest does not match the predeclared holdout ledger")
    excluded = as_strs(reference, "excluded_prior_task_set_digests")
    if not excluded or digest in excluded:
        raise ValueError("transfer ledger must be fresh relative to every excluded prior digest")


def _check_single_family_roster(design: dict[str, object]) -> None:
    """One installed Gemma row: the family this study transfers ACROSS TASKS, not across models."""
    families = as_strs(design, "required_model_families")
    roster = as_rows(design, "roster")
    if families != [MODEL_FAMILY] or len(roster) != 1:
        raise ValueError("task-family transfer requires exactly one Gemma roster row")
    if (
        roster[0].get("model_family") != MODEL_FAMILY
        or roster[0].get("backend") != "ollama"
        or not roster[0].get("model")
    ):
        raise ValueError("transfer roster must declare one installed Ollama Gemma model")


def _check_notice_contract(design: dict[str, object], candidate_variant: str) -> str:
    """The candidate arm and its wording, which must name no task and no answer."""
    candidate = candidate_variant_of(design)
    if candidate != candidate_variant:
        raise ValueError("transfer candidate must be the immutable registered notice")
    if as_strs(design, "repeat_feedback_variants") != [DEFAULT_REPEAT_FEEDBACK, candidate]:
        raise ValueError("transfer must isolate current versus its registered candidate feedback")
    notice = as_str(design, "notice_text")
    if notice != REPEATED_NOOP_OBSERVATIONS[candidate]:
        raise ValueError("transfer notice does not match the registered immutable text")
    if not notice.isascii() or len(notice) > as_int(design, "maximum_notice_chars"):
        raise ValueError("transfer notice violates the concise ASCII controller contract")
    forbidden = as_strs(design, "forbidden_notice_terms")
    notice_words = set(re.findall(r"[a-z]+", notice.casefold()))
    if forbidden != list(FORBIDDEN_NOTICE_TERMS) or notice_words.intersection(forbidden):
        raise ValueError("transfer notice contains task-specific or answer-specific language")
    return candidate


def _validated_seeds(design: dict[str, object]) -> list[int]:
    """Two unique seeds at a decodable temperature."""
    seeds = as_ints(design, "run_seeds")
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("task-family transfer requires exactly two unique seeds")
    if not 0.0 < as_float(as_mapping(design, "sampling"), "temperature") <= 1.0:
        raise ValueError("seeded repetitions require temperature in (0, 1]")
    return seeds


def _check_response_rule(design: dict[str, object], seeds: list[int]) -> None:
    """How many task families must respond, on how many seeds, before this may be adopted."""
    response_rule = as_mapping(design, "task_family_response_rule")
    response_floor = as_float(response_rule, "minimum_response_rate")
    minimum_families = as_int(response_rule, "minimum_supported_task_families_per_seed")
    required_families = cast(dict[str, int], design["required_task_families"])
    if not 0.0 < response_floor <= 1.0 or not 3 <= minimum_families <= len(required_families):
        raise ValueError("task-family response thresholds are outside the declared ledger")
    if as_int(response_rule, "minimum_supported_seeds") != len(seeds):
        raise ValueError("transfer adoption must require every predeclared seed")


def validate_neutral_feedback_design(
    design: dict[str, object],
    tasks: list[AgenticTask],
    *,
    study_kind: str,
    hypothesis: str,
    candidate_variant: str,
) -> None:
    """Validate one prospective neutral-feedback contract before model inference."""
    _check_transfer_identity(design, tasks, study_kind, hypothesis)
    _check_single_family_roster(design)
    candidate = _check_notice_contract(design, candidate_variant)
    seeds = _validated_seeds(design)
    _check_response_rule(design, seeds)
    fixed = as_mapping(design, "fixed_policy")
    validate_repeat_feedback_design(
        design,
        tasks,
        cells=policy_grid(
            [as_int(fixed, "max_steps")],
            [as_str(fixed, "malformed_call")],
            as_strs(fixed, "repeated_call"),
            [DEFAULT_REPEAT_FEEDBACK, candidate],
        ),
        model_family=MODEL_FAMILY,
        run_seed=seeds[0],
    )
