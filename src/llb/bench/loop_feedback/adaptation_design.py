"""The prospective contract a family-specific repeat-feedback study is validated against.

Everything here runs BEFORE any model call: the task ledger and the families it must reach, the
candidate notices and their wording, the seed grid, the adoption rule, and the per-family cells.
`agentic_loop_feedback_adaptation` reads the runs those settings produced.
"""

from typing import cast

from llb.bench.agentic.loop_policy import (
    DEFAULT_REPEAT_FEEDBACK,
    REPEATED_NOOP_OBSERVATIONS,
    REPEAT_FEEDBACK_VARIANTS,
)
from llb.bench.agentic.model import AgenticTask
from llb.bench.context_policy.run import task_set_digest
from llb.bench.agentic.design_fields import (
    as_float,
    as_int,
    as_ints,
    as_mapping,
    as_str,
    as_strs,
)
from llb.bench.loop_feedback.run import validate_repeat_feedback_design
from llb.bench.loop_policy.grid import policy_grid

STUDY_KIND = "repeat_feedback_family_adaptation"
REQUIRED_FAMILIES = {"aya", "mistral", "gemma"}


def roster_rows(design: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], design["roster"])


def candidate_of(row: dict[str, object]) -> str:
    return cast(str, row["candidate_feedback_variant"])


def _check_adaptation_ledger(design: dict[str, object], tasks: list[AgenticTask]) -> list[str]:
    """The study, its task ledger, and the families it must route over."""
    if design.get("study_kind") != STUDY_KIND:
        raise ValueError(f"study_kind must be {STUDY_KIND}")
    reference = as_mapping(design, "reference")
    if reference.get("task_set_digest") != task_set_digest(tasks):
        raise ValueError("adaptation task digest does not match the predeclared ledger")
    families = as_strs(design, "required_model_families")
    if set(families) != REQUIRED_FAMILIES or len(families) != len(REQUIRED_FAMILIES):
        raise ValueError("adaptation requires exactly the Aya, Mistral, and Gemma families")
    roster = roster_rows(design)
    if [row.get("model_family") for row in roster] != families:
        raise ValueError("roster must contain each required family exactly once and in order")
    if any(row.get("backend") != "ollama" or not row.get("model") for row in roster):
        raise ValueError("every adaptation roster row needs an Ollama model")
    return families


def _validated_candidates(design: dict[str, object]) -> list[str]:
    """One unique registered candidate variant per family, declared against the current default."""
    candidates = [candidate_of(row) for row in roster_rows(design)]
    if len(set(candidates)) != len(candidates) or any(
        name == DEFAULT_REPEAT_FEEDBACK or name not in REPEAT_FEEDBACK_VARIANTS
        for name in candidates
    ):
        raise ValueError("each family needs one unique registered candidate feedback variant")
    if as_strs(design, "repeat_feedback_variants") != [DEFAULT_REPEAT_FEEDBACK, *candidates]:
        raise ValueError(
            "repeat_feedback_variants must declare current then every roster candidate"
        )
    return candidates


def _check_notice_contract(design: dict[str, object], candidates: list[str]) -> None:
    """Each candidate's wording is the registered one, concise, ASCII, and hypothesized."""
    notices = cast(dict[str, str], design["notice_text"])
    hypotheses = cast(dict[str, str], design["candidate_hypotheses"])
    maximum_notice_chars = as_int(design, "maximum_notice_chars")
    if set(notices) != set(candidates) or set(hypotheses) != set(candidates):
        raise ValueError("notice text and hypotheses must cover the exact candidate set")
    for name in candidates:
        _check_one_notice(name, notices[name], hypotheses[name], maximum_notice_chars)


def _check_one_notice(name: str, notice: str, hypothesis: str, maximum_chars: int) -> None:
    """One candidate's notice: the registered text, inside the concise controller contract."""
    if notice != REPEATED_NOOP_OBSERVATIONS[name]:
        raise ValueError(f"predeclared notice text does not match registered variant {name}")
    if not notice.isascii() or len(notice) > maximum_chars or not notice.startswith("[loop]"):
        raise ValueError(f"candidate notice {name} violates the concise controller contract")
    if not hypothesis.strip():
        raise ValueError(f"candidate hypothesis {name} must be predeclared")


def _validated_seeds(design: dict[str, object]) -> list[int]:
    """Two unique seeds at a decodable temperature, so a family route is not one lucky draw."""
    seeds = as_ints(design, "run_seeds")
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("family adaptation requires exactly two unique seeds")
    if not 0.0 < as_float(as_mapping(design, "sampling"), "temperature") <= 1.0:
        raise ValueError("seeded repetitions require temperature in (0, 1]")
    return seeds


def _check_adoption_rule(design: dict[str, object], families: list[str], seeds: int) -> None:
    """How much support a route needs before it may be adopted for a family."""
    rule = as_mapping(design, "cross_family_adoption_rule")
    if as_int(rule, "minimum_supported_seeds_per_family") != seeds:
        raise ValueError("family routing must require support on both seeds")
    minimum_families = as_int(rule, "minimum_supported_families")
    minimum_fraction = as_float(rule, "minimum_supported_fraction")
    if not 1 <= minimum_families <= len(families) or not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("cross-family thresholds are outside the roster range")


def _check_per_family_cells(design: dict[str, object], tasks: list[AgenticTask], seed: int) -> None:
    """Each family's own two-arm grid, checked through the shared repeat-feedback contract."""
    fixed = as_mapping(design, "fixed_policy")
    for row in roster_rows(design):
        validate_repeat_feedback_design(
            design,
            tasks,
            cells=policy_grid(
                [as_int(fixed, "max_steps")],
                [as_str(fixed, "malformed_call")],
                as_strs(fixed, "repeated_call"),
                [DEFAULT_REPEAT_FEEDBACK, candidate_of(row)],
            ),
            model_family=as_str(row, "model_family"),
            run_seed=seed,
        )


def validate_feedback_adaptation_design(
    design: dict[str, object], tasks: list[AgenticTask]
) -> None:
    """Refuse inference unless wording, hypotheses, routes, seeds, and gates are fixed."""
    families = _check_adaptation_ledger(design, tasks)
    candidates = _validated_candidates(design)
    _check_notice_contract(design, candidates)
    seeds = _validated_seeds(design)
    _check_adoption_rule(design, families, len(seeds))
    _check_per_family_cells(design, tasks, seeds[0])
