"""Study identity and the reading for restating published crossovers under the shipped cap.

The published compact routing numbers were measured with the summarize call's input capped at the
compaction trigger. That bound discounted compact's own cost wherever it actually trimmed the folded
transcript, so a crossover read off those cells is read off a one-sided discount. The question this
study answers is not "re-run everything" but "where could the bound have mattered, and does the
routing rule change there".

The invariance criterion is the fold-step study's, not a char tolerance: a crossover guard that moves
INSIDE one fold step's guard interval names the same fold step, and the fold step is where the cost
actually changes. A restated crossover matters only when it crosses a step boundary.
"""

from typing import cast

STUDY_KIND = "compact_published_crossover_restatement"
METHOD = "agentic-compact-crossover-restatement"
REPORTING_CONFIDENCE = 0.975

RESTATEMENT_RULE = "restate_only_bound_sensitive_cells"
INVARIANCE_RULE = "restated_crossover_names_the_same_fold_step"

# What a published crossover is stated as, and therefore how it is restated.
FORM_INTERPOLATED = "interpolated_guard"
FORM_FOLD_STEP = "fold_step_boundary"
FORM_PORTABLE_RATIO = "portable_trigger_ratio"
CROSSOVER_FORMS = (FORM_INTERPOLATED, FORM_FOLD_STEP, FORM_PORTABLE_RATIO)

# Why a published crossover needs no run of its own.
BASIS_INVARIANT = "every_contributing_cell_is_bound_invariant"
BASIS_RESTATED = "restated_from_a_re_measured_cell"
BASIS_ALREADY_MEASURED = "already_re_measured_under_the_shipped_cap"

READING_ALL_INVARIANT = "every_published_cell_is_bound_invariant"
READING_UNCHANGED = "published_crossovers_hold_under_the_shipped_cap"
READING_MOVED = "a_published_crossover_moves_under_the_shipped_cap"
READING_INCOMPLETE = "a_bound_sensitive_cell_could_not_be_re_measured"
READING_INELIGIBLE = "pinned_family_control_ineligible"


def restatement_reading(
    eligible: bool, crossovers: list[dict[str, object]], n_sensitive: int
) -> tuple[str, str]:
    """Every published crossover must either be bound-invariant or survive its restatement."""
    if not eligible:
        return (
            READING_INELIGIBLE,
            "the pinned family no longer passes the unchanged token-chain control",
        )
    unresolved = [row for row in crossovers if row["basis"] is None]
    if unresolved:
        named = ", ".join(f"{row['study_kind']} depth {row['depth']}" for row in unresolved)
        return READING_INCOMPLETE, f"a bound-sensitive cell was not re-measured: {named}"
    moved = [row for row in crossovers if not row["names_same_fold_step"]]
    if moved:
        named = "; ".join(
            f"{row['study_kind']} depth {row['depth']}: fold step "
            f"{row['published_fold_step']} -> {row['restated_fold_step']}"
            for row in moved
        )
        return READING_MOVED, f"a restated crossover names a different fold step: {named}"
    if n_sensitive == 0:
        return (
            READING_ALL_INVARIANT,
            "no published cell's summarize input changes under the shipped cap, so every published "
            "crossover stands as measured with no run at all",
        )
    return (
        READING_UNCHANGED,
        f"{n_sensitive} bound-sensitive cell(s) were re-measured under the shipped cap and every "
        f"published crossover still names the fold step it named before",
    )


def operator_lines(
    crossovers: list[dict[str, object]], reading: str, shipped_cap: str
) -> list[str]:
    """What an operator takes away: which published numbers they may still apply, and why."""
    if reading == READING_INELIGIBLE:
        return [f"[{reading}] no restatement line is supported"]
    lines = [
        f"every published compact routing number below is stated for `summary_input_cap="
        f"{shipped_cap}`, the bound the shipped runtime runs"
    ]
    for row in crossovers:
        lines.append(_crossover_line(row))
    if reading == READING_MOVED:
        lines.append(
            "at least one crossover moved across a fold-step boundary -- re-derive the routing rule "
            "from the restated numbers, not the published ones"
        )
    return lines


def _crossover_line(row: dict[str, object]) -> str:
    label = f"{row['study_kind']} depth {row['depth']}"
    if row["form"] == FORM_FOLD_STEP:
        return (
            f"{label}: fold no later than step {row['published_fold_step']} -- unchanged "
            f"[{row['basis']}]"
        )
    if row["form"] == FORM_PORTABLE_RATIO:
        return f"{label}: the portable trigger ratio is unchanged [{row['basis']}]"
    published = cast(float, row["published_value"])
    restated = cast(float | None, row["restated_value"])
    if restated is None or restated == published:
        return (
            f"{label}: the interpolated crossover guard stays {published:.0f} chars, inside fold "
            f"step {row['published_fold_step']} [{row['basis']}]"
        )
    return (
        f"{label}: the interpolated crossover guard moves {published:.0f} -> {restated:.0f} chars "
        f"({restated - published:+.0f}), still inside fold step {row['restated_fold_step']}'s guard "
        f"interval, where every guard costs the same [{row['basis']}]"
    )
