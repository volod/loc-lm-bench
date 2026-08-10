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

from llb.bench.agentic_published_value_readings import CRITERION_ROUNDED_BAND

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
BASIS_DERIVED = "derived_from_the_restated_surface_guard"

# What a restated crossover is CHECKED against, which the FORM decides. A guard is a point on the
# deterministic step ladder, so it holds while it names the same fold step. The portable ratio is not
# a guard and names no step of its own -- it is a trigger over a cap peak, DERIVED from the surface's
# interpolated guard and published as a BAND across the tested depths -- so it holds while the
# restated ratio still falls inside that band at the precision the band is quoted to.
CRITERION_FOLD_STEP = INVARIANCE_RULE
CRITERION_BAND = CRITERION_ROUNDED_BAND

# Whether the geometry that measured the restated guard is the geometry the published peak came
# from. A guard ratio is a guard OVER a cap peak, so the two must be read off one prompt sequence;
# these say which case a restated depth is in rather than letting a mismatch vanish into the divide.
PEAK_INVARIANT = "the_re_measured_geometry_has_the_published_cap_peak"
PEAK_MOVED = "the_re_measured_geometry_has_a_different_cap_peak"
PEAK_UNPUBLISHED = "the_published_surface_states_no_cap_peak_at_this_depth"

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
    moved = [row for row in crossovers if not row["invariance_holds"]]
    if moved:
        named = "; ".join(_moved_phrase(row) for row in moved)
        return READING_MOVED, f"a restated crossover left the range it was published in: {named}"
    if n_sensitive == 0:
        return (
            READING_ALL_INVARIANT,
            "no published cell's summarize input changes under the shipped cap, so every published "
            "crossover stands as measured with no run at all",
        )
    return (
        READING_UNCHANGED,
        f"{n_sensitive} bound-sensitive cell(s) were re-measured under the shipped cap and every "
        f"published crossover still holds the statement it was published in",
    )


def _moved_phrase(row: dict[str, object]) -> str:
    """Name what a withdrawn crossover left, in the terms its own FORM was published in."""
    label = f"{row['study_kind']} depth {row['depth']}"
    if "reading_phrase" in row:
        return f"{label}: {row['reading_phrase']}"
    return f"{label}: fold step {row['published_fold_step']} -> {row['restated_fold_step']}"


def operator_lines(
    crossovers: list[dict[str, object]],
    cap_peaks: list[dict[str, object]],
    reading: str,
    shipped_cap: str,
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
    for row in cap_peaks:
        lines.append(_cap_peak_line(row))
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
        return _portable_ratio_line(label, row)
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


def _portable_ratio_line(label: str, row: dict[str, object]) -> str:
    """The derived ratio, the two numbers it is a quotient of, and the band it is read against.

    Both parts are named because the ratio is the one published form with no measurement of its own:
    an operator who reads only the quotient cannot tell a moved guard from a moved cap peak, and the
    two call for opposite actions -- re-derive the routing rule, or re-measure the geometry.
    """
    restated = cast(float | None, row["restated_value"])
    if restated is None:
        return f"{label}: the portable trigger ratio is unchanged [{row['basis']}]"
    return (
        f"{label}: the portable trigger ratio is {restated:.3f}x -- a "
        f"{cast(int, row['restated_trigger_chars'])}-char trigger, derived from the restated "
        f"{cast(float, row['derived_from_guard_chars']):.0f}-char guard, over the re-measured "
        f"{cast(int, row['restated_cap_peak_prompt_chars'])}-char cap peak -- "
        f"{row['reading_phrase']} [{row['basis']}]"
    )


def _cap_peak_line(row: dict[str, object]) -> str:
    """The peak a restated ratio is stated against, and whether the published one still names it."""
    label = f"{row['study_kind']} depth {row['depth']}"
    measured = cast(int, row["measured_cap_peak_prompt_chars"])
    ratio = cast(float | None, row["restated_guard_ratio"])
    against = (
        "no guard is bracketed at this depth, so no ratio is restated against it"
        if ratio is None
        else f"the restated guard ratio is {ratio:.2f}x it"
    )
    if row["reading"] == PEAK_MOVED:
        return (
            f"{label}: the cap peak moved "
            f"{cast(int, row['published_cap_peak_prompt_chars'])} -> {measured} chars "
            f"({cast(int, row['cap_peak_delta_chars']):+d}), so the published ratio was stated "
            f"against a peak this task world has retired -- {against} [{row['reading']}]"
        )
    if row["reading"] == PEAK_UNPUBLISHED:
        return (
            f"{label}: the published surface states no cap peak here; the re-measured peak is "
            f"{measured} chars and {against} [{row['reading']}]"
        )
    return (
        f"{label}: the re-measured cap peak is the published {measured} chars and {against} "
        f"[{row['reading']}]"
    )
