"""Resolve every published crossover back to the aggregate that measured it.

The fold-step ANNOTATION beside a published crossover is checked against the study's own geometry,
which catches a slip large enough to leave the fold step -- and nothing else. The VALUE was a
hand-copied float, so a dropped digit landed inside the same step for any small slip and passed
every rule the design had. This module closes that: each published crossover names the run artifact
and the field it came from, and design validation reads the number back out rather than trusting it.

The three published forms resolve differently, for the same reason they place differently:

- an interpolated guard IS a field of the boundary surface's per-depth row;
- a fold-step boundary IS a field of the fold-step study's per-depth ladder;
- a portable ratio is DERIVED, so nothing in the collapse's aggregate holds it. What that aggregate
  holds is the cap peak the ratio divides by, and the guard is the surface value resolved just
  above, so the published band is re-derived with the runtime's own trigger arithmetic and the two
  edges are checked against the depths that produced them.
"""

from pathlib import Path
from typing import cast

from llb.bench.agentic_memory_fold_step_ladder import compaction_trigger_chars
from llb.bench.agentic_memory_crossover_restatement_placement import DERIVED_RATIO_SOURCE_KIND
from llb.bench.agentic_memory_crossover_restatement_reading import (
    FORM_INTERPOLATED,
    FORM_PORTABLE_RATIO,
)
from llb.bench.agentic_published_value_provenance import PublishedValueResolver


def validate_published_provenance(
    crossovers: list[dict[str, object]], *, root: Path, data_dir: Path | None = None
) -> None:
    """Refuse any published crossover whose value the aggregate it cites does not state."""
    resolver = PublishedValueResolver(root=root, data_dir=data_dir)
    guards = _resolved_guards(crossovers, resolver)
    ratios: dict[int, float] = {}
    for crossover in crossovers:
        form = str(crossover["form"])
        if form == FORM_PORTABLE_RATIO:
            ratios[_depth(crossover)] = _published_ratio(crossover, guards, resolver)
        elif form != FORM_INTERPOLATED:
            # Interpolated guards are already resolved above, because the ratios divide them. Every
            # OTHER form that states a value resolves here, phrased as an exclusion rather than as a
            # list, so a form added later is checked by default instead of silently skipped.
            _check_stated_value(crossover, resolver)
    _validate_published_band(crossovers, ratios)


def _resolved_guards(
    crossovers: list[dict[str, object]], resolver: PublishedValueResolver
) -> dict[int, float]:
    """Every interpolated guard, resolved first because the derived ratios are quotients of them."""
    return {
        _depth(crossover): _check_stated_value(crossover, resolver)
        for crossover in crossovers
        if crossover["form"] == FORM_INTERPOLATED
    }


def _check_stated_value(crossover: dict[str, object], resolver: PublishedValueResolver) -> float:
    """Compare a published value with the aggregate's own, exactly.

    Exactly, not within a tolerance: the aggregate writes the full float and JSON round-trips it, so
    any tolerance at all is a licence for precisely the transcription slip this check exists to
    catch -- one that stays inside the published fold step.
    """
    label = _label(crossover)
    measured = resolver.resolve(crossover.get("provenance"), where=label)
    stated = float(cast(float, crossover["value"]))
    if stated != measured:
        raise ValueError(
            f"{label}: the design publishes {stated!r} while the aggregate it cites measured "
            f"{measured!r} -- the published number was transcribed rather than resolved, so the "
            "restatement would re-check a number no run produced"
        )
    return measured


def _published_ratio(
    crossover: dict[str, object],
    guards: dict[int, float],
    resolver: PublishedValueResolver,
) -> float:
    """Re-derive one depth's published trigger ratio from the two aggregates that produced it.

    The ratio's own study measured no ratio -- it measured the cap PEAK the trigger is read against
    -- so this is the only form whose resolution spans two artifacts. The arithmetic is the runtime's
    (`compaction_trigger_chars`), and it is the same arithmetic the restatement applies to the
    RESTATED guard, so a published edge and a restated ratio are never computed two different ways.
    """
    label = _label(crossover)
    depth = _depth(crossover)
    guard = guards.get(depth)
    if guard is None:
        raise ValueError(
            f"{label}: the published band is derived from the {DERIVED_RATIO_SOURCE_KIND} guard at "
            "this depth, and no such guard is published here, so nothing resolves it"
        )
    peak = resolver.resolve(crossover.get("provenance"), where=label)
    if peak <= 0.0:
        raise ValueError(f"{label}: the aggregate states a non-positive cap peak {peak!r}")
    share = float(cast(float, crossover["compact_share"]))
    trigger = compaction_trigger_chars(int(guard), share)
    return round(trigger / peak, int(cast(int, crossover["band_decimals"])))


def _validate_published_band(crossovers: list[dict[str, object]], ratios: dict[int, float]) -> None:
    """The band's EDGES are the rounded ratios of the depths it was published across."""
    if not ratios:
        return
    resolved = [min(ratios.values()), max(ratios.values())]
    for crossover in crossovers:
        if crossover["form"] != FORM_PORTABLE_RATIO:
            continue
        published = [float(edge) for edge in cast(list[float], crossover["published_band"])]
        if published != resolved:
            decimals = int(cast(int, crossover["band_decimals"]))
            named = ", ".join(
                f"depth {depth} {ratios[depth]:.{decimals}f}x" for depth in sorted(ratios)
            )
            raise ValueError(
                f"{_label(crossover)}: the design publishes the "
                f"{published[0]}-{published[1]}x band while the aggregates it cites derive "
                f"{resolved[0]}-{resolved[1]}x from the published depths ({named}) -- the band was "
                "transcribed rather than resolved"
            )


def _label(crossover: dict[str, object]) -> str:
    return f"{crossover.get('study_kind')} depth {crossover.get('depth')}"


def _depth(crossover: dict[str, object]) -> int:
    return int(cast(int, crossover["depth"]))
