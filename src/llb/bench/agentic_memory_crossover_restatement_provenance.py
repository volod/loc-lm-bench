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
  above, so the published band is re-derived through the operation the design names and the two
  edges are checked against the depths that produced them.

The walk COLLECTS (`agentic_published_value_collection`), so a re-run that moved three of this
study's six published numbers is met with one refusal naming three, not three runs of the check.
Derivation is what makes that more than a loop over the forms: the band is a QUOTIENT of an
interpolated guard, so a guard the evidence no longer states is the CAUSE of its band being
unresolvable rather than a second moved number. Neither WHICH guard nor HOW it divides one is
knowledge of this module -- the design declares the sources (`derived_from`) and names the arithmetic
(`operation`), the accumulator carries the graph, and the marking is transitive. Hence four passes --
shape, then every STATED value in the design's own order, then the consequences of anything that did
not resolve, then the DERIVED band from what did.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from llb.bench.agentic_memory_crossover_restatement_reading import FORM_PORTABLE_RATIO
from llb.bench.agentic_published_value_collection import CollectedRefusals
from llb.bench.agentic_published_value_derivation import (
    ValueKey,
    published_key,
    required_derivation,
)
from llb.bench.agentic_published_value_derivation_graph import derivation_graph
from llb.bench.agentic_published_value_operations import DerivedValue
from llb.bench.agentic_published_value_provenance import PublishedValueResolver, provenance_pair


@dataclass(slots=True)
class _Stated:
    """What the aggregates state, gathered before anything is derived from it.

    Keyed by published value rather than by depth, because what a derived form reads is the value at
    the identity it DECLARED as its source -- which need not be the same depth, and in a design with
    two forms at a depth is not even named by one. The cap peaks stay per depth: they are the ratio
    study's own stated value, read at the depth that ratio is published at.
    """

    values: dict[ValueKey, float] = field(default_factory=dict)
    peaks: dict[int, float] = field(default_factory=dict)


def validate_published_provenance(
    crossovers: list[dict[str, object]], *, root: Path, data_dir: Path | None = None
) -> None:
    """Refuse, once, every published crossover whose value the aggregate it cites does not state."""
    _refuse_malformed(crossovers)
    collected = CollectedRefusals(derivations=derivation_graph(crossovers))
    resolver = PublishedValueResolver(root=root, data_dir=data_dir)
    stated = _resolve_stated_values(crossovers, resolver, collected)
    consequences = _mark_derived_consequences(crossovers, collected)
    ratios = _derived_ratios(crossovers, stated, collected, consequences)
    _check_published_band(crossovers, ratios, collected)
    collected.refuse(total=len(crossovers))


def _resolve_stated_values(
    crossovers: list[dict[str, object]],
    resolver: PublishedValueResolver,
    collected: CollectedRefusals,
) -> _Stated:
    """Read every value an aggregate states back out of it, in the order the design publishes them.

    The portable ratio is the one form no aggregate states, so what resolves here for it is the cap
    PEAK its band divides by; the quotient itself is derived a pass later.
    """
    stated = _Stated()
    for crossover in crossovers:
        key = published_key(crossover)
        if key.form == FORM_PORTABLE_RATIO:
            peak = collected.collect(lambda: _stated_cap_peak(crossover, resolver), key=key)
            if peak is not None:
                stated.peaks[key.depth] = peak
            continue
        # Every OTHER form that states a value resolves here, phrased as an exclusion rather than as
        # a list, so a form added later is checked by default instead of silently skipped.
        value = collected.collect(lambda: _check_stated_value(crossover, resolver), key=key)
        if value is not None:
            stated.values[key] = value
    return stated


def _mark_derived_consequences(
    crossovers: list[dict[str, object]], collected: CollectedRefusals
) -> set[ValueKey]:
    """Mark NOT JUDGED every value whose declaration rests on something that did not resolve.

    Its own pass, over every published value rather than over one form, because "is this number a
    consequence of a number already named" is a question of the design's declared edges and not of
    what any particular form is computed with. The keys are returned so the derivation below skips
    what has already been spoken for instead of naming it a second way.
    """
    consequences: set[ValueKey] = set()
    for crossover in crossovers:
        key = published_key(crossover)
        if collected.rests_on_unresolved(key):
            consequences.add(key)
    return consequences


def _check_stated_value(crossover: dict[str, object], resolver: PublishedValueResolver) -> float:
    """Compare a published value with the aggregate's own, exactly.

    Exactly, not within a tolerance: the aggregate writes the full float and JSON round-trips it, so
    any tolerance at all is a licence for precisely the transcription slip this check exists to
    catch -- one that stays inside the published fold step.
    """
    label = _label(crossover)
    measured = resolver.resolve(crossover.get("provenance"), where=label)
    stated = _stated_number(crossover, "value")
    if stated != measured:
        raise ValueError(
            f"{label}: the design publishes {stated!r} while the aggregate it cites measured "
            f"{measured!r} -- the published number was transcribed rather than resolved, so the "
            "restatement would re-check a number no run produced"
        )
    return measured


def _stated_cap_peak(crossover: dict[str, object], resolver: PublishedValueResolver) -> float:
    """The cap peak a published band is read against, which is what its aggregate does state."""
    label = _label(crossover)
    peak = resolver.resolve(crossover.get("provenance"), where=label)
    if peak <= 0.0:
        raise ValueError(f"{label}: the aggregate states a non-positive cap peak {peak!r}")
    return peak


def _derived_ratios(
    crossovers: list[dict[str, object]],
    stated: _Stated,
    collected: CollectedRefusals,
    consequences: set[ValueKey],
) -> dict[int, float]:
    """Re-derive each published depth's trigger ratio from the two aggregates that produced it.

    The ratio's own study measured no ratio -- it measured the cap PEAK the trigger is read against
    -- so this is the only form whose resolution spans two artifacts. Neither WHICH guard it divides
    nor HOW is this module's knowledge: the design declares the sources and names the arithmetic, and
    `required_derivation` hands back both. The restatement re-derives a restated ratio through that
    same registered operation, so a published edge and a restated row cannot be computed two
    different ways.

    A ratio whose guard did not resolve was already marked NOT JUDGED a pass above, and one whose own
    cap peak did not resolve is marked so here, because the moved number is the one already named and
    this is its consequence.
    """
    ratios: dict[int, float] = {}
    for crossover in crossovers:
        key = published_key(crossover)
        if key.form != FORM_PORTABLE_RATIO or key in consequences:
            continue
        if key.depth not in stated.peaks:
            collected.not_judged(
                f"{_label(crossover)}: the cap peak this band is a quotient over is named above, so "
                "the band cannot be re-derived from here"
            )
            continue
        derived = collected.collect(lambda: _re_derived(crossover, stated), key=key)
        if derived is None:
            continue
        ratios[key.depth] = round(derived.value, int(_stated_number(crossover, "band_decimals")))
    return ratios


def _re_derived(crossover: dict[str, object], stated: _Stated) -> DerivedValue:
    """Run this value's declared operation over the values its declared sources resolved to.

    Total by construction rather than by a fallback: the declaration is validated against what the
    design publishes before any value is read, and every published value either resolved into
    `stated` or was collected as unresolved -- and an unresolved source is marked a pass above, so
    what reaches here are sources that resolved.
    """
    derivation = required_derivation(crossover)
    sources = tuple(stated.values[source] for source in derivation.sources)
    return derivation.compute(sources, measured=stated.peaks[published_key(crossover).depth])


def _check_published_band(
    crossovers: list[dict[str, object]], ratios: dict[int, float], collected: CollectedRefusals
) -> None:
    """The band's EDGES are the rounded ratios of the depths it was published across.

    Grouped by the band each row publishes, so one band published across several depths is one line
    naming the rows that publish it rather than the same sentence once per depth.
    """
    published: dict[tuple[float, ...], list[dict[str, object]]] = {}
    for crossover in crossovers:
        if crossover["form"] == FORM_PORTABLE_RATIO:
            published.setdefault(tuple(_band_edges(crossover)), []).append(crossover)
    if not published:
        return
    depths = {_depth(row) for rows in published.values() for row in rows}
    if len(ratios) != len(depths):
        # Stated even though every missing quotient is named above, because the band is a claim of
        # its own: without this line, a band nothing could re-derive reads exactly like one that was
        # re-derived and held.
        collected.not_judged(
            "the published band itself: its edges are the smallest and largest of the per-depth "
            f"quotients, and {len(depths) - len(ratios)} of the {len(depths)} published depths did "
            "not resolve above, so the band is not re-derived at all here"
        )
        return
    resolved = [min(ratios.values()), max(ratios.values())]
    for edges, rows in published.items():
        if list(edges) == resolved:
            continue
        decimals = int(_stated_number(rows[0], "band_decimals"))
        named = ", ".join(
            f"depth {depth} {ratios[depth]:.{decimals}f}x" for depth in sorted(ratios)
        )
        collected.unresolvable(
            f"{', '.join(_label(row) for row in rows)}: the design publishes the "
            f"{edges[0]}-{edges[1]}x band while the aggregates it cites derive "
            f"{resolved[0]}-{resolved[1]}x from the published depths ({named}) -- the band was "
            "transcribed rather than resolved"
        )


def _refuse_malformed(crossovers: list[dict[str, object]]) -> None:
    """Refuse a crossover whose SHAPE this resolution cannot read, before any value is read.

    Fail-fast rather than collected: a value with no provenance is not a value the evidence failed to
    state, it is a design that never said what would state it, and that message must not arrive
    underneath a list of numbers that could not be checked because of it.
    """
    for crossover in crossovers:
        provenance_pair(crossover.get("provenance"), where=_label(crossover))
        _stated_number(crossover, "depth")
        if str(crossover.get("form")) == FORM_PORTABLE_RATIO:
            _band_edges(crossover)
            _stated_number(crossover, "compact_share")
            _stated_number(crossover, "band_decimals")
        else:
            _stated_number(crossover, "value")


def _band_edges(crossover: dict[str, object]) -> list[float]:
    """The two edges a derived band is published as, which the re-derived quotients are read against."""
    band = crossover.get("published_band")
    if not isinstance(band, list) or len(band) != 2 or any(_not_a_number(edge) for edge in band):
        raise ValueError(
            f"{_label(crossover)}: a {FORM_PORTABLE_RATIO} crossover must carry the two numeric "
            "`published_band` edges its re-derived quotients are read against"
        )
    return [float(cast(float, edge)) for edge in band]


def _stated_number(crossover: dict[str, object], name: str) -> float:
    """One numeric field this resolution reads, named in the refusal rather than raised as a KeyError."""
    value = crossover.get(name)
    if _not_a_number(value):
        raise ValueError(
            f"{_label(crossover)}: a published {crossover.get('form')} crossover must state a "
            f"numeric `{name}` for its value to be resolved, got {value!r}"
        )
    return float(cast(float, value))


def _not_a_number(value: object) -> bool:
    return isinstance(value, bool) or not isinstance(value, (int, float))


def _label(crossover: dict[str, object]) -> str:
    return f"{crossover.get('study_kind')} depth {crossover.get('depth')}"


def _depth(crossover: dict[str, object]) -> int:
    return int(cast(int, crossover["depth"]))
