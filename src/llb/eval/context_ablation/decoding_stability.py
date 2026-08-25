"""How much of a lane's number is the decode, not the model (pure).

Every interval this comparison prints is a paired bootstrap over the ITEM SAMPLE. None of them can
see the other source of uncertainty: scoring the identical configuration on the identical items
again and getting a different answer. Greedy decoding is not bit-reproducible on a GGUF runtime --
kernel scheduling and cache state decide the last few floating-point bits, and where two candidate
tokens are nearly tied that flips the token.

That flip is NOT equally likely in every lane. A grounded prompt carries the answer in its context,
so the next-token distribution is sharply peaked and a bit of drift changes nothing; a closed-book
prompt leaves a much flatter distribution, so the same drift rewrites the answer. A contamination
rate quoted to one decimal place off a single closed-book run therefore claims precision it does
not have, and the fix is to MEASURE the band rather than to assume it is small.

The measurement is one repeat of each lane -- the identical config, the identical items -- and this
module states, per lane, the band its own mean objective, token F1, and reference-match rate
occupy, plus the count of items that moved at all. It then re-reads each derived delta against the
decoding floor of the two lanes it is taken over, so an operator sees which readings survive the
decode and which ones are inside it.

Pure and file-driven like the rest of the comparison: the input is one list of canonical per-case
rows per (lane, repeat), so the whole statistic is unit-tested with dict rows -- no backend, no GPU.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from llb.eval.context_ablation.derived import is_contaminated
from llb.eval.context_ablation.models import (
    LANE_CLOSED_BOOK,
    METRIC_OBJECTIVE,
    METRIC_TOKEN_F1,
    STABILITY_DRIFTS,
    STABILITY_REPRODUCIBLE,
    DecodingFloorMargin,
    DecodingStabilityReport,
    DerivedComparison,
    LaneDecodingSpread,
)
from llb.eval.paired_cases import CaseRows, rows_by_item
from llb.rag.fusion_evidence.spread import ValueSpread, value_spread

# The recorded answer text a divergence is read off. It is a PREVIEW, so two answers that agree for
# this many characters and diverge afterwards are counted as identical -- the count is a lower
# bound, and the objective divergence beside it is the number that is exact.
ANSWER_COLUMN = "answer_preview"

# Two repeats are the fewest that can have a spread at all; below that there is nothing to report.
MIN_REPEATS = 2


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _repeat_rows(rows: CaseRows, item_ids: Sequence[str], lane: str) -> CaseRows:
    """One repeat's rows in the shared item order; a repeat that scored a different set raises.

    A band drawn over two different item sets is not a band, so the mismatch fails loudly rather
    than being silently intersected away -- the same rule `shared_item_ids` applies across lanes.
    """
    by_item = rows_by_item(rows)
    missing = [item_id for item_id in item_ids if item_id not in by_item]
    if missing:
        raise ValueError(
            f"lane {lane!r} repeat scored a different item set (missing {missing[:3]})"
        )
    return [by_item[item_id] for item_id in item_ids]


def _column(rows: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    return [float(row.get(metric, 0.0) or 0.0) for row in rows]


def _divergent(per_repeat: Sequence[Sequence[Any]]) -> int:
    """Items whose value is not identical in every repeat."""
    return sum(len(set(values)) > 1 for values in zip(*per_repeat, strict=True))


def _outcome_groups(per_repeat: Sequence[Sequence[float]]) -> list[int]:
    """Sizes of the groups of repeats that produced the identical per-item vector.

    The half-width says how FAR the drift went; this says what shape it had. A lane that answers
    differently on its first pass and then settles is a warm-up transient with a remedy; a lane
    whose every pass is new is irreducible noise. Both can print the same band.
    """
    groups: dict[tuple[float, ...], int] = {}
    for values in per_repeat:
        key = tuple(values)
        groups[key] = groups.get(key, 0) + 1
    return list(groups.values())


def lane_spread(
    lane: str,
    repeats: Sequence[CaseRows],
    item_ids: Sequence[str],
    *,
    baseline: str,
    run_dirs: Sequence[Sequence[str]] = (),
) -> LaneDecodingSpread:
    """One lane's between-repeat band over the shared item set."""
    aligned = [_repeat_rows(rows, item_ids, lane) for rows in repeats]
    objectives = [_column(rows, METRIC_OBJECTIVE) for rows in aligned]
    token_f1s = [_column(rows, METRIC_TOKEN_F1) for rows in aligned]
    matches = [[float(is_contaminated(row)) for row in rows] for rows in aligned]
    answers = [[str(row.get(ANSWER_COLUMN, "")) for row in rows] for rows in aligned]
    return {
        "lane": lane,
        "grounded": lane != baseline,
        "run_dirs": [str(path) for dirs in run_dirs for path in dirs],
        "objective": _band(objectives),
        "token_f1": _band(token_f1s),
        "match_rate": _band(matches),
        "divergent_items": _divergent(objectives),
        "answer_divergent_items": _divergent(answers),
        "outcome_groups": _outcome_groups(objectives),
    }


def _band(per_repeat: Sequence[Sequence[float]]) -> ValueSpread:
    """The band the per-repeat MEANS occupy, quoted against the first repeat's mean.

    The first repeat is the one the comparison itself was taken over, so it is the value the
    artifact prints and the value this band qualifies.
    """
    means = [_mean(values) for values in per_repeat]
    return value_spread(means[0], means)


def delta_floors(
    derived: Sequence[DerivedComparison], lanes: Mapping[str, LaneDecodingSpread]
) -> list[DecodingFloorMargin]:
    """Each derived delta read against the decoding floor of its own two lanes.

    A delta whose lanes were not both repeated is absent rather than floored at zero: an unmeasured
    floor is not a floor of zero, and printing one would read as "this delta is decode-stable".
    """
    margins: list[DecodingFloorMargin] = []
    for entry in derived:
        candidate, reference = lanes.get(entry["candidate"]), lanes.get(entry["reference"])
        if candidate is None or reference is None:
            continue
        floor = candidate["objective"]["half_width"] + reference["objective"]["half_width"]
        delta = entry["paired"]["delta"]["mean"]
        margins.append(
            {
                "label": entry["label"],
                "n": entry["n"],
                "delta": delta,
                "floor": floor,
                "clears_floor": abs(delta) > floor,
                "floor_multiple": abs(delta) / floor if floor > 0.0 else None,
            }
        )
    return margins


def _reason(
    baseline: str, baseline_floor: float, grounded_floor: float, multiple: float | None
) -> str:
    """The one sentence the reading rests on: which lane the decode moves, and by how much.

    Stated in both directions on purpose. The premise this measurement was built to check -- that
    an ungrounded prompt is the noisier one, because its next-token distribution is flatter -- is
    a hypothesis, and a run where the baseline lane is the QUIETER one has to say so rather than
    print the same sentence with a small number in it.
    """
    if baseline_floor == 0.0 and grounded_floor == 0.0:
        return "every lane reproduced its own mean exactly, so no number here is decode noise"
    band = (
        f"the `{baseline}` lane's mean objective moves +/-{baseline_floor:.4f} between identical "
        "repeats"
    )
    if multiple is None:
        return f"{band}, and every grounded lane reproduced exactly; quote it with that band"
    if multiple > 1.0:
        return (
            f"{band}, {multiple:.1f}x the widest grounded band (+/-{grounded_floor:.4f}) -- the "
            "ungrounded lane is the noisier measurement, so quote its numbers with that band"
        )
    return (
        f"{band}, against a widest grounded band of +/-{grounded_floor:.4f} -- the ungrounded lane "
        "is NOT the noisier measurement here, so quote every lane with its own band"
    )


def measure_decoding_stability(
    repeats: Mapping[str, Sequence[CaseRows]],
    item_ids: Sequence[str],
    derived: Sequence[DerivedComparison],
    *,
    run_dirs: Mapping[str, Sequence[Sequence[str]]] | None = None,
    baseline: str = LANE_CLOSED_BOOK,
) -> DecodingStabilityReport:
    """Band every lane's own numbers occupy across identical repeats, and floor the deltas."""
    counts = {len(rows) for rows in repeats.values()}
    if len(counts) != 1:
        raise ValueError("every lane must be repeated the same number of times")
    count = counts.pop()
    if count < MIN_REPEATS:
        raise ValueError(f"decoding stability needs at least {MIN_REPEATS} repeats of every lane")
    if baseline not in repeats:
        raise ValueError(f"baseline lane {baseline!r} is not among the repeated lanes")
    lanes = {
        label: lane_spread(
            label,
            lane_repeats,
            item_ids,
            baseline=baseline,
            run_dirs=(run_dirs or {}).get(label, ()),
        )
        for label, lane_repeats in repeats.items()
    }
    baseline_floor = lanes[baseline]["objective"]["half_width"]
    grounded_floor = max(
        (lane["objective"]["half_width"] for lane in lanes.values() if lane["grounded"]),
        default=0.0,
    )
    multiple = baseline_floor / grounded_floor if grounded_floor > 0.0 else None
    reproducible = all(lane["divergent_items"] == 0 for lane in lanes.values())
    return {
        "repeats": count,
        "n": len(item_ids),
        "baseline": baseline,
        "lanes": lanes,
        "baseline_floor": baseline_floor,
        "grounded_floor": grounded_floor,
        "noise_multiple": multiple,
        "deltas": delta_floors(derived, lanes),
        "reading": STABILITY_REPRODUCIBLE if reproducible else STABILITY_DRIFTS,
        "reason": _reason(baseline, baseline_floor, grounded_floor, multiple),
    }


__all__ = ["delta_floors", "lane_spread", "measure_decoding_stability"]
