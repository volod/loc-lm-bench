"""Rendering of the measurement-floor report (`llb.rag.noise_floor.measure`).

Split from the measurement so the four lanes that publish a floor -- `compare-retrieval`, the
embedder bake-off, `compare-vector-stores`, and the graph-fusion sweep -- share one ASCII block
and one Markdown block instead of each shaping the same numbers. The measurement module stays
free of presentation.
"""

from llb.rag.noise_floor.models import (
    FloorMargin,
    MetricSpread,
    NoiseFloorReport,
)


def format_noise_floor(report: NoiseFloorReport) -> list[str]:
    """ASCII lines for the floor block (AGENTS.md: ASCII-only, no box-drawing)."""
    lines = [
        f"  noise floor ({report['replicates']} replicates, jitter {report['jitter']:.1e}, "
        f"candidates {report['candidates']}):"
    ]
    lanes = report["lanes"]
    if lanes:
        width = max(len(label) for label in lanes)
        for label in sorted(lanes):
            lane = lanes[label]
            lines.append(
                f"    {label.ljust(width)}   recall@k {_band(lane['recall_at_k'])}"
                f"   mrr {_band(lane['mrr'])}"
                f"   fragile {lane['fragile_items']}/{lane['n']}"
            )
    for label in report["unscored"]:
        lines.append(f"    {label}: no candidate score to perturb -- floor not measured")
    lines.append(
        f"  floor: recall@k +/-{report['floor_recall_at_k']:.3f}, "
        f"mrr +/-{report['floor_mrr']:.3f} -- read any smaller delta as noise"
    )
    margin = report.get("margin")
    if margin is not None:
        lines.append(f"  {format_margin(margin)}")
    return lines


def format_margin(margin: FloorMargin) -> str:
    """One sentence restating the lane ranking as clearing the floor or not."""
    if margin["runner_up"] is None:
        return f"top lane: {margin['leader']} (only measured lane -- nothing to clear)"
    verdict = (
        "clears the floor"
        if margin["clears_floor"]
        else "does NOT clear the floor -- the two lanes are not distinguished"
    )
    clearance = margin.get("clearance", margin["delta"] - margin["floor"])
    floor_multiple = margin.get(
        "floor_multiple",
        margin["delta"] / margin["floor"] if margin["floor"] > 0.0 else None,
    )
    distance = f"clearance {clearance:+.3f}"
    if floor_multiple is not None:
        distance += f", {floor_multiple:.2f}x the floor"
    return (
        f"top two: {margin['leader']} leads {margin['runner_up']} by "
        f"{margin['delta']:.3f} recall@k against a +/-{margin['floor']:.3f} floor "
        f"({distance}) -- {verdict}"
    )


def _per_lane_jitter_note(report: NoiseFloorReport) -> list[str]:
    """Name the per-lane amplitudes when a lane overrode the shared one.

    A reranked lane is read at a jitter scaled to its own score range, because two cross-encoder
    heads do not share a scale -- so the headline amplitude above is not what every row was read at,
    and a floor table that did not say so would invite exactly the wrong comparison.
    """
    per_lane = report.get("jitter_by_lane")
    if not per_lane:
        return []
    amplitudes = ", ".join(f"`{label}` {value:.1e}" for label, value in sorted(per_lane.items()))
    return [
        "Amplitudes are SCALE-MATCHED per row (each row's own ranking-score range x the amplitude "
        f"above), so rows on different score scales are read at comparable noise: {amplitudes}.",
        "",
    ]


def render_noise_floor_markdown(
    report: NoiseFloorReport, *, title: str = "Measurement floor", scored: str = ""
) -> list[str]:
    """The floor as a Markdown block, for the lanes whose artifact is a `report.md`.

    `scored` names the item set the floor was measured over, for a lane that measures it more than
    once (the fusion sweep measures the whole set and its focus slice, because the verdict reads
    the slice).
    """
    n = next((lane["n"] for lane in report["lanes"].values()), 0)
    lines = [
        f"### {title}",
        "",
        f"Every candidate score perturbed by `N(0, {report['jitter']:.1e})` over "
        f"{report['replicates']} seeded replicates of a {report['candidates']}-candidate pool "
        f"({scored or 'items'} scored: {n}; `src/llb/rag/noise_floor/measure.py`).",
        "",
        *_per_lane_jitter_note(report),
        "| row | recall@k band | MRR band | fragile |",
        "| --- | ---: | ---: | ---: |",
    ]
    lanes = report["lanes"]
    for label in sorted(lanes):
        lane = lanes[label]
        lines.append(
            f"| {label} | {_band(lane['recall_at_k'])} | {_band(lane['mrr'])} "
            f"| {lane['fragile_items']}/{lane['n']} |"
        )
    lines.append("")
    for label in report["unscored"]:
        lines.append(f"- `{label}`: no candidate score to perturb -- floor not measured")
    if report["unscored"]:
        lines.append("")
    lines.append(
        f"**Floor:** recall@k +/-{report['floor_recall_at_k']:.3f}, "
        f"MRR +/-{report['floor_mrr']:.3f} -- read any smaller delta as noise."
    )
    margin = report.get("margin")
    if margin is not None:
        lines.extend(["", f"**Reading:** {format_margin(margin)}."])
    lines.append("")
    return lines


def _band(spread: MetricSpread) -> str:
    return f"{spread['min']:.3f}-{spread['max']:.3f} (+/-{spread['half_width']:.3f})"
