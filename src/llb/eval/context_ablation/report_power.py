"""Power-contract section for the context-ablation report."""

from llb.eval.context_ablation.models import ContextAblationReport


def power_section(report: ContextAblationReport) -> list[str]:
    analysis = report.get("power_analysis")
    if analysis is None:
        return []
    entry = next(
        (
            item
            for item in reversed(report["derived"])
            if item["label"].startswith("long_context_delta")
        ),
        None,
    )
    stability = entry["paired"].get("stability") if entry else None
    p_positive = f"{stability['p_positive']:.4f}" if stability else "not measured"
    reached = "yes" if analysis["target_reached"] else "no"
    planned_reached = (
        "yes" if analysis.get("planned_target_reached", analysis["target_reached"]) else "no"
    )
    lines = [
        "### Predeclared long-context resolution",
        "",
        f"- minimum detectable delta: {analysis['minimum_detectable_delta']:+.3f} objective",
        f"- target: {analysis['target_power']:.0%} power at alpha={analysis['alpha']:.3f}, "
        f"two-sided `{analysis['method']}`",
        f"- reference: n={analysis['reference_n']}, paired sample SD "
        f"{analysis['reference_sample_sd']:.3f}, `{analysis['reference_artifact']}`",
        f"- required / planned items: {analysis['required_n']} / {analysis['planned_n']} "
        f"(planned target reached: {planned_reached})",
    ]
    if "realized_sample_sd" in analysis:
        lines += [
            f"- realized paired SD / resolvable MDE: {analysis['realized_sample_sd']:.3f} / "
            f"{analysis['resolvable_mde']:+.3f}",
            f"- realized required / reached items: {analysis['realized_required_n']} / "
            f"{analysis['realized_n']} (target reached: {reached}; binding floor: "
            f"{analysis['realized_binding_floor']})",
        ]
    lines += [
        f"- result: **{analysis.get('resolution', 'undecidable')}** "
        f"(direction: `{analysis.get('direction', 'none')}`, p_positive: {p_positive}) -- "
        f"{analysis.get('reason', 'the new run has no resolution')}",
        "",
    ]
    return lines
