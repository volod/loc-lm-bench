"""Markdown rendering for the shared paired-power contract."""

from llb.rag.fusion_evidence.power import PowerAnalysis


def power_summary(analysis: PowerAnalysis, *, title: str) -> list[str]:
    """Compact report block shared by paired comparison lanes."""
    reached = "yes" if analysis["target_reached"] else "no"
    lines = [
        f"### {title}",
        "",
        f"- declared MDE: {analysis['minimum_detectable_delta']:+.3f}",
        f"- target: {analysis['target_power']:.0%} power at alpha={analysis['alpha']:.3f}",
        f"- reference SD / required n: {analysis['reference_sample_sd']:.3f} / "
        f"{analysis['required_n']} (binding floor: {analysis.get('binding_floor', 'variance')})",
        f"- planned n: {analysis['planned_n']}",
    ]
    if "realized_sample_sd" in analysis:
        lines += [
            f"- realized SD / resolvable MDE: {analysis['realized_sample_sd']:.3f} / "
            f"{analysis['resolvable_mde']:+.3f}",
            f"- realized required / reached n: {analysis['realized_required_n']} / "
            f"{analysis['realized_n']} (target reached: {reached}; binding floor: "
            f"{analysis['realized_binding_floor']})",
            f"- result: **{analysis['resolution']}** -- {analysis['reason']}",
        ]
    return lines + [""]


__all__ = ["power_summary"]
