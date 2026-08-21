"""Render the host throughput summary as ASCII and as Markdown.

Rendering is kept out of the summary for the reason it is kept out of scoring everywhere else: a
column width is not a measurement, and a report change must not be able to move a number.
"""

from llb.rag.encoders.throughput import HostThroughputSummary, ThroughputProfile


def _profile_line(profile: ThroughputProfile) -> str:
    """One encoder's decomposed row: cold load, first pass, compile estimate, warm rate."""
    return (
        f"  {profile['model']:<48} {profile['device']:<5} "
        f"{profile['load_seconds']:7.2f} {profile['first_pass_seconds']:7.2f} "
        f"{profile['compile_estimate_seconds']:9.2f} "
        f"{profile['warm_median_seconds']:8.2f} {profile['warm_iqr_seconds']:6.2f} "
        f"{profile['warm_relative_precision']:6.3f} "
        f"{profile['warm_chunks_per_s']:8.1f} {profile['one_pass_chunks_per_s']:9.1f} "
        f"{profile['stopping_reason']}"
    )


def _baseline_lines(summary: HostThroughputSummary) -> list[str]:
    """The baseline comparison line, present only when a baseline was named."""
    baseline = summary.get("baseline_model")
    if not baseline:
        return []
    faster = summary.get("faster_than_baseline") or []
    if not faster:
        return [f"  faster_than_baseline ({baseline}): (none)"]
    named = ", ".join(
        f"{row['model']}={row['warm_chunks_per_s']:.1f}c/s({row['speedup_vs_baseline']:.2f}x)"
        if row.get("speedup_vs_baseline") is not None
        else f"{row['model']}={row['warm_chunks_per_s']:.1f}c/s"
        for row in faster
    )
    return [f"  faster_than_baseline ({baseline}): {named}"]


def format_host_summary(summary: HostThroughputSummary) -> str:
    """ASCII host summary for the encoder-throughput artifact."""
    devices = summary["devices"]
    headline = "cuda" if "cuda" in devices else (devices[0] if devices else "host")
    lines = [
        "[encoder-throughput] decomposed embedder rates",
        f"  texts={summary['corpus_n_texts']} devices={','.join(devices) or '-'}",
        (
            f"  {'model':<48} {'dev':<5} {'load_s':>7} {'first_s':>7} {'compile_s':>9} "
            f"{'warm_med':>8} {'iqr':>6} {'prec':>6} {'warm_c/s':>8} {'1pass_c/s':>9} stop"
        ),
    ]
    lines.extend(
        _profile_line(profile)
        for profile in sorted(
            summary["profiles"], key=lambda row: (-row["warm_chunks_per_s"], row["model"])
        )
    )
    lines.append(f"  one-pass order ({headline}): {' > '.join(summary['one_pass_order']) or '-'}")
    lines.append(f"  warm order ({headline}):     {' > '.join(summary['warm_order']) or '-'}")
    lines.append(f"  ordering_survives={summary['ordering_survives']}")
    lines.extend(
        f"  [{device}] survives={info['ordering_survives']} "
        f"warm={' > '.join(info['warm_order']) or '-'}"
        for device, info in sorted(summary.get("by_device", {}).items())
    )
    lines.extend(_baseline_lines(summary))
    lines.append(f"  verdict: {summary['verdict']}")
    return "\n".join(lines)


def render_host_markdown(summary: HostThroughputSummary) -> str:
    """Durable markdown for `$DATA_DIR/encoder-throughput/<run>/report.md`."""
    lines = [
        "# Encoder throughput decomposition",
        "",
        f"- texts: {summary['corpus_n_texts']}",
        f"- devices: {', '.join(summary['devices']) or '-'}",
        f"- ordering survives warm measurement: `{summary['ordering_survives']}`",
        f"- baseline: `{summary.get('baseline_model') or '-'}`",
        f"- verdict: {summary['verdict']}",
        "",
        "| model | device | load_s | first_s | compile_est_s | warm_median_s | warm_iqr_s "
        "| rel_precision | warm_chunks/s | one_pass_chunks/s | warm_passes | stop | "
        "peak_vram_mb | mean_power_w | power_limit_w |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | "
        "---: | ---: | ---: |",
    ]
    for profile in sorted(summary["profiles"], key=lambda p: (-p["warm_chunks_per_s"], p["model"])):
        lines.append(
            f"| `{profile['model']}` | {profile['device']} | {profile['load_seconds']:.3f} | "
            f"{profile['first_pass_seconds']:.3f} | {profile['compile_estimate_seconds']:.3f} | "
            f"{profile['warm_median_seconds']:.3f} | {profile['warm_iqr_seconds']:.3f} | "
            f"{profile['warm_relative_precision']:.3f} | {profile['warm_chunks_per_s']:.1f} | "
            f"{profile['one_pass_chunks_per_s']:.1f} | {profile['warm_passes']} | "
            f"{profile['stopping_reason']} | {profile.get('peak_vram_mb', '-')} | "
            f"{profile.get('mean_power_w', '-')} | {profile.get('power_limit_w', '-')} |"
        )
    faster = summary.get("faster_than_baseline") or []
    if summary.get("baseline_model"):
        if faster:
            named = ", ".join(
                f"`{row['model']}` ({row['warm_chunks_per_s']:.1f} c/s, "
                f"{row['speedup_vs_baseline']:.2f}x)"
                if row.get("speedup_vs_baseline") is not None
                else f"`{row['model']}` ({row['warm_chunks_per_s']:.1f} c/s)"
                for row in faster
            )
            faster_line = f"- faster than baseline on warm headline device: {named}"
        else:
            faster_line = "- faster than baseline on warm headline device: (none)"
    else:
        faster_line = "- faster than baseline on warm headline device: (baseline not set)"
    lines += [
        "",
        f"- one-pass order: {' > '.join(f'`{m}`' for m in summary['one_pass_order']) or '-'}",
        f"- warm order: {' > '.join(f'`{m}`' for m in summary['warm_order']) or '-'}",
        faster_line,
        "",
    ]
    return "\n".join(lines)
