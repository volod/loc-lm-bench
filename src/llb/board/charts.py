"""Comparison chart for benchmarked models (matplotlib, optional [viz] extra).

A multi-panel bar figure -- accuracy, reliability, throughput, efficiency -- over the benchmarked
models, in the spirit of the MamayLM release charts. matplotlib is imported lazily so the rest of the
board works without the [viz] extra; `render_comparison_chart` returns None (and logs) when it is
missing.
"""

import logging
from pathlib import Path

from llb.board.recommend import Recommendation, _short

_LOG = logging.getLogger(__name__)

# (title, unit, accessor) for each panel; accessor returns a float or None per model run.
_PANELS = (
    ("Accuracy (objective)", "", lambda s: s.result.objective_score),
    ("Reliability", "", lambda s: s.result.reliability),
    ("Throughput (tok/s)", "", lambda s: s.result.tokens_per_s),
    ("Efficiency (quality/W)", "", lambda s: s.quality_per_watt),
)
_HIGHLIGHT = "#1f6feb"  # the recommended-for-host model
_BASE = "#9aa7b4"


def render_comparison_chart(
    rec: Recommendation, out_path: Path, *, title: str | None = None
) -> Path | None:
    """Render the model-comparison chart to `out_path` (PNG). Returns the path, or None if matplotlib
    is unavailable. Bars are ordered by accuracy; the recommended-for-host model is highlighted."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _LOG.warning("[recommend] matplotlib missing; skipping chart (install the [viz] extra)")
        return None

    summaries = sorted(rec.summaries, key=lambda s: s.result.objective_score)
    labels = [_short(s.model) for s in summaries]
    recommended = rec.recommended_for_host.model
    colors = [_HIGHLIGHT if s.model == recommended else _BASE for s in summaries]

    fig, axes = plt.subplots(
        1,
        len(_PANELS),
        figsize=(3.2 * len(_PANELS) + 2.0, 0.6 * len(labels) + 2.0),
        sharey=True,
    )
    panel_axes = list(axes) if len(_PANELS) > 1 else [axes]
    for col, (ax, (panel_title, unit, accessor)) in enumerate(zip(panel_axes, _PANELS)):
        values = [float(accessor(s) or 0.0) for s in summaries]
        ax.barh(labels, values, color=colors)
        ax.set_title(panel_title, fontsize=10)
        if col == 0:
            ax.tick_params(axis="y", labelsize=8)  # labels only on the leftmost (shared) axis
        top = max(values) if values else 1.0
        for y, v in enumerate(values):
            ax.text(v + top * 0.02, y, f"{v:.3g}{unit}", va="center", fontsize=7)
        ax.set_xlim(0, top * 1.22 if top else 1.0)
        ax.margins(y=0.04)

    host = rec.host
    host_label = (
        f"{host.gpu_name or 'GPU'} ({host.tier_gb} GiB)"
        if host.detected
        else f"{host.tier_gb} GiB tier"
    )
    fig.suptitle(
        title
        or f"loc-lm-bench model comparison -- {host_label}  (blue = recommended for this host)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
