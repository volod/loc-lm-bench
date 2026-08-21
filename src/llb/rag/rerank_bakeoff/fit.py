"""Can this candidate run HERE, and what did holding it cost?

The reranker bake-off is run on the host the stack is served from, so "which reranker is best" is
never the whole question: a cross-encoder shares a device with a resident generator, and one that
does not fit is not a slower option -- it is not an option. This module owns that side of the lane:
the VRAM budget arithmetic, the load attempt, and the recorded reason a candidate produced no row.

Every refusal here carries a MEASUREMENT. A candidate skipped for headroom names the footprint that
decided it and the budget it was measured against; a candidate that could not load names what the
host said. A report that simply had fewer rows would be indistinguishable from one where a model
lost.
"""

import logging

from llb.rag.encoders.candidate_screen import SkippedCandidate
from llb.rag.rerank_bakeoff.families import resolve_convention
from llb.rag.rerank_bakeoff.models import (
    SKIP_LOAD_FAILED,
    SKIP_NO_HEADROOM,
    LoadedScorer,
    ScorerLoadError,
    ScorerLoader,
    VramHeadroom,
)

_LOG = logging.getLogger(__name__)


def fit_verdict(vram_mb: float | None, headroom: VramHeadroom | None) -> bool | None:
    """Does a measured footprint fit the declared budget? None when either side is unknown.

    An undeclared generator is NOT a pass: the footprint is reported and the gate simply does not
    run, because a measurement without a budget is not a verdict.
    """
    if headroom is None or headroom["headroom_mb"] is None or vram_mb is None:
        return None
    return vram_mb <= headroom["headroom_mb"]


def skip_row(model: str, reason: str, detail: str) -> SkippedCandidate:
    """One recorded not-scored entry, resolved to the candidate's declared family."""
    return {
        "model": model,
        "family": resolve_convention(model).family,
        "reason": reason,
        "detail": detail,
    }


def load_candidate(
    model: str, load: ScorerLoader, headroom: VramHeadroom | None
) -> tuple[LoadedScorer | None, SkippedCandidate | None]:
    """Load one candidate, or return the recorded reason it produced no row.

    Both refusals carry a MEASUREMENT: a candidate the host cannot hold beside the generator is
    reported with the footprint that decided it, never silently omitted.
    """
    try:
        loaded = load(model)
    except ScorerLoadError as exc:
        _LOG.warning("[compare-rerankers] %s did not load: %s", model, exc)
        return None, skip_row(model, SKIP_LOAD_FAILED, str(exc))
    if fit_verdict(loaded.vram_mb, headroom) is False:
        loaded.release()
        assert headroom is not None and headroom["headroom_mb"] is not None  # narrowed by the gate
        return None, skip_row(
            model,
            SKIP_NO_HEADROOM,
            f"resident footprint {loaded.vram_mb:.0f} MB exceeds the {headroom['headroom_mb']:.0f} "
            "MB left beside the declared generator residency",
        )
    return loaded, None


def peak_footprint(loaded: LoadedScorer) -> float | None:
    """The larger of the load-time footprint and the post-pass reading (None without a GPU)."""
    after = loaded.read_vram()
    measured = [value for value in (loaded.vram_mb, after) if value is not None]
    return max(measured) if measured else None
