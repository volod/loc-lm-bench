"""The heavy half of the reranker bake-off: load one candidate and measure what holding it costs.

Everything that touches a GPU, a download, or repository-supplied modelling code lives here, behind
the `ScorerLoader` seam the lane calls -- which is what keeps the lane, the rows, the fit gate, and
the verdict testable with a fake cross-encoder.

Two measurements are taken around the load, and both feed a decision rather than a log line:

  - **resident footprint.** NVML before the weights are loaded and after the first real scoring
    call, because a lazy loader that has not run a batch has not yet allocated its activations.
    This is the number the fit gate reads: a reranker that does not fit beside the generator is not
    a slower option, it is not an option.
  - **load wall-clock.** A cold load is paid once per process, so it is reported apart from the
    per-query rerank latency instead of being folded into it.
"""

import logging
import time
from collections.abc import Callable

from llb.rag.rerank import CrossEncoderReranker
from llb.rag.rerank_bakeoff.families import resolve_convention
from llb.rag.rerank_bakeoff.models import (
    LoadedScorer,
    ScorerLoadError,
    ScorerLoader,
)

_LOG = logging.getLogger(__name__)

# sentence-transformers' own default predict batch; recorded in the report because a candidate's
# measured latency is a property of the batch it was scored in.
DEFAULT_BATCH_SIZE = 32

# `auto` keeps each card's declared `torch_dtype` (fp16 for several candidates), which is the
# precision an operator would actually deploy -- and therefore the footprint worth measuring.
DTYPE_AUTO = "auto"

# One short pair, scored right after the load, so the footprint reading includes the first
# allocation the model makes rather than only its weights.
_WARMUP_QUESTION = "warmup"
_WARMUP_TEXT = "warmup passage"

VramReader = Callable[[], int]


def _footprint(reader: VramReader | None, baseline: int | None) -> float | None:
    """Used VRAM above the pre-load baseline, in MB (None without a reader)."""
    if reader is None or baseline is None:
        return None
    return float(max(reader() - baseline, 0))


def cross_encoder_loader(
    *,
    device: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dtype: str = DTYPE_AUTO,
    vram_reader: VramReader | None = None,
) -> ScorerLoader:
    """Bind a loader that materializes one candidate per call and measures its footprint.

    The candidate's registered convention decides HOW it loads: `trust_remote_code` is passed only
    for a family that declares it (and only when the operator armed the roster screen), and the
    query-side instruction stays the model's own. Any failure -- missing extra, download error,
    CUDA OOM -- becomes a `ScorerLoadError`, so the roster continues and the report records why the
    row is absent.
    """

    def load(model: str) -> LoadedScorer:
        convention = resolve_convention(model)
        baseline = vram_reader() if vram_reader is not None else None
        scorer = CrossEncoderReranker(
            model,
            device=device,
            trust_remote_code=convention.trust_remote_code,
            model_kwargs={"torch_dtype": dtype},
            batch_size=batch_size,
        )
        started = time.perf_counter()
        try:
            scorer(_WARMUP_QUESTION, [_WARMUP_TEXT])
        except SystemExit as exc:  # the [rag] extra guard raises SystemExit by design
            raise ScorerLoadError(str(exc)) from exc
        except Exception as exc:
            raise ScorerLoadError(f"{type(exc).__name__}: {exc}") from exc
        load_seconds = time.perf_counter() - started
        resolved = getattr(getattr(scorer, "_model", None), "device", None)
        _LOG.info(
            "[compare-rerankers] loaded %s in %.1fs on %s", model, load_seconds, resolved or device
        )
        return LoadedScorer(
            scorer=scorer,
            device=str(resolved) if resolved is not None else device,
            load_seconds=load_seconds,
            vram_mb=_footprint(vram_reader, baseline),
            read_vram=lambda: _footprint(vram_reader, baseline),
            release=scorer.release,
        )

    return load
