"""Candidate screening for the RERANKER bake-off: which roster entries are allowed to be loaded.

The policy is the shared one (`llb.rag.encoders.candidate_screen`); this module supplies the reranker half:
the input-convention registry (`llb.rag.rerank_bakeoff.families`) and the wording an operator sees.
The reranker-OFF row is never screened -- it downloads nothing and runs no code.
"""

from collections.abc import Sequence

from llb.rag.encoders.candidate_screen import SkippedCandidate, screen_roster
from llb.rag.rerank_bakeoff.families import is_registered, resolve_convention
from llb.rag.rerank_bakeoff.models import ROW_NO_RERANK

_REGISTRY_MODULE = "llb.rag.rerank_bakeoff.families"


def screen_rerankers(
    models: Sequence[str],
    *,
    allow_remote_code: bool = False,
    transformers_major: int | None = None,
) -> tuple[list[str], list[SkippedCandidate]]:
    """Split the reranker roster into candidates to load and candidates skipped with a reason.

    A `none` entry is dropped from the roster rather than screened: the reranker-off row is added by
    the lane itself, so an operator who spells it out gets the row once, not twice.
    """
    requested = [model for model in models if model != ROW_NO_RERANK]
    return screen_roster(
        requested,
        resolve=resolve_convention,
        registered=is_registered,
        registry_module=_REGISTRY_MODULE,
        subject="a reranker",
        convention_label="input convention",
        allow_remote_code=allow_remote_code,
        transformers_major=transformers_major,
    )
