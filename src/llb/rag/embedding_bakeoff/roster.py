"""Candidate screening for the EMBEDDER bake-off: which roster entries are allowed to be built.

The policy -- refuse an id with no declared convention, decline `trust_remote_code` unless the
operator opted in -- is shared with the reranker bake-off and lives in `llb.rag.encoders.candidate_screen`.
This module supplies the encoder half: the query/passage convention registry
(`llb.rag.encoders.families`), read through `llb.rag.encoders.tuned` so a locally fine-tuned
directory is screened under its BASE model's convention, plus the wording an operator sees when an
encoder is refused.

Pure and dependency-free: no torch, no network, no store. The CLI screens once and passes the
survivors to `run_bakeoff`.
"""

from collections.abc import Sequence

from llb.rag.encoders.candidate_screen import (
    SkippedCandidate,
    UnregisteredCandidateError,  # noqa: F401  -- the CLI catches it through this lane
    screen_roster,
)
from llb.rag.encoders.tuned import convention_registered, resolved_convention

_REGISTRY_MODULE = "llb.rag.encoders.families"


def screen_candidates(
    models: Sequence[str],
    *,
    allow_remote_code: bool = False,
    transformers_major: int | None = None,
) -> tuple[list[str], list[SkippedCandidate]]:
    """Split the encoder roster into candidates to build and candidates skipped with a reason."""
    return screen_roster(
        models,
        resolve=resolved_convention,
        registered=convention_registered,
        registry_module=_REGISTRY_MODULE,
        subject="an encoder",
        convention_label="query/passage convention",
        allow_remote_code=allow_remote_code,
        transformers_major=transformers_major,
    )
