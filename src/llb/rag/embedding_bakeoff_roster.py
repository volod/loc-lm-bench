"""Candidate screening for the EMBEDDER bake-off: which roster entries are allowed to be built.

The policy -- refuse an id with no declared convention, decline `trust_remote_code` unless the
operator opted in -- is shared with the reranker bake-off and lives in `llb.rag.candidate_screen`.
This module supplies the encoder half: the query/passage convention registry
(`llb.rag.embedding_families`) and the wording an operator sees when an encoder is refused.

Pure and dependency-free: no torch, no network, no store. The CLI screens once and passes the
survivors to `run_bakeoff`.
"""

from collections.abc import Sequence

from llb.rag.candidate_screen import (
    SkippedCandidate,
    UnregisteredCandidateError,  # noqa: F401  -- the CLI catches it through this lane
    screen_roster,
)
from llb.rag.embedding_families import is_registered, resolve_convention

_REGISTRY_MODULE = "llb.rag.embedding_families"


def screen_candidates(
    models: Sequence[str], *, allow_remote_code: bool = False
) -> tuple[list[str], list[SkippedCandidate]]:
    """Split the encoder roster into candidates to build and candidates skipped with a reason."""
    return screen_roster(
        models,
        resolve=resolve_convention,
        registered=is_registered,
        registry_module=_REGISTRY_MODULE,
        subject="an encoder",
        convention_label="query/passage convention",
        allow_remote_code=allow_remote_code,
    )
