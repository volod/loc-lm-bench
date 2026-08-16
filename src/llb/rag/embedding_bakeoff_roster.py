"""Candidate screening for the embedder bake-off: which roster entries are allowed to be built.

Two things can be wrong with a candidate BEFORE a single chunk is embedded, and both are silent if
nobody checks:

  - **No registered convention.** An unknown id is encoded with no instruction at all
    (`llb.rag.embedding_families`), so a retrieval-tuned encoder is scored under a format its card
    never documented and simply looks bad. A bake-off exists to rank encoders, so scoring one
    under a guessed format is the one failure it must not commit -- an unregistered id is REFUSED,
    not run.
  - **Repository-supplied modelling code.** `trust_remote_code` executes code downloaded with the
    weights. That is an operator's decision, so without the opt-in the candidate is SKIPPED with
    its reason recorded in the report -- the rest of the roster still ranks, and the report says
    which rows are missing and why rather than quietly shrinking.

Pure and dependency-free: no torch, no network, no store. The CLI screens once and passes the
survivors to `run_bakeoff`.
"""

from collections.abc import Sequence

from llb.rag.embedding_families import is_registered, resolve_convention
from llb.rag.embedding_bakeoff_models import SKIP_REMOTE_CODE, SkippedCandidate


class UnregisteredCandidateError(ValueError):
    """A bake-off candidate whose query/passage convention nobody declared."""


def screen_candidates(
    models: Sequence[str], *, allow_remote_code: bool = False
) -> tuple[list[str], list[SkippedCandidate]]:
    """Split the roster into candidates to build and candidates skipped with a recorded reason.

    Raises `UnregisteredCandidateError` on any id with no declared convention: that one is a
    measurement bug, not a policy choice, so it fails the run instead of quietly dropping a row.
    """
    unregistered = [model for model in models if not is_registered(model)]
    if unregistered:
        raise UnregisteredCandidateError(
            "no registered query/passage convention for: "
            + ", ".join(sorted(unregistered))
            + ". Scoring an encoder under a guessed format silently caps its recall -- add its "
            "documented convention to llb.rag.embedding_families first."
        )
    runnable: list[str] = []
    skipped: list[SkippedCandidate] = []
    for model in models:
        convention = resolve_convention(model)
        if convention.trust_remote_code and not allow_remote_code:
            skipped.append(
                {
                    "model": model,
                    "family": convention.family,
                    "reason": SKIP_REMOTE_CODE,
                    "detail": (
                        "needs trust_remote_code (runs repository-supplied modelling code); "
                        f"re-run with --allow-remote-code after reviewing {convention.source}"
                    ),
                }
            )
            continue
        runnable.append(model)
    return runnable, skipped
