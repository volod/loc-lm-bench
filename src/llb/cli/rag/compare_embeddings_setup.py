"""Pre-run resolution for `compare-embeddings`: the candidate roster and the run's paths.

Both are decided before any store is built, and both can end the run with an exit code rather than
a traceback, so they are kept out of the command module -- which is left as the flag declaration
plus the wiring that scores the roster.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer

from llb.core import env

if TYPE_CHECKING:
    from llb.rag.encoders.candidate_screen import SkippedCandidate


def resolve_roster(
    models: str, allow_remote_code: bool, dtype: str
) -> tuple[list[str], list["SkippedCandidate"]]:
    """Screen the requested roster into candidates to build plus visibly declined entries.

    An id with no declared query/passage convention exits the run (scoring it would guess a format
    and understate the encoder); a candidate needing `trust_remote_code` is declined unless the
    operator opted in, and a candidate whose repository code targets a transformers major this
    interpreter is not lands in the legacy pass (`llb.rag.encoders.model_stack`). Both the opt-in and the
    declared precision are exported process-wide -- like `LLB_EMBED_DEVICE`, because the store
    build, the lazy reload behind `retrieve()`, the card-parity probe, and the throughput profiler
    each construct their own `Embedder` and all four must agree.
    """
    from llb.rag.embedding_bakeoff.models import DEFAULT_LOCAL_CANDIDATES
    from llb.rag.embedding_bakeoff.roster import UnregisteredCandidateError, screen_candidates
    from llb.rag.encoders.precision import DTYPE_AUTO, UnsupportedDtypeError, normalize_dtype
    from llb.rag.encoders.model_stack import installed_transformers_major

    roster = [m.strip() for m in models.split(",") if m.strip()] or DEFAULT_LOCAL_CANDIDATES
    try:
        resolved_dtype = normalize_dtype(dtype)
    except UnsupportedDtypeError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    try:
        local_models, skipped = screen_candidates(
            roster,
            allow_remote_code=allow_remote_code,
            transformers_major=installed_transformers_major(),
        )
    except UnregisteredCandidateError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    for row in skipped:
        typer.echo(f"[compare-embeddings] skipping {row['model']}: {row['detail']}", err=True)
    if allow_remote_code:
        os.environ[env.LLB_TRUST_REMOTE_CODE] = "1"
    if resolved_dtype != DTYPE_AUTO:
        os.environ[env.LLB_EMBED_DTYPE] = resolved_dtype
    return local_models, skipped


def prepared_power_plan(
    report_path: Path,
    *,
    power_reference: Optional[Path],
    candidate: Optional[str],
    metric: str,
    minimum_detectable_delta: Optional[float],
    target_power: float,
    confidence: float,
    planned_n: int,
) -> Any:
    """The paired-power contract this item set is priced against, or `None` when none was asked for.

    A malformed contract exits the run here rather than after every candidate store has been built.
    """
    from llb.rag.embedding_bakeoff.power import prepare_embedding_power

    try:
        return prepare_embedding_power(
            power_reference,
            candidate=candidate,
            metric=metric,
            minimum_detectable_delta=minimum_detectable_delta,
            target_power=target_power,
            confidence=confidence,
            planned_n=planned_n,
            plan_path=report_path.parent / "power-plan.json",
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
