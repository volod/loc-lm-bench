"""Roster screening shared by the model bake-offs: which candidates are allowed to be loaded.

Three things can be wrong with a candidate BEFORE a single pair is scored, and all three are silent
if nobody checks. They are the same three whether the candidate is an encoder or a reranker, so the
policy lives here once and each lane supplies its own convention registry:

  - **No registered convention.** An unknown id is run with no instruction at all, so a model whose
    card documents a query prefix (or a task instruction) is scored under a format nobody read and
    simply looks bad. A bake-off exists to RANK models, so scoring one under a guessed format is the
    single failure it must not commit -- an unregistered id is REFUSED, not run.
  - **Repository-supplied modelling code.** `trust_remote_code` executes code downloaded with the
    weights. That is an operator's decision, so without the opt-in the candidate is SKIPPED with its
    reason recorded in the report -- the rest of the roster still ranks, and the report says which
    rows are missing and why rather than quietly shrinking.
  - **A stack this interpreter is not.** Some repository code targets a transformers major the repo
    does not pin, and on the pinned stack such a candidate either raises at load or -- worse --
    loads and returns numbers that do not reproduce its own card. That is a PACKAGING fact about
    the candidate, so the row is SKIPPED here with the pin it needs and the legacy scoring pass
    that provides it (`llb.rag.model_stack`), instead of failing the run or vanishing.

Pure and dependency-free: no torch, no network, no model. A lane screens once and passes the
survivors to its run.
"""

from collections.abc import Sequence
from typing import Callable, Protocol

from typing_extensions import TypedDict

from llb.rag.model_stack import legacy_pass_hint

# Why a roster entry produced no row. This one is a policy decline, not a failure.
SKIP_REMOTE_CODE = "trust_remote_code_not_opted_in"
# ...and this one is a packaging fact about the interpreter, not about the candidate's quality.
SKIP_LEGACY_STACK = "legacy_transformers_required"


class SkippedCandidate(TypedDict):
    """A roster entry that produced no row, and why -- so the report shrinks visibly, not quietly."""

    model: str
    family: str
    reason: str
    detail: str


class Convention(Protocol):
    """What screening needs from a lane's convention record (see the lane's `families` module)."""

    @property
    def family(self) -> str: ...

    @property
    def source(self) -> str: ...

    @property
    def trust_remote_code(self) -> bool: ...

    @property
    def requires_transformers_major(self) -> int | None: ...


class UnregisteredCandidateError(ValueError):
    """A bake-off candidate whose input convention nobody declared."""


def screen_roster(
    models: Sequence[str],
    *,
    resolve: Callable[[str], Convention],
    registered: Callable[[str], bool],
    registry_module: str,
    subject: str,
    convention_label: str,
    allow_remote_code: bool = False,
    transformers_major: int | None = None,
) -> tuple[list[str], list[SkippedCandidate]]:
    """Split a roster into candidates to run and candidates skipped with a recorded reason.

    Raises `UnregisteredCandidateError` on any id with no declared convention: that one is a
    measurement bug, not a policy choice, so it fails the run instead of quietly dropping a row.
    `subject` / `convention_label` / `registry_module` only shape the message the operator acts on.

    `transformers_major` is the major THIS interpreter would load a candidate with; a candidate
    whose repository code declares a different one is skipped for the legacy pass. Passed in rather
    than read here so the screen stays pure and its verdicts stay testable on any host.
    """
    unregistered = [model for model in models if not registered(model)]
    if unregistered:
        raise UnregisteredCandidateError(
            f"no registered {convention_label} for: "
            + ", ".join(sorted(unregistered))
            + f". Scoring {subject} under a guessed format silently caps its measured quality -- "
            f"add its documented convention to {registry_module} first."
        )
    runnable: list[str] = []
    skipped: list[SkippedCandidate] = []
    for model in models:
        convention = resolve(model)
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
        required = convention.requires_transformers_major
        if (
            required is not None
            and transformers_major is not None
            and required != transformers_major
        ):
            skipped.append(
                {
                    "model": model,
                    "family": convention.family,
                    "reason": SKIP_LEGACY_STACK,
                    "detail": legacy_pass_hint(subject),
                }
            )
            continue
        runnable.append(model)
    return runnable, skipped
