"""Does this candidate reproduce its own model card before it is allowed into a ranking?

`Alibaba-NLP/gte-multilingual-base` is the case that makes this gate necessary: on the pinned
transformers it LOADS, encodes without raising, and returns embeddings whose similarities do not
match the ones its card publishes. A bake-off that scores it publishes a number about a broken
load and calls it a model. Loading is therefore not evidence that a candidate can be ranked --
reproducing its card is.

The check is declared per candidate, and there are two kinds of declaration because cards differ:

  - **published values.** The card prints reference similarities (or reference scores) for named
    inputs. Parity is reproducing them within a stated tolerance, and the tolerance is a
    fp/device-noise budget, not a fudge factor -- a half-precision checkpoint moves the third
    decimal, a wrong load moves the first.
  - **the card's own reference implementation.** Some cards publish a runnable snippet but no
    numbers. Then the reference is what that snippet returns HERE, computed at check time, and
    parity is agreement between the card's documented path and the path the bake-off actually
    scores through. It catches the same failure -- a load that runs and is wrong -- without
    inventing a threshold nobody published.

A card publishes its reference in the space its snippet prints: raw logits for a cross-encoder the
card calls through transformers, sigmoid probabilities for one it calls through its own helper,
similarities times 100 for one card that scales them. So a reference declares the `transform` and
`scale` that carry its published numbers into the space the bake-off's loaded model returns, and
the comparison happens there -- never by rewriting the published values.

Pure and dependency-free: this module compares numbers. Producing the observed ones is the lane's
job (`llb.rag.encoders.cards`, `llb.rag.rerank_bakeoff.cards`).
"""

import math
from dataclasses import dataclass

from typing_extensions import TypedDict

from llb.rag.encoders.candidate_screen import SkippedCandidate

# How a card's published numbers map into the space a loaded model returns.
TRANSFORM_IDENTITY = "identity"
# A card that prints raw classifier logits for a model sentence-transformers scores through a
# sigmoid activation: the published logit is compared after the same squashing.
TRANSFORM_SIGMOID = "sigmoid"

# What a reference declares as its source of truth.
MODE_PUBLISHED_VALUES = "published_values"
MODE_REFERENCE_IMPLEMENTATION = "reference_implementation"

# The verdict a candidate carries into the report.
STATUS_REPRODUCED = "reproduced"
STATUS_MISMATCH = "mismatch"
STATUS_UNPUBLISHED = "no_reference_declared"
STATUS_ERROR = "probe_failed"

# Why a roster entry produced no row. Distinct from a load failure: this candidate RAN.
SKIP_CARD_PARITY = "card_parity_mismatch"

# fp32-vs-fp16 noise on a normalized cosine is a few units in the third decimal; a broken load
# moves the first. The default sits between them so neither is a judgement call per candidate.
DEFAULT_TOLERANCE = 0.01


@dataclass(frozen=True)
class CardExpectation:
    """The published half of a card reference: the numbers and the space they are printed in.

    `values` is empty for a card that publishes a runnable snippet but no numbers; the lane then
    computes the reference from that snippet and the mode is `reference_implementation`.
    `scale` divides the published numbers (one card prints similarities times 100), and
    `transform` squashes them into the space the loaded model returns.
    """

    values: tuple[float, ...] = ()
    transform: str = TRANSFORM_IDENTITY
    scale: float = 1.0
    tolerance: float = DEFAULT_TOLERANCE

    @property
    def mode(self) -> str:
        """Which kind of reference this is."""
        return MODE_PUBLISHED_VALUES if self.values else MODE_REFERENCE_IMPLEMENTATION


class CardParityResult(TypedDict):
    """One candidate's card-parity verdict, carried on its row or on its skip entry."""

    model: str
    status: str
    mode: str
    source: str
    tolerance: float
    expected: list[float]
    observed: list[float]
    max_abs_diff: float | None
    detail: str


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def expected_in_model_space(expectation: CardExpectation) -> tuple[float, ...]:
    """The card's published numbers moved into the space the loaded model returns."""
    scaled = tuple(value / expectation.scale for value in expectation.values)
    if expectation.transform == TRANSFORM_SIGMOID:
        return tuple(_sigmoid(value) for value in scaled)
    return scaled


def compare_to_card(
    model: str,
    source: str,
    expectation: CardExpectation,
    observed: tuple[float, ...],
    *,
    expected: tuple[float, ...] | None = None,
) -> CardParityResult:
    """Verdict for one candidate: did it reproduce its card's reference behavior?

    `expected` overrides the published values for a `reference_implementation` reference, where the
    lane computed the reference by running the card's own snippet. A length mismatch is a mismatch,
    not a crash: it means the probe and the reference are not describing the same call.
    """
    reference = expected if expected is not None else expected_in_model_space(expectation)
    if len(reference) != len(observed):
        return {
            "model": model,
            "status": STATUS_MISMATCH,
            "mode": expectation.mode,
            "source": source,
            "tolerance": expectation.tolerance,
            "expected": list(reference),
            "observed": list(observed),
            "max_abs_diff": None,
            "detail": (
                f"card reference has {len(reference)} values, the probe returned {len(observed)}"
            ),
        }
    diffs = [abs(a - b) for a, b in zip(reference, observed)]
    worst = max(diffs) if diffs else 0.0
    reproduced = worst <= expectation.tolerance
    return {
        "model": model,
        "status": STATUS_REPRODUCED if reproduced else STATUS_MISMATCH,
        "mode": expectation.mode,
        "source": source,
        "tolerance": expectation.tolerance,
        "expected": [round(value, 6) for value in reference],
        "observed": [round(value, 6) for value in observed],
        "max_abs_diff": round(worst, 6),
        "detail": (
            f"reproduced within {expectation.tolerance} (worst |delta| {worst:.4f})"
            if reproduced
            else (
                f"does NOT reproduce {source}: worst |delta| {worst:.4f} exceeds the "
                f"{expectation.tolerance} tolerance"
            )
        ),
    }


def unpublished_result(model: str) -> CardParityResult:
    """The verdict for a candidate whose card declares no reference this lane can check.

    Recorded rather than silently passed: "nobody checked" and "it reproduces" are different facts
    about a row, and only one of them is evidence.
    """
    return {
        "model": model,
        "status": STATUS_UNPUBLISHED,
        "mode": MODE_PUBLISHED_VALUES,
        "source": "",
        "tolerance": 0.0,
        "expected": [],
        "observed": [],
        "max_abs_diff": None,
        "detail": "no card reference is declared for this id; the row is scored ungated",
    }


def probe_error_result(model: str, source: str, detail: str) -> CardParityResult:
    """The verdict for a candidate whose card probe raised before producing a number."""
    return {
        "model": model,
        "status": STATUS_ERROR,
        "mode": MODE_PUBLISHED_VALUES,
        "source": source,
        "tolerance": 0.0,
        "expected": [],
        "observed": [],
        "max_abs_diff": None,
        "detail": detail,
    }


def blocks_scoring(result: CardParityResult) -> bool:
    """Whether this verdict must keep the candidate out of the ranking."""
    return result["status"] in (STATUS_MISMATCH, STATUS_ERROR)


def parity_skip_row(result: CardParityResult, family: str) -> SkippedCandidate:
    """A recorded not-scored entry for a candidate that ran and failed its card."""
    return {
        "model": result["model"],
        "family": family,
        "reason": SKIP_CARD_PARITY,
        "detail": result["detail"],
    }
