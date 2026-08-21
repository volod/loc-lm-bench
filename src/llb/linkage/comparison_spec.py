"""One compared field: which column, which similarity measure, and at which cut points.

Split from `spec.py` at the class seam. This half describes ONE field's agreement ladder and the
rules a bad ladder is refused by; `spec.py` describes a whole identity decision built out of
several of them.
"""

from collections.abc import Callable
from dataclasses import dataclass

from llb.core.contracts.common import JsonObject
from llb.linkage.constants import (
    COMPARISON_KINDS,
    DATE_METRICS,
    DEFAULT_DATE_FORMAT,
    DEFAULT_DATE_METRIC,
    KINDS_NEEDING_THRESHOLDS,
    KINDS_WITH_CONTAINMENT,
    KINDS_WITH_INTEGER_THRESHOLDS,
    KINDS_WITH_POSITIVE_THRESHOLDS,
    KINDS_WITH_SCORE_THRESHOLDS,
    KIND_COSINE,
)


# What a BAD threshold looks like, per kind, and the sentence that explains it. A table rather
# than a chain of ifs so adding a kind means adding a row, not editing a validator.
_THRESHOLD_RULES: tuple[tuple[tuple[str, ...], Callable[[float], bool], str], ...] = (
    (
        KINDS_WITH_SCORE_THRESHOLDS,
        lambda value: not 0.0 < value <= 1.0,
        "are similarity scores in (0, 1]",
    ),
    (
        KINDS_WITH_INTEGER_THRESHOLDS,
        lambda value: value != int(value) or value < 0,
        "are non-negative whole numbers",
    ),
    (
        KINDS_WITH_POSITIVE_THRESHOLDS,
        lambda value: value == 0,
        "cannot be zero, which repeats the exact-match level the ladder already carries",
    ),
)


@dataclass(frozen=True)
class ComparisonSpec:
    """One field's agreement ladder: which column, which similarity, at which cut points."""

    column: str
    kind: str
    thresholds: tuple[float, ...] = ()
    date_metric: str = DEFAULT_DATE_METRIC
    date_format: str = DEFAULT_DATE_FORMAT
    term_frequency: bool = False
    dimension: int = 0  # cosine only: the embedding width the column is declared at
    # `set_overlap` only: the SECOND ladder of the same set pair. Mutual overlap (Jaccard) and
    # one-sided coverage (containment) measure different relations between two sets -- near-identity
    # against one set sitting inside the other -- and a short set inside a long one has LOW Jaccard
    # by construction, so one ladder cannot express both. They share a column because they read the
    # same two sets, and the levels are emitted mutual-first, which is the order the caller's own
    # decision rule already applies.
    containment_thresholds: tuple[float, ...] = ()

    def validate(self) -> None:
        if not self.column:
            raise ValueError("a comparison needs a column name")
        if self.kind not in COMPARISON_KINDS:
            raise ValueError(
                f"unknown comparison kind {self.kind!r} for column {self.column!r}; "
                f"known kinds: {', '.join(COMPARISON_KINDS)}"
            )
        self._validate_thresholds()
        if self.containment_thresholds and self.kind not in KINDS_WITH_CONTAINMENT:
            raise ValueError(
                f"containment thresholds are only defined for {', '.join(KINDS_WITH_CONTAINMENT)} "
                f"comparisons; column {self.column!r} is {self.kind!r}"
            )
        if self.kind == KIND_COSINE and self.dimension < 1:
            raise ValueError(
                f"a cosine comparison on {self.column!r} must declare its embedding dimension: "
                "the similarity is defined on fixed-width vectors, so the column type needs it"
            )
        if self.date_metric not in DATE_METRICS:
            raise ValueError(
                f"unknown date metric {self.date_metric!r} for column {self.column!r}; "
                f"known metrics: {', '.join(DATE_METRICS)}"
            )

    def _validate_thresholds(self) -> None:
        if self.kind in KINDS_NEEDING_THRESHOLDS and not self.thresholds:
            raise ValueError(
                f"comparison kind {self.kind!r} on column {self.column!r} needs at least one "
                "threshold; without one it has no levels to score"
            )
        for name, values in (
            ("thresholds", self.thresholds),
            ("containment thresholds", self.containment_thresholds),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"repeated {name} on {self.column!r}: {list(values)}")
            for kinds, is_bad, reason in _THRESHOLD_RULES:
                bad = [t for t in values if self.kind in kinds and is_bad(t)]
                if bad:
                    raise ValueError(f"{self.kind} {name} on {self.column!r} {reason}; got {bad}")

    def payload(self) -> JsonObject:
        return {
            "column": self.column,
            "kind": self.kind,
            "thresholds": list(self.thresholds),
            "date_metric": self.date_metric,
            "date_format": self.date_format,
            "term_frequency": self.term_frequency,
            "dimension": self.dimension,
            "containment_thresholds": list(self.containment_thresholds),
        }

    @classmethod
    def from_payload(cls, payload: JsonObject) -> "ComparisonSpec":
        return cls(
            column=str(payload["column"]),
            kind=str(payload["kind"]),
            thresholds=tuple(float(t) for t in payload.get("thresholds", ())),
            date_metric=str(payload.get("date_metric", DEFAULT_DATE_METRIC)),
            date_format=str(payload.get("date_format", DEFAULT_DATE_FORMAT)),
            term_frequency=bool(payload.get("term_frequency", False)),
            dimension=int(payload.get("dimension", 0)),
            containment_thresholds=tuple(
                float(t) for t in payload.get("containment_thresholds", ())
            ),
        )
