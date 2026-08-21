"""The typed comparison and blocking specification a linkage run is defined by.

This is the seam's contract, not Splink's: a `LinkageSpec` round-trips through `settings.json`,
so an identity decision can be re-read, diffed, and replayed without the library loaded. The
translation into Splink objects lives in `comparisons.py`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llb.core.contracts.common import JsonObject
from llb.linkage.constants import (
    COMPARISON_KINDS,
    DATE_METRICS,
    DEFAULT_DATE_FORMAT,
    DEFAULT_DATE_METRIC,
    DEFAULT_DUCKDB_THREADS,
    DEFAULT_EM_CONVERGENCE,
    DEFAULT_EM_MAX_ITERATIONS,
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_MAX_PAIRS,
    DEFAULT_RANDOM_MATCH_PROBABILITY,
    DEFAULT_SEED,
    DEFAULT_UNIQUE_ID_COLUMN,
    KINDS_NEEDING_THRESHOLDS,
    KINDS_WITH_INTEGER_THRESHOLDS,
    KINDS_WITH_POSITIVE_THRESHOLDS,
    KINDS_WITH_SCORE_THRESHOLDS,
    KIND_COSINE,
    MIN_COMPARISONS,
    RESERVED_COLUMNS,
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

    def validate(self) -> None:
        if not self.column:
            raise ValueError("a comparison needs a column name")
        if self.kind not in COMPARISON_KINDS:
            raise ValueError(
                f"unknown comparison kind {self.kind!r} for column {self.column!r}; "
                f"known kinds: {', '.join(COMPARISON_KINDS)}"
            )
        self._validate_thresholds()
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
        if len(set(self.thresholds)) != len(self.thresholds):
            raise ValueError(f"repeated thresholds on {self.column!r}: {list(self.thresholds)}")
        for kinds, is_bad, reason in _THRESHOLD_RULES:
            bad = [t for t in self.thresholds if self.kind in kinds and is_bad(t)]
            if bad:
                raise ValueError(f"{self.kind} thresholds on {self.column!r} {reason}; got {bad}")

    def payload(self) -> JsonObject:
        return {
            "column": self.column,
            "kind": self.kind,
            "thresholds": list(self.thresholds),
            "date_metric": self.date_metric,
            "date_format": self.date_format,
            "term_frequency": self.term_frequency,
            "dimension": self.dimension,
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
        )


@dataclass(frozen=True)
class BlockingRule:
    """One candidate-generation rule: records agreeing on every expression are compared."""

    expressions: tuple[str, ...]

    def validate(self) -> None:
        if not self.expressions or not all(e.strip() for e in self.expressions):
            raise ValueError("a blocking rule needs at least one non-empty expression")

    @property
    def label(self) -> str:
        return " AND ".join(self.expressions)

    def payload(self) -> JsonObject:
        return {"expressions": list(self.expressions)}

    @classmethod
    def from_payload(cls, payload: JsonObject | list[str] | str) -> "BlockingRule":
        if isinstance(payload, str):
            return cls((payload,))
        if isinstance(payload, list):
            return cls(tuple(str(e) for e in payload))
        return cls(tuple(str(e) for e in payload["expressions"]))


@dataclass(frozen=True)
class LinkageSpec:
    """A whole identity decision: what is compared, what is blocked, and where it is cut."""

    comparisons: tuple[ComparisonSpec, ...]
    blocking_rules: tuple[BlockingRule, ...]
    training_rules: tuple[BlockingRule, ...] = field(default_factory=tuple)
    unique_id_column: str = DEFAULT_UNIQUE_ID_COLUMN
    # Columns the record table carries but no comparison scores: provenance a reader needs beside
    # a proposed merge, and any column a blocking expression references without comparing.
    retain_columns: tuple[str, ...] = ()
    match_threshold: float = DEFAULT_MATCH_THRESHOLD
    max_pairs: int = DEFAULT_MAX_PAIRS
    seed: int = DEFAULT_SEED
    em_max_iterations: int = DEFAULT_EM_MAX_ITERATIONS
    em_convergence: float = DEFAULT_EM_CONVERGENCE
    random_match_probability: float = DEFAULT_RANDOM_MATCH_PROBABILITY
    duckdb_threads: int = DEFAULT_DUCKDB_THREADS

    def validate(self) -> None:
        """Reject a spec the method cannot price before any table is read."""
        if len(self.comparisons) < MIN_COMPARISONS:
            raise ValueError(
                f"record linkage needs at least {MIN_COMPARISONS} comparisons; a single-column "
                "spec is a one-feature threshold, which is what this seam replaces"
            )
        for comparison in self.comparisons:
            comparison.validate()
        columns = [c.column for c in self.comparisons]
        duplicates = sorted({c for c in columns if columns.count(c) > 1})
        if duplicates:
            raise ValueError(f"each column may carry one comparison; repeated: {duplicates}")
        if self.unique_id_column in columns:
            raise ValueError(
                f"the identifier column {self.unique_id_column!r} cannot also be compared"
            )
        self._validate_column_names(columns)
        if not self.blocking_rules:
            raise ValueError("a linkage run needs at least one blocking rule")
        for rule in (*self.blocking_rules, *self.training_rules):
            rule.validate()
        if not 0.0 < self.match_threshold <= 1.0:
            raise ValueError(f"match_threshold must be in (0, 1]; got {self.match_threshold}")
        if self.max_pairs <= 0:
            raise ValueError(f"max_pairs must be positive; got {self.max_pairs}")
        if self.duckdb_threads < 1:
            raise ValueError(f"duckdb_threads must be at least 1; got {self.duckdb_threads}")

    def _validate_column_names(self, columns: list[str]) -> None:
        """Refuse a column name Splink's clustering SQL introduces itself.

        The collision does not surface at fit time: it lands as an ambiguous-reference binder
        error inside the connected-components step, after the model has already trained. Naming
        it here costs one check and saves reading a generated SQL statement.
        """
        taken = sorted(
            {self.unique_id_column, *columns, *self.retain_columns} & set(RESERVED_COLUMNS)
        )
        if taken:
            raise ValueError(
                f"column name(s) {taken} are reserved by the clustering step; rename them in the "
                f"record table (reserved: {', '.join(RESERVED_COLUMNS)})"
            )

    @property
    def compared_columns(self) -> tuple[str, ...]:
        return tuple(c.column for c in self.comparisons)

    @property
    def em_rules(self) -> tuple[BlockingRule, ...]:
        """Blocking rules the expectation-maximisation passes train on.

        Defaults to the prediction rules: each pass holds one rule's columns fixed and learns the
        remaining comparisons' match parameters from the pairs it generates.
        """
        return self.training_rules or self.blocking_rules

    def payload(self) -> JsonObject:
        return {
            "comparisons": [c.payload() for c in self.comparisons],
            "blocking_rules": [r.payload() for r in self.blocking_rules],
            "training_rules": [r.payload() for r in self.training_rules],
            "unique_id_column": self.unique_id_column,
            "retain_columns": list(self.retain_columns),
            "match_threshold": self.match_threshold,
            "max_pairs": self.max_pairs,
            "seed": self.seed,
            "em_max_iterations": self.em_max_iterations,
            "em_convergence": self.em_convergence,
            "random_match_probability": self.random_match_probability,
            "duckdb_threads": self.duckdb_threads,
        }

    @classmethod
    def from_payload(cls, payload: JsonObject) -> "LinkageSpec":
        defaults = cls(comparisons=(), blocking_rules=())
        return cls(
            comparisons=tuple(
                ComparisonSpec.from_payload(c) for c in payload.get("comparisons", ())
            ),
            blocking_rules=tuple(
                BlockingRule.from_payload(r) for r in payload.get("blocking_rules", ())
            ),
            training_rules=tuple(
                BlockingRule.from_payload(r) for r in payload.get("training_rules", ())
            ),
            unique_id_column=str(payload.get("unique_id_column", defaults.unique_id_column)),
            retain_columns=tuple(str(c) for c in payload.get("retain_columns", ())),
            match_threshold=float(payload.get("match_threshold", defaults.match_threshold)),
            max_pairs=int(payload.get("max_pairs", defaults.max_pairs)),
            seed=int(payload.get("seed", defaults.seed)),
            em_max_iterations=int(payload.get("em_max_iterations", defaults.em_max_iterations)),
            em_convergence=float(payload.get("em_convergence", defaults.em_convergence)),
            random_match_probability=float(
                payload.get("random_match_probability", defaults.random_match_probability)
            ),
            duckdb_threads=int(payload.get("duckdb_threads", defaults.duckdb_threads)),
        )


def load_spec(payload: Any) -> LinkageSpec:
    """Parse and validate a specification payload in one step."""
    spec = LinkageSpec.from_payload(dict(payload))
    spec.validate()
    return spec
