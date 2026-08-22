"""The blocking rules and the whole identity decision a linkage run is defined by.

This is the seam's contract, not Splink's: a `LinkageSpec` round-trips through `settings.json`,
so an identity decision can be re-read, diffed, and replayed without the library loaded. One
field's agreement ladder lives in `comparison_spec.py`; the translation into Splink objects lives
in `comparisons.py`.
"""

from dataclasses import dataclass, field
from typing import Any

from llb.core.contracts.common import JsonObject
from llb.linkage.comparison_spec import ComparisonSpec
from llb.linkage.constants import (
    DEFAULT_DUCKDB_THREADS,
    DEFAULT_EM_CONVERGENCE,
    DEFAULT_EM_MAX_ITERATIONS,
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_MAX_PAIRS,
    DEFAULT_MIN_LEVEL_PROBABILITY,
    DEFAULT_RANDOM_MATCH_PROBABILITY,
    DEFAULT_SEED,
    DEFAULT_UNIQUE_ID_COLUMN,
    MIN_COMPARISONS,
    RESERVED_COLUMNS,
)


@dataclass(frozen=True)
class BlockingRule:
    """One candidate-generation rule: records agreeing on every expression are compared."""

    expressions: tuple[str, ...]
    # Array columns the rule EXPLODES: two records are compared when they share at least one
    # element, rather than when a whole column is equal. It is what lets an inverted index over set
    # elements -- the blocking a caller already computes in Python -- be the rule Splink runs, so
    # the candidate list is the caller's, not a second one that has to be kept in step with it.
    arrays_to_explode: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.expressions or not all(e.strip() for e in self.expressions):
            raise ValueError("a blocking rule needs at least one non-empty expression")
        if any(not column.strip() for column in self.arrays_to_explode):
            raise ValueError(f"blocking rule {self.label!r} names an empty exploded column")

    @property
    def explodes(self) -> bool:
        return bool(self.arrays_to_explode)

    @property
    def label(self) -> str:
        joined = " AND ".join(self.expressions)
        return f"{joined} (exploded)" if self.explodes else joined

    def payload(self) -> JsonObject:
        return {
            "expressions": list(self.expressions),
            "arrays_to_explode": list(self.arrays_to_explode),
        }

    @classmethod
    def from_payload(cls, payload: JsonObject | list[str] | str) -> "BlockingRule":
        if isinstance(payload, str):
            return cls((payload,))
        if isinstance(payload, list):
            return cls(tuple(str(e) for e in payload))
        return cls(
            tuple(str(e) for e in payload["expressions"]),
            tuple(str(e) for e in payload.get("arrays_to_explode", ())),
        )


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
    # Pseudo-count floor applied to every fitted m and u before scoring; see the constant.
    min_level_probability: float = DEFAULT_MIN_LEVEL_PROBABILITY
    # Whether every compared column's two values ride along in each scored pair row. On for
    # inspection, and worth turning OFF for a comparison whose column is a large set: the pair table
    # would then carry two whole element sets per row, which is the comparison's INPUT rather than
    # provenance a reader needs beside a proposed merge. The published pair payload is the same
    # either way -- it is built from the level agreements, never from the matching columns.
    retain_matching_columns: bool = True

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
        self._validate_rules()
        self._validate_knobs()

    def _validate_rules(self) -> None:
        """Every rule is well formed, and no EXPLODING rule is asked to train a model."""
        if not self.blocking_rules:
            raise ValueError("a linkage run needs at least one blocking rule")
        for rule in (*self.blocking_rules, *self.training_rules):
            rule.validate()
        exploding = [rule.label for rule in self.training_rules if rule.explodes]
        if exploding:
            raise ValueError(
                f"an exploding blocking rule cannot train a model: {exploding}. Splink refuses it "
                "for expectation-maximisation, and it fixes no comparison to train the others "
                "against -- give `training_rules` an ordinary rule instead"
            )

    def _validate_knobs(self) -> None:
        """The numeric settings, each in the range its meaning has."""
        if not 0.0 < self.match_threshold <= 1.0:
            raise ValueError(f"match_threshold must be in (0, 1]; got {self.match_threshold}")
        if self.max_pairs <= 0:
            raise ValueError(f"max_pairs must be positive; got {self.max_pairs}")
        if self.duckdb_threads < 1:
            raise ValueError(f"duckdb_threads must be at least 1; got {self.duckdb_threads}")
        if not 0.0 <= self.min_level_probability < 0.5:
            raise ValueError(
                "min_level_probability is a pseudo-count floor in [0, 0.5); got "
                f"{self.min_level_probability}"
            )

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
    def exploded_columns(self) -> tuple[str, ...]:
        """Every array column a blocking rule explodes, in first-named order.

        The record table has to materialise these as arrays whether or not a comparison scores
        them, so the DDL reads them from here rather than from a second declaration that could
        disagree with the rules.
        """
        seen: list[str] = []
        for rule in self.blocking_rules:
            seen.extend(c for c in rule.arrays_to_explode if c not in seen)
        return tuple(seen)

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
            "min_level_probability": self.min_level_probability,
            "retain_matching_columns": self.retain_matching_columns,
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
            min_level_probability=float(
                payload.get("min_level_probability", defaults.min_level_probability)
            ),
            retain_matching_columns=bool(
                payload.get("retain_matching_columns", defaults.retain_matching_columns)
            ),
        )


def load_spec(payload: Any) -> LinkageSpec:
    """Parse and validate a specification payload in one step."""
    spec = LinkageSpec.from_payload(dict(payload))
    spec.validate()
    return spec
