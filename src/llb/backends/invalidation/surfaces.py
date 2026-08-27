"""Where a model identity is RECORDED, and how each of those places is read.

Three surfaces carry a measurement's model, and a swap invalidates all three at once:

  - the COMMITTED RUN AGGREGATES -- the verbatim analysis files the repo carries for every cited
    run, each declaring the model it held fixed;
  - the PUBLISHED VALUES -- the numbers a registered design states out of those runs, which is a
    separate list because a design can publish several values out of one aggregate and each of them
    is a claim an operator would have to restate;
  - the BASELINE TABLES in the delivered docs -- the rows a reader actually trusts.

Each surface is read into the same record, and a surface that cannot be read becomes a stated reason
rather than an exception. A report that lost a surface to one unreadable fixture would understate
the swap's cost in the exact direction that makes a swap look cheap, so the reading says which
surfaces answered and which did not, the same way the upstream currency report does per registry.

The two structured surfaces declare a model FIELD, so a value that surface holds and the register
cannot place is UNRESOLVED and reported. The doc surface SEARCHES cells instead: the delivered docs
publish embedder, reranker, and judge tables whose models are not roster entries at all, so a cell
that resolves to nothing there is not a gap, it is a table about something else.
"""

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Callable

from llb.backends.invalidation.doc_tables import read_doc_tree
from llb.backends.invalidation.identity import ModelIndex, ResolvedModel

COMMITTED_AGGREGATES = "committed-aggregates"
PUBLISHED_VALUES = "published-values"
BASELINE_TABLES = "baseline-tables"

# The delivered evidence tree. Baseline rows are published here; the guides and the spec cite them.
BASELINE_DOC_ROOT = Path("docs/impl/current")

_HELD_FIXED = "held_fixed"
_MODEL = "model"
_NO_MODEL_FIELD = f"declares no `{_HELD_FIXED}.{_MODEL}`"


@dataclass(frozen=True)
class MeasuredRecord:
    """One recorded measurement: where it lives, the model it names, and what that model resolves to.

    `resolved` is None only on a surface that declares a model field, which is what makes the field
    meaningful -- it says the evidence named a model and the register could not place it.
    """

    surface: str
    location: str
    recorded: str
    detail: str
    resolved: ResolvedModel | None = None

    def named(self) -> str:
        """One line an operator can act on: what to re-measure, and where it is published."""
        where = f"{self.location} ({self.detail})" if self.detail else self.location
        return f"{where}: `{self.recorded}`"


@dataclass(frozen=True)
class SurfaceReading:
    """What one evidence surface answered, or why it could not answer at all."""

    surface: str
    describe: str
    records: tuple[MeasuredRecord, ...] = ()
    error: str | None = None


def _held_model(payload: dict[str, object]) -> str:
    held = payload.get(_HELD_FIXED)
    return str(held.get(_MODEL, "")) if isinstance(held, dict) else ""


def read_committed_aggregates(root: Path, index: ModelIndex) -> SurfaceReading:
    """The model every committed run aggregate held fixed, read out of the pinned copies.

    Read through the provenance fixture rather than by globbing the tree, so the aggregates this
    surface reports are exactly the ones the repo pins and verifies -- a stray copy nothing cites is
    not evidence a swap invalidates, and a pin that no longer matches its bytes is a failure this
    surface should report rather than measure against.
    """
    from llb.bench.published_value.fixture import PROVENANCE_FIXTURE, load_provenance_fixture

    describe = f"run aggregates pinned by {PROVENANCE_FIXTURE}"
    try:
        committed = load_provenance_fixture(root)
    except (OSError, ValueError, KeyError) as exc:
        return SurfaceReading(COMMITTED_AGGREGATES, describe, error=str(exc))
    records = []
    for artifact, aggregate in sorted(committed.items()):
        recorded = _held_model(aggregate.payload)
        records.append(
            MeasuredRecord(
                surface=COMMITTED_AGGREGATES,
                location=artifact,
                recorded=recorded,
                detail=f"{_HELD_FIXED}.{_MODEL}" if recorded else _NO_MODEL_FIELD,
                resolved=index.resolve(recorded) if recorded else None,
            )
        )
    return SurfaceReading(COMMITTED_AGGREGATES, describe, tuple(records))


def _value_detail(value: dict[str, object]) -> str:
    """How one published value names itself, in the terms its own design publishes it under."""
    named = [f"{key}={value[key]}" for key in ("study_kind", "depth", "form") if value.get(key)]
    return " ".join(named) or "published value"


def read_published_values(root: Path, index: ModelIndex) -> SurfaceReading:
    """Every value a registered design publishes, tagged with the model that design held fixed.

    Listed per VALUE rather than per design: a design publishing six crossovers out of three runs
    leaves six statements an operator has to restate, and collapsing them to one design row would
    report the swap's cost as a third of what it is.
    """
    from llb.bench.published_value.registry import PUBLISHED_VALUE_DESIGNS, registered_design_path

    describe = f"{len(PUBLISHED_VALUE_DESIGNS)} registered published-value design(s)"
    records: list[MeasuredRecord] = []
    for kind, design in sorted(PUBLISHED_VALUE_DESIGNS.items()):
        try:
            path = registered_design_path(kind, design, root)
            recorded = _held_model(json.loads(path.read_text(encoding="utf-8")))
            values = design.published_values(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return SurfaceReading(
                PUBLISHED_VALUES, describe, tuple(records), error=f"{kind}: {exc}"
            )
        resolved = index.resolve(recorded) if recorded else None
        records.extend(
            MeasuredRecord(
                surface=PUBLISHED_VALUES,
                location=design.design_path,
                recorded=recorded,
                detail=_value_detail(dict(value)) if recorded else _NO_MODEL_FIELD,
                resolved=resolved,
            )
            for value in values
        )
    return SurfaceReading(PUBLISHED_VALUES, describe, tuple(records))


def read_baseline_tables(root: Path, index: ModelIndex) -> SurfaceReading:
    """Every delivered-doc table row whose model column names a roster entry.

    Rows are deduplicated per resolved entry: the throughput table names the logical model AND the
    served artifact of the same row, and reporting one row twice would inflate a swap's cost with
    work that is one edit.
    """
    tree = root / BASELINE_DOC_ROOT
    describe = f"model-column tables under {BASELINE_DOC_ROOT}"
    if not tree.is_dir():
        return SurfaceReading(
            BASELINE_TABLES, describe, error=f"{BASELINE_DOC_ROOT} does not exist"
        )
    seen: set[tuple[str, str]] = set()
    records: list[MeasuredRecord] = []
    for cell in read_doc_tree(tree):
        resolved = index.resolve(cell.identity)
        location = f"{cell.path.relative_to(root).as_posix()}:{cell.line}"
        if resolved is None or (location, resolved.model_name) in seen:
            continue
        seen.add((location, resolved.model_name))
        records.append(
            MeasuredRecord(
                surface=BASELINE_TABLES,
                location=location,
                recorded=cell.identity,
                detail=f"`{cell.column}` column of the table at line {cell.header_line}",
                resolved=resolved,
            )
        )
    return SurfaceReading(BASELINE_TABLES, describe, tuple(records))


SurfaceReader = Callable[[Path, ModelIndex], SurfaceReading]

# The surfaces a swap is costed against, in the order the report prints them. A new place that
# records a model identity is registered HERE, so "what a swap invalidates" has one answer rather
# than one per caller.
EVIDENCE_SURFACES: tuple[SurfaceReader, ...] = (
    read_committed_aggregates,
    read_published_values,
    read_baseline_tables,
)


def read_evidence(root: Path, index: ModelIndex) -> tuple[SurfaceReading, ...]:
    """Read every registered evidence surface, keeping a surface's failure as its own reason."""
    return tuple(read(root, index) for read in EVIDENCE_SURFACES)
