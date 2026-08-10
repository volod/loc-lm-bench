"""Which published figures a set of invalidated cells retires -- keyed by study and cell id.

The pin gate already names every cell a constant change moves. The numbers those cells stand under
are prose in the docs until something ties each figure back to the cells and the run artifact it
came from. This module is that tie: every registered published value carries the cell ids its
committed aggregate measured at that depth, and a drift that invalidates any of those cells retires
the figure -- plus every derived figure that declares it as a source.

The resolution seam is the same `(artifact, field)` walk and committed aggregates the six design
values already use; this only adds the study/cell key the gate needs to join its re-run scope to the
figures an operator must restate.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from llb.bench.agentic_published_value_derivation import ValueKey, published_key
from llb.bench.agentic_published_value_derivation_graph import derivation_graph
from llb.bench.agentic_published_value_fixture import load_provenance_fixture
from llb.bench.agentic_published_value_pointer import read_field
from llb.bench.agentic_published_value_provenance import provenance_pair
from llb.bench.agentic_published_value_registry import (
    PUBLISHED_VALUE_DESIGNS,
    registered_design_path,
)
from llb.core.paths import PROJECT_ROOT

# The design field that names the cells a published figure rests on.
CELL_IDS = "cell_ids"


@dataclass(frozen=True, slots=True)
class PublishedFigure:
    """One published statement the docs quote, tied to the cells and artifact that measured it."""

    study: str
    depth: object
    form: str
    statement: str
    cell_ids: tuple[str, ...]
    artifact: str
    field: str

    def named(self) -> str:
        """The restate-scope line naming the figure and the cells that retire it."""
        cells = ", ".join(self.cell_ids)
        return (
            f"{self.study} depth {self.depth} {self.form}: {self.statement} "
            f"(cells {cells}; {self.artifact}#{self.field})"
        )


def published_figures_for_cells(
    invalidated: Sequence[Mapping[str, object]],
    *,
    design_root: Path = PROJECT_ROOT,
) -> tuple[PublishedFigure, ...]:
    """Every registered figure an invalidated cell retires, including derived consequences."""
    if not invalidated:
        return ()
    hit_keys = _keys_hit_by_cells(invalidated, design_root=design_root)
    if not hit_keys:
        return ()
    values = _registered_published_values(design_root)
    graph = derivation_graph(values)
    retired = set(hit_keys)
    for key in hit_keys:
        retired.update(graph.consequences_of(key))
    by_key = {published_key(value): value for value in values}
    return tuple(
        _figure(by_key[key])
        for key in sorted(retired, key=lambda item: (item.study_kind, item.depth, item.form))
        if key in by_key
    )


def validate_published_cell_ids(values: Sequence[Mapping[str, object]], *, root: Path) -> None:
    """Refuse a published value whose cell_ids are missing, stale, or not in its cited artifact.

    The cell list must be exactly the cells the committed aggregate carries at that depth -- no
    hand-narrowing, no extras -- and each id must resolve through the same pointer walk a figure
    field uses, so an artifact path the evidence does not carry fails here rather than at the gate.
    """
    fixture = load_provenance_fixture(root)
    for value in values:
        where = f"{value.get('study_kind')} depth {value.get('depth')} {value.get('form')}"
        artifact, _field = provenance_pair(value.get("provenance"), where=where)
        committed = fixture.get(artifact)
        if committed is None:
            raise ValueError(
                f"{where}: no committed copy of {artifact!r} exists, so the published value "
                f"cannot be resolved without the run -- add it with "
                f"make bench-agentic-published-provenance"
            )
        declared = _declared_cell_ids(value, where=where)
        measured = _cells_at_depth(committed.payload, int(cast(int, value["depth"])), where=where)
        if declared != measured:
            raise ValueError(
                f"{where}: cell_ids {list(declared)} must be exactly the cells the cited "
                f"artifact measures at this depth ({list(measured)})"
            )
        for cell_id in declared:
            read_field(
                committed.payload,
                f"cells[cell_id={cell_id}].cell_id",
                where=where,
            )


def _keys_hit_by_cells(
    invalidated: Sequence[Mapping[str, object]], *, design_root: Path
) -> list[ValueKey]:
    wanted = {
        (str(row["study_kind"]), str(row["cell_id"]))
        for row in invalidated
        if "study_kind" in row and "cell_id" in row
    }
    hits: list[ValueKey] = []
    for value in _registered_published_values(design_root):
        study = str(value["study_kind"])
        cells = cast(list[str], value.get(CELL_IDS, []))
        if any((study, cell_id) in wanted for cell_id in cells):
            hits.append(published_key(value))
    return hits


def _registered_published_values(design_root: Path) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for kind, design in sorted(PUBLISHED_VALUE_DESIGNS.items()):
        path = registered_design_path(kind, design, design_root)
        values.extend(design.published_values(path))
    return values


def _figure(value: Mapping[str, object]) -> PublishedFigure:
    where = f"{value.get('study_kind')} depth {value.get('depth')} {value.get('form')}"
    artifact, field = provenance_pair(value.get("provenance"), where=where)
    return PublishedFigure(
        study=str(value["study_kind"]),
        depth=value.get("depth"),
        form=str(value["form"]),
        statement=_statement(value),
        cell_ids=tuple(cast(list[str], value[CELL_IDS])),
        artifact=artifact,
        field=field,
    )


def _statement(value: Mapping[str, object]) -> str:
    band = value.get("published_band")
    if isinstance(band, list):
        return f"published band {band!r}"
    if "value" in value:
        return f"published value {value['value']!r}"
    return "published statement"


def _declared_cell_ids(value: Mapping[str, object], *, where: str) -> tuple[str, ...]:
    raw = value.get(CELL_IDS)
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"{where}: a published figure must declare non-empty string `{CELL_IDS}` naming the "
            "cells its cited artifact measured at this depth"
        )
    if not all(isinstance(item, str) and item for item in raw):
        raise ValueError(f"{where}: `{CELL_IDS}` must be a list of non-empty strings, got {raw!r}")
    if len(set(raw)) != len(raw):
        raise ValueError(f"{where}: `{CELL_IDS}` must not repeat a cell id, got {raw!r}")
    return tuple(cast(list[str], raw))


def _cells_at_depth(payload: Mapping[str, object], depth: int, *, where: str) -> tuple[str, ...]:
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"{where}: the cited artifact carries no cells list")
    return tuple(
        str(cast(dict[str, object], cell)["cell_id"])
        for cell in cells
        if isinstance(cell, dict) and cell.get("depth") == depth and "cell_id" in cell
    )


def cell_ids_from_aggregate(
    payload: Mapping[str, object], depth: int, *, where: str = "aggregate"
) -> tuple[str, ...]:
    """The cell ids one depth of a committed aggregate measures -- what a design must declare."""
    return _cells_at_depth(payload, depth, where=where)
