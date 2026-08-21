"""Published values whose registered arithmetic depends on moved shipped policy fields."""

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from llb.bench.published_value.operations.registry import OPERATION, registered_operation
from llb.bench.published_value.registry import (
    PUBLISHED_VALUE_DESIGNS,
    registered_design_path,
)


@dataclass(frozen=True, slots=True)
class PolicyAffectedPublishedValue:
    """One published statement whose operation declares a moved shipped-policy dependency."""

    design: str
    study: str
    depth: object
    form: str
    operation: str
    fields: tuple[str, ...]
    statement: str

    def named(self) -> str:
        """The re-derivation scope line an operator needs when a policy pin moves."""
        fields = ", ".join(f"`{field}`" for field in self.fields)
        return (
            f"{self.study} depth {self.depth} {self.form}: {self.statement} via "
            f"`{self.operation}` ({fields})"
        )


def policy_affected_published_values(
    design_root: Path, fields: Collection[str]
) -> tuple[PolicyAffectedPublishedValue, ...]:
    """Walk registered designs and name values whose operation declares any moved field."""
    moved = set(fields)
    affected: list[PolicyAffectedPublishedValue] = []
    for kind, design in sorted(PUBLISHED_VALUE_DESIGNS.items()):
        path = registered_design_path(kind, design, design_root)
        for value in design.published_values(path):
            named_operation = value.get(OPERATION)
            if named_operation is None:
                continue
            where = f"{value.get('study_kind')} depth {value.get('depth')} {value.get('form')}"
            operation = registered_operation(named_operation, where=where)
            dependencies = tuple(field for field in operation.policy_fields if field in moved)
            if dependencies:
                affected.append(
                    PolicyAffectedPublishedValue(
                        design=kind,
                        study=str(value.get("study_kind")),
                        depth=value.get("depth"),
                        form=str(value.get("form")),
                        operation=operation.name,
                        fields=dependencies,
                        statement=_statement(value),
                    )
                )
    return tuple(affected)


def _statement(value: dict[str, object]) -> str:
    """The published figure to restate, preferring a band over a point value."""
    band = value.get("published_band")
    if isinstance(band, list):
        return f"published band {band!r}"
    if "value" in value:
        return f"published value {value['value']!r}"
    return "published statement"
