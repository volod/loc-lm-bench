"""Resolve a published number back to the run artifact and the field that produced it.

A number a design file states about a completed run is a TRANSCRIPTION until something reads it
back out of the run. Transcription is exactly the failure a downstream check cannot catch: a digit
dropped out of an interpolated crossover lands inside the same fold step for any small slip, so the
placement rules pass and the restatement reports a number nobody measured.

The seam here is one pair per published value -- the artifact it came from, and the field inside it
-- plus one rule for reading them: the value is resolved out of a COMMITTED slice of that artifact
so CI needs no run, and, on a host that still has the run, out of the artifact itself as well, with
the two required to agree. The committed slice is generated from the artifact rather than typed,
which is what keeps it from being a second transcription of the first one.

The field pointer is a dotted path with one extra form, a row selector, because every one of these
aggregates keys its per-depth rows by a field rather than by position:

    depth_surface[depth=6].crossover_max_prompt_chars
    depth_ladders[depth=10].boundary.guard_boundary_chars
    cap_peak_prompt_chars.6
"""

import json
from pathlib import Path
import re
from typing import cast

# The committed slice of every referenced aggregate, project-root relative. `refresh_provenance_
# fixture` regenerates it from the artifacts under DATA_DIR; nothing here ever writes it implicitly.
PROVENANCE_FIXTURE = Path("samples/benchmarks/agentic_published_crossover_aggregates.json")
FIXTURE_SCHEMA_VERSION = 1

# `name[key=value]`: pick the row of the `name` list whose integer `key` field is `value`.
_ROW_SELECTOR = re.compile(r"^(?P<name>\w+)\[(?P<key>\w+)=(?P<value>-?\d+)\]$")


class PublishedValueResolver:
    """Read published values back out of the artifacts that measured them.

    Two sources, deliberately not interchangeable. The committed slice is the one CI has and is
    therefore the authority the design is validated against; the artifact under DATA_DIR is present
    only on a host that ran the study, and is read as a CHECK on the slice. A host that never ran
    the study validates against the slice alone rather than skipping the check.
    """

    def __init__(self, *, root: Path, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir
        self._fixture = load_provenance_fixture(root)
        self._measured: dict[str, dict[str, object] | None] = {}

    def resolve(self, provenance: object, *, where: str) -> float:
        """The number the run recorded at this provenance pair, refusing anything unresolvable."""
        artifact, field = provenance_pair(provenance, where=where)
        committed = self._committed(artifact, where=where)
        value = _number(read_field(committed, field, where=f"{where}: {artifact}"), where=where)
        measured = self._measured_payload(artifact)
        if measured is not None:
            recorded = _number(
                read_field(measured, field, where=f"{where}: the run artifact {artifact}"),
                where=where,
            )
            if recorded != value:
                raise ValueError(
                    f"{where}: the committed provenance slice states {value!r} for "
                    f"{field!r} while the run artifact {artifact} records {recorded!r} -- the "
                    f"slice is stale, regenerate it with make bench-agentic-published-provenance"
                )
        return value

    def _committed(self, artifact: str, *, where: str) -> dict[str, object]:
        payload = self._fixture.get(artifact)
        if payload is None:
            raise ValueError(
                f"{where}: no committed slice of {artifact!r} exists, so the published value "
                f"cannot be resolved without the run -- add it with "
                f"make bench-agentic-published-provenance"
            )
        return payload

    def _measured_payload(self, artifact: str) -> dict[str, object] | None:
        """The run artifact itself when this host still has it, cached per artifact."""
        if artifact not in self._measured:
            self._measured[artifact] = (
                None if self._data_dir is None else _read_artifact(self._data_dir / artifact)
            )
        return self._measured[artifact]


def provenance_pair(provenance: object, *, where: str) -> tuple[str, str]:
    """The `(artifact, field)` a published value must name, or a refusal saying which is missing."""
    if not isinstance(provenance, dict):
        raise ValueError(
            f"{where}: a published value must carry a `provenance` object naming the run artifact "
            "it came from and the field inside it, so it is resolved rather than transcribed"
        )
    artifact = provenance.get("artifact")
    field = provenance.get("field")
    if not isinstance(artifact, str) or not artifact:
        raise ValueError(f"{where}: `provenance.artifact` must be a DATA_DIR-relative run artifact")
    if Path(artifact).is_absolute() or ".." in Path(artifact).parts:
        raise ValueError(
            f"{where}: `provenance.artifact` must stay inside DATA_DIR, got {artifact!r}"
        )
    if not isinstance(field, str) or not field:
        raise ValueError(
            f"{where}: `provenance.field` must be the field pointer inside the artifact"
        )
    return artifact, field


def read_field(payload: dict[str, object], field: str, *, where: str) -> object:
    """Walk a field pointer, naming the segment that failed rather than the whole path."""
    node: object = payload
    for segment in field.split("."):
        node = _step(node, segment, field=field, where=where)
    return node


def merge_field_slice(target: dict[str, object], payload: dict[str, object], field: str) -> None:
    """Copy just the addressed path of `payload` into `target`, preserving the artifact's shape.

    Shape-preserving is the point: the committed slice is then read by the SAME pointer walk as the
    artifact, so a slice that resolves and an artifact that does not (or the reverse) is impossible.
    """
    _merge(target, payload, field.split("."), field=field)


def load_provenance_fixture(root: Path) -> dict[str, dict[str, object]]:
    """The committed slices, keyed by DATA_DIR-relative artifact path."""
    path = root / PROVENANCE_FIXTURE
    if not path.is_file():
        raise ValueError(f"the committed provenance fixture {PROVENANCE_FIXTURE} does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ValueError(
            f"{PROVENANCE_FIXTURE} must declare schema_version {FIXTURE_SCHEMA_VERSION}"
        )
    return cast(dict[str, dict[str, object]], payload["aggregates"])


def write_provenance_fixture(root: Path, aggregates: dict[str, dict[str, object]]) -> Path:
    """Write the committed slices back, sorted so a regeneration diffs to what actually moved."""
    path = root / PROVENANCE_FIXTURE
    payload = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "note": (
            "committed slices of the run artifacts the published agentic numbers were measured in; "
            "regenerate with make bench-agentic-published-provenance rather than by hand"
        ),
        "aggregates": {key: aggregates[key] for key in sorted(aggregates)},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_artifact(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _step(node: object, segment: str, *, field: str, where: str) -> object:
    selector = _ROW_SELECTOR.match(segment)
    if selector is None:
        if not isinstance(node, dict) or segment not in node:
            raise ValueError(
                f"{where}: the field pointer {field!r} reaches no {segment!r} in the artifact"
            )
        return node[segment]
    name, key, value = selector["name"], selector["key"], int(selector["value"])
    rows = node.get(name) if isinstance(node, dict) else None
    row = _select_row(rows, key, value)
    if row is None:
        raise ValueError(
            f"{where}: the field pointer {field!r} selects the {name!r} row with {key}={value}, "
            "which the artifact does not carry"
        )
    return row


def _select_row(rows: object, key: str, value: int) -> dict[str, object] | None:
    if not isinstance(rows, list):
        return None
    return next(
        (
            row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get(key), int) and row[key] == value
        ),
        None,
    )


def _merge(
    target: dict[str, object], payload: dict[str, object], segments: list[str], *, field: str
) -> None:
    segment, rest = segments[0], segments[1:]
    selector = _ROW_SELECTOR.match(segment)
    if selector is None:
        if segment not in payload:
            raise ValueError(f"the field pointer {field!r} reaches no {segment!r} in the artifact")
        if not rest:
            target[segment] = payload[segment]
            return
        nested = payload[segment]
        if not isinstance(nested, dict):
            raise ValueError(f"the field pointer {field!r} walks into a non-object at {segment!r}")
        _merge(cast(dict[str, object], target.setdefault(segment, {})), nested, rest, field=field)
        return
    if not rest:
        raise ValueError(f"the field pointer {field!r} ends on a row selector, not on a field")
    name, key, value = selector["name"], selector["key"], int(selector["value"])
    row = _select_row(payload.get(name), key, value)
    if row is None:
        raise ValueError(
            f"the field pointer {field!r} selects a {name!r} row the artifact does not carry"
        )
    bucket = cast(list[dict[str, object]], target.setdefault(name, []))
    kept = _select_row(bucket, key, value)
    if kept is None:
        kept = {key: value}
        bucket.append(kept)
    _merge(kept, row, rest, field=field)


def _number(value: object, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{where}: the provenance field resolves to {value!r}, which is not a number"
        )
    return float(value)
