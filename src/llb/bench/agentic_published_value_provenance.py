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

Every slice also PINS the file it was cut from, by content digest. Without that the slice is only
self-consistent: on a host that never ran the study -- CI, a fresh clone, this host after a `.data`
cleanup -- nothing could tell a slice cut from the cited run from one cut from a different run or
edited by hand, so the resolution proved internal agreement rather than provenance. A slice with no
digest is refused outright, and where the host still has the run, a digest that does not match the
file is refused as pinning a different artifact.

The field pointer that addresses a value inside either source lives in
`llb.bench.agentic_published_value_pointer`.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import cast

from llb.bench.agentic_published_value_pointer import read_field

# The committed slice of every referenced aggregate, project-root relative. `refresh_provenance_
# fixture` regenerates it from the artifacts under DATA_DIR; nothing here ever writes it implicitly.
PROVENANCE_FIXTURE = Path("samples/benchmarks/agentic_published_crossover_aggregates.json")
FIXTURE_SCHEMA_VERSION = 2

# The artifact pin: sha256 over the aggregate's bytes, algorithm-prefixed so a later change of
# algorithm is a visible fixture change rather than a silently re-read hex string.
DIGEST_ALGORITHM = "sha256"
_DIGEST = re.compile(rf"^{DIGEST_ALGORITHM}:[0-9a-f]{{64}}$")

_REGENERATE = "regenerate it with make bench-agentic-published-provenance"


@dataclass(frozen=True)
class CommittedSlice:
    """One cited aggregate's committed evidence: the file it was cut from, and the cut itself."""

    digest: str
    payload: dict[str, object]


class PublishedValueResolver:
    """Read published values back out of the artifacts that measured them.

    Two sources, deliberately not interchangeable. The committed slice is the one CI has and is
    therefore the authority the design is validated against; the artifact under DATA_DIR is present
    only on a host that ran the study, and is read as a CHECK on the slice -- first that it is the
    file the slice pins, then that it states the same value. A host that never ran the study
    validates against the slice alone rather than skipping the check.
    """

    def __init__(self, *, root: Path, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir
        self._fixture = load_provenance_fixture(root)
        self._measured: dict[str, dict[str, object] | None] = {}

    def resolve(self, provenance: object, *, where: str) -> float:
        """The number the run recorded at this provenance pair, refusing anything unresolvable."""
        artifact, field = provenance_pair(provenance, where=where)
        committed = self._committed(artifact, where=where)
        value = _number(
            read_field(committed.payload, field, where=f"{where}: {artifact}"), where=where
        )
        measured = self._measured_payload(artifact, committed, where=where)
        if measured is not None:
            recorded = _number(
                read_field(measured, field, where=f"{where}: the run artifact {artifact}"),
                where=where,
            )
            if recorded != value:
                raise ValueError(
                    f"{where}: the committed provenance slice states {value!r} for "
                    f"{field!r} while the run artifact {artifact} records {recorded!r} -- the "
                    f"slice was edited by hand or cut from another field, {_REGENERATE}"
                )
        return value

    def _committed(self, artifact: str, *, where: str) -> CommittedSlice:
        committed = self._fixture.get(artifact)
        if committed is None:
            raise ValueError(
                f"{where}: no committed slice of {artifact!r} exists, so the published value "
                f"cannot be resolved without the run -- add it with "
                f"make bench-agentic-published-provenance"
            )
        return committed

    def _measured_payload(
        self, artifact: str, committed: CommittedSlice, *, where: str
    ) -> dict[str, object] | None:
        """The run artifact itself when this host still has it, cached per artifact."""
        if artifact not in self._measured:
            self._measured[artifact] = (
                None
                if self._data_dir is None
                else _read_pinned_artifact(self._data_dir / artifact, committed, where=where)
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


def artifact_digest(path: Path) -> str:
    """The content digest that pins one run aggregate, computed over its bytes as written."""
    return f"{DIGEST_ALGORITHM}:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def load_provenance_fixture(root: Path) -> dict[str, CommittedSlice]:
    """The committed slices and their artifact pins, keyed by DATA_DIR-relative artifact path."""
    path = root / PROVENANCE_FIXTURE
    if not path.is_file():
        raise ValueError(f"the committed provenance fixture {PROVENANCE_FIXTURE} does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ValueError(
            f"{PROVENANCE_FIXTURE} must declare schema_version {FIXTURE_SCHEMA_VERSION}"
        )
    entries = cast(dict[str, dict[str, object]], payload["aggregates"])
    return {artifact: _committed_slice(artifact, entry) for artifact, entry in entries.items()}


def write_provenance_fixture(root: Path, aggregates: dict[str, CommittedSlice]) -> Path:
    """Write the committed slices back, sorted so a regeneration diffs to what actually moved."""
    path = root / PROVENANCE_FIXTURE
    payload = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "note": (
            "committed slices of the run artifacts the published agentic numbers were measured in, "
            "each pinned by the content digest of the artifact it was cut from; regenerate with "
            "make bench-agentic-published-provenance rather than by hand"
        ),
        "aggregates": {
            artifact: {
                "digest": aggregates[artifact].digest,
                "slice": aggregates[artifact].payload,
            }
            for artifact in sorted(aggregates)
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _committed_slice(artifact: str, entry: object) -> CommittedSlice:
    """One fixture entry, refusing a slice that pins no artifact rather than reading it anyway."""
    digest = entry.get("digest") if isinstance(entry, dict) else None
    payload = entry.get("slice") if isinstance(entry, dict) else None
    if digest is None:
        raise ValueError(
            f"{PROVENANCE_FIXTURE}: the committed slice of {artifact!r} records no content digest "
            "of the artifact it was cut from, so no host without that run can tell it from a slice "
            f"cut from a different one -- {_REGENERATE}"
        )
    if not isinstance(digest, str) or not _DIGEST.match(digest):
        raise ValueError(
            f"{PROVENANCE_FIXTURE}: the committed slice of {artifact!r} must pin its artifact with "
            f"a {DIGEST_ALGORITHM}:<hex> content digest, got {digest!r}"
        )
    if not isinstance(payload, dict):
        raise ValueError(
            f"{PROVENANCE_FIXTURE}: the committed slice of {artifact!r} carries no `slice` object"
        )
    return CommittedSlice(digest=digest, payload=cast(dict[str, object], payload))


def _read_pinned_artifact(
    path: Path, committed: CommittedSlice, *, where: str
) -> dict[str, object] | None:
    """The run aggregate on this host, refused unless it is the file the slice was cut from."""
    if not path.is_file():
        return None
    digest = artifact_digest(path)
    if digest != committed.digest:
        raise ValueError(
            f"{where}: the committed slice pins its aggregate at {committed.digest}, while "
            f"{path} digests to {digest} -- the slice was cut from a different run or the artifact "
            f"was edited, so the published value is not the one this design cites; "
            f"{_REGENERATE} once the cited run is the one under DATA_DIR"
        )
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _number(value: object, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{where}: the provenance field resolves to {value!r}, which is not a number"
        )
    return float(value)
