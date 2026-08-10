"""Resolve a published number back to the run artifact and the field that produced it.

A number a design file states about a completed run is a TRANSCRIPTION until something reads it
back out of the run. Transcription is exactly the failure a downstream check cannot catch: a digit
dropped out of an interpolated crossover lands inside the same fold step for any small slip, so the
placement rules pass and the restatement reports a number nobody measured.

The seam here is one pair per published value -- the artifact it came from, and the field inside it
-- plus one rule for reading them: the value is resolved out of the repo's own verbatim copy of that
artifact, so a host that never ran the study resolves it with nothing taken on faith, and on a host
that still HAS the run root the copy is falsified against the run before any value is read. The
committed copy, its pin, and the growth policy that bounds them live in
`llb.bench.agentic_published_value_fixture`; the field pointer that addresses a value inside an
aggregate lives in `llb.bench.agentic_published_value_pointer`; and the per-pointer re-derivation
from the aggregate's own cells lives in `llb.bench.agentic_published_aggregate_consistency`.
"""

from pathlib import Path

from llb.bench.agentic_published_aggregate_consistency import validate_aggregate_field
from llb.bench.agentic_published_value_fixture import (
    REGENERATE,
    CommittedAggregate,
    artifact_digest,
    is_contained_artifact,
    load_provenance_fixture,
)
from llb.bench.agentic_published_value_pointer import read_field


class PublishedValueResolver:
    """Read published values back out of the artifacts that measured them.

    Two sources, deliberately not interchangeable. The committed copy is the one every host has and
    is therefore what the design is validated against; the artifact under DATA_DIR is present only
    on a host that ran the study, and is read as a CHECK on the copy -- byte for byte, since the
    copy is the file. A host that never ran the study validates against the copy alone rather than
    skipping the check.
    """

    def __init__(self, *, root: Path, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir
        self._fixture = load_provenance_fixture(root)
        self._confirmed: set[str] = set()

    def resolve(self, provenance: object, *, where: str) -> float:
        """The number the run recorded at this provenance pair, refusing anything unresolvable."""
        artifact, field = provenance_pair(provenance, where=where)
        committed = self._committed(artifact, where=where)
        self._confirm_run_root(artifact, committed, where=where)
        resolved = _number(
            read_field(committed.payload, field, where=f"{where}: {artifact}"), where=where
        )
        validate_aggregate_field(committed.payload, field, resolved, where=f"{where}: {artifact}")
        return resolved

    def _committed(self, artifact: str, *, where: str) -> CommittedAggregate:
        committed = self._fixture.get(artifact)
        if committed is None:
            raise ValueError(
                f"{where}: no committed copy of {artifact!r} exists, so the published value "
                f"cannot be resolved without the run -- add it with "
                f"make bench-agentic-published-provenance"
            )
        return committed

    def _confirm_run_root(
        self, artifact: str, committed: CommittedAggregate, *, where: str
    ) -> None:
        """Where this host still has the run, its artifact must be the committed copy exactly."""
        if self._data_dir is None or artifact in self._confirmed:
            return
        self._confirmed.add(artifact)
        path = self._data_dir / artifact
        if not path.is_file():
            return
        digest = artifact_digest(path)
        if digest != committed.digest:
            raise ValueError(
                f"{where}: the committed evidence pins its aggregate at {committed.digest}, while "
                f"{path} digests to {digest} -- the committed copy came from a different run or "
                f"the artifact was edited, so the published value is not the one this design "
                f"cites; {REGENERATE} once the cited run is the one under DATA_DIR"
            )


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
    if not is_contained_artifact(artifact):
        raise ValueError(
            f"{where}: `provenance.artifact` must stay inside DATA_DIR, got {artifact!r}"
        )
    if not isinstance(field, str) or not field:
        raise ValueError(
            f"{where}: `provenance.field` must be the field pointer inside the artifact"
        )
    return artifact, field


def _number(value: object, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{where}: the provenance field resolves to {value!r}, which is not a number"
        )
    return float(value)
