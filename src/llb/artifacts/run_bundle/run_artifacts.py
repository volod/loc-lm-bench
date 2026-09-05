"""The additional files a run bundle may publish beside its manifest and its score rows.

`persist_run` used to take a `name -> text` mapping, which made the bundle's own transaction the
one place in this project where any bytes could land under any name. A later reader then had to
guess what `power-analysis.json` was from its filename, and nothing stopped two studies from
publishing incompatible records under the same one.

An artifact now arrives already DECLARED. It is either a record this project models -- validated
against its registered contract before the staging directory is renamed -- or a human report: a
rendered table or narrative nobody parses, which says so and says why. There is no third form, so
an artifact that is neither cannot be published at all.

Each constructor validates the BYTES the producer is about to write rather than a mapping beside
them, for the reason the comparison sidecars do: what a later reader gets back is the file and
nothing else. Keeping the producer's own encoding is not a convenience either -- a published
study aggregate is cited by digest, and re-encoding one would break the citation that points at it.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.errors import DatasetReadError
from llb.artifacts.registry import ContractRegistry
from llb.artifacts.run_bundle.studies import study_record
from llb.core.contracts.artifacts import ContractReference
from llb.core.contracts.run_bundle.manifest import RunArtifactDeclaration
from llb.core.contracts.run_bundle.studies import STUDY_ANALYSIS_SCHEMA_ID, STUDY_DESIGN_SCHEMA_ID

MARKDOWN_MEDIA_TYPE = "text/markdown"
JSON_MEDIA_TYPE = "application/json"

# The reason a rendered report is exempt from a record contract. It is one sentence rather than a
# flag because the exemption has to stay arguable: the moment something parses one of these, the
# reason stops being true and the file needs a contract.
RENDERED_REPORT_REASON = "Rendered human-readable report; no consumer parses it."


@dataclass(frozen=True)
class RunArtifact:
    """One additional bundle member: its bytes, and what makes them readable."""

    name: str
    content: str
    media_type: str
    record_contract: ContractReference | None = None
    human_report: str | None = None
    study_id: str | None = None

    def declaration(self) -> RunArtifactDeclaration:
        """The manifest entry this artifact publishes itself as, digest included."""
        if not self.name or PurePosixPath(self.name).name != self.name:
            raise ValueError(f"invalid additional artifact name: {self.name!r}")
        encoded = self.content.encode("utf-8")
        return RunArtifactDeclaration(
            name=self.name,
            media_type=self.media_type,
            digest=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
            n_bytes=len(encoded),
            record_contract=self.record_contract,
            human_report=self.human_report,
            study_id=self.study_id,
        )


def human_report(name: str, content: str, reason: str = RENDERED_REPORT_REASON) -> RunArtifact:
    """A rendered table or narrative, declared exempt with the reason the exemption holds."""
    return RunArtifact(
        name=name, content=content, media_type=MARKDOWN_MEDIA_TYPE, human_report=reason
    )


def contract_document(
    name: str, schema_id: str, content: str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> RunArtifact:
    """A JSON document whose encoded bytes validate against the contract it declares."""
    definition = registry.definition(schema_id)
    registry.read_as(schema_id, _decoded(name, content), source=name)
    return RunArtifact(
        name=name,
        content=content,
        media_type=JSON_MEDIA_TYPE,
        record_contract=ContractReference(
            schema_id=schema_id, schema_version=definition.current_version
        ),
    )


def study_artifact(
    name: str,
    schema_id: str,
    content: str,
    *,
    study_id: str,
    registry: ContractRegistry = DEFAULT_REGISTRY,
) -> RunArtifact:
    """A study design or analysis: validated as a record, written in the study's local form."""
    if schema_id not in {STUDY_DESIGN_SCHEMA_ID, STUDY_ANALYSIS_SCHEMA_ID}:
        raise ValueError(f"{name}: {schema_id!r} is not a study record contract")
    definition = registry.definition(schema_id)
    record = study_record(
        _decoded(name, content, allow_table=True),
        schema_id,
        study_id=study_id,
        registry=registry,
        source=name,
    )
    return RunArtifact(
        name=name,
        content=content,
        media_type=JSON_MEDIA_TYPE,
        record_contract=ContractReference(
            schema_id=schema_id, schema_version=definition.current_version
        ),
        study_id=record.study_id,
    )


def _decoded(name: str, content: str, *, allow_table: bool = False) -> dict[str, object]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise DatasetReadError(f"{name}: additional artifact is not readable JSON: {exc}") from exc
    if isinstance(payload, list) and allow_table:
        return payload  # type: ignore[return-value]
    if not isinstance(payload, dict):
        raise DatasetReadError(f"{name}: expected one JSON object record")
    return payload
