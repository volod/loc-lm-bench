"""Read and write a prompt-system package's five members through their contracts.

Three of the members were written as bare JSON arrays or a bare mapping, which is a shape that
cannot carry an identity at all. Each is published from now on as a document whose single field
holds that same list or mapping, and each reader accepts both forms: a package prepared before
the contract existed still opens, and it opens at the current version.
"""

import json
from pathlib import Path
from typing import TypeAlias

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.errors import DatasetReadError
from llb.artifacts.io import json_document
from llb.artifacts.registry import ContractRegistry
from llb.artifacts.serialization import stated_sections
from llb.core.contracts.common import JsonObject
from llb.core.contracts.retrieval_graph.prompt_system import (
    ANTHOLOGY_SCHEMA_ID,
    CANDIDATES_SCHEMA_ID,
    DOC_METADATA_SCHEMA_ID,
    MAPPING_SCHEMA_ID,
    PROMPT_SYSTEM_MANIFEST_SCHEMA_ID,
    AnthologyDocument,
    DocMetadataDocument,
    GraphRagMappingDocument,
    PromptCandidatesDocument,
    PromptSystemManifestDocument,
)

PackageDocument: TypeAlias = (
    AnthologyDocument | DocMetadataDocument | GraphRagMappingDocument | PromptCandidatesDocument
)

# The single field each wrapped member carries. A package written before the contract existed
# holds the bare list or mapping directly, so the reader wraps it and stamps the legacy version.
PACKAGE_FIELDS: dict[str, str] = {
    ANTHOLOGY_SCHEMA_ID: "passages",
    DOC_METADATA_SCHEMA_ID: "documents",
    MAPPING_SCHEMA_ID: "mapping",
    CANDIDATES_SCHEMA_ID: "candidates",
}


def write_member(
    path: Path, schema_id: str, payload: object, registry: ContractRegistry = DEFAULT_REGISTRY
) -> None:
    """Publish one wrapped package member with its contract identity."""
    document = _validated({PACKAGE_FIELDS[schema_id]: payload}, schema_id, str(path), registry)
    _write_json(path, stated_sections(document))


def read_member(
    path: Path, schema_id: str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> list[JsonObject] | dict[str, list[str]]:
    """The wrapped member's own list or mapping, from either the document or the bare form.

    Returned as JSON-ready values rather than as contract models, because the callers are the
    package's own readers: they rebuild their dataclasses from exactly the shape the file holds.
    """
    document = _read_wrapped(_load(path), schema_id, str(path), registry)
    member: list[JsonObject] | dict[str, list[str]] = document.model_dump(mode="json")[
        PACKAGE_FIELDS[schema_id]
    ]
    return member


def write_prompt_system_manifest(
    path: Path, manifest: JsonObject, registry: ContractRegistry = DEFAULT_REGISTRY
) -> None:
    """Validate a prompt-system run manifest and publish it with its contract identity."""
    _write_json(
        path, stated_sections(read_prompt_system_manifest_record(manifest, str(path), registry))
    )


def read_prompt_system_manifest(
    path: Path, registry: ContractRegistry = DEFAULT_REGISTRY
) -> PromptSystemManifestDocument:
    """Read one prompt-system `manifest.json`, current or pre-contract."""
    return read_prompt_system_manifest_record(json_document(path), str(path), registry)


def read_prompt_system_manifest_record(
    record: object, source: str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> PromptSystemManifestDocument:
    if not isinstance(record, dict):
        raise DatasetReadError(f"{source}: prompt-system manifest must be a JSON object")
    read = registry.read_as(PROMPT_SYSTEM_MANIFEST_SCHEMA_ID, record, source=source)
    assert isinstance(read, PromptSystemManifestDocument)
    return read


def _read_wrapped(
    record: object, schema_id: str, source: str, registry: ContractRegistry
) -> PackageDocument:
    """One member read from either form: the published document, or the bare list or mapping."""
    field = PACKAGE_FIELDS[schema_id]
    if isinstance(record, (list, tuple)):
        payload: JsonObject = {field: list(record)}
    elif isinstance(record, dict):
        # A published document always names its own identity; a bare mapping never can, so
        # `schema_id` -- and not a field name a salient term could collide with -- is what tells
        # the two apart.
        payload = record if "schema_id" in record else {field: record}
    else:
        raise DatasetReadError(f"{source}: expected a JSON object or array for {schema_id}")
    return _validated(payload, schema_id, source, registry)


def _validated(
    payload: JsonObject, schema_id: str, source: str, registry: ContractRegistry
) -> PackageDocument:
    read = registry.read_as(schema_id, payload, source=source)
    assert isinstance(
        read,
        (AnthologyDocument, DocMetadataDocument, GraphRagMappingDocument, PromptCandidatesDocument),
    )
    return read


def _load(path: Path) -> object:
    """The member's raw JSON: an object for a published document, an array or mapping before it."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetReadError(f"{path}: cannot read record: {exc}") from exc


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
