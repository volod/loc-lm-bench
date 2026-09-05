"""Read and write a vector store's project-owned records through their contracts.

`store_meta.json` is the member every other decision hangs off, so it is the one this module
reads FIRST and refuses on: an unreadable or future-major meta means the encoder identity, the
corpus fingerprint, and the opaque index members are all things this build cannot see whole.
The chunk and parent rows keep the compact form they have always been written in -- a store holds
hundreds of thousands of them and stamping an identity on each would multiply the file for no
reader -- so the producer validates each row through its contract and writes the row itself.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.errors import DatasetReadError
from llb.artifacts.io import json_document
from llb.artifacts.registry import ContractRegistry
from llb.artifacts.retrieval_graph.opaque import refuse_changed_members
from llb.artifacts.serialization import stated_sections
from llb.core.contracts.common import JsonObject
from llb.core.contracts.retrieval_graph.common import OpaqueIndexMember
from llb.core.contracts.retrieval_graph.stores import (
    CHUNK_SCHEMA_ID,
    STORE_META_SCHEMA_ID,
    ChunkRow,
    RagStoreMetaDocument,
)
from llb.rag.vector_store.layout import META_FILE


def store_meta_document(
    meta: JsonObject,
    index_members: Iterable[OpaqueIndexMember] = (),
    registry: ContractRegistry = DEFAULT_REGISTRY,
) -> RagStoreMetaDocument:
    """The current store-meta record for `meta`, whatever version the caller assembled.

    A caller that built the mapping in this release passes a current record and it validates
    directly; one replaying an older generation's meta reaches the same record through the
    registered migration, so both produce the same published form.
    """
    record = dict(meta)
    record.pop("index_members", None)  # the members are described by this publication, not carried
    read = registry.read_as(STORE_META_SCHEMA_ID, record, source="<store meta>")
    assert isinstance(read, RagStoreMetaDocument)
    return read.model_copy(update={"index_members": list(index_members)})


def write_store_meta(
    path: Path,
    meta: JsonObject,
    index_members: Iterable[OpaqueIndexMember] = (),
    registry: ContractRegistry = DEFAULT_REGISTRY,
) -> None:
    """Validate the store metadata and publish it with its contract identity."""
    document = store_meta_document(meta, index_members, registry)
    path.write_text(
        json.dumps(stated_sections(document), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_store_meta(
    path: Path, registry: ContractRegistry = DEFAULT_REGISTRY
) -> RagStoreMetaDocument:
    """Read one `store_meta.json`, current or pre-contract, at the current contract version."""
    read = registry.read_as(STORE_META_SCHEMA_ID, json_document(path), source=str(path))
    assert isinstance(read, RagStoreMetaDocument)
    return read


def readable_store_meta(path: Path, registry: ContractRegistry = DEFAULT_REGISTRY) -> JsonObject:
    """The store metadata as its consumers read it: current, with absent sections omitted.

    A generation written before the contract existed comes back with the duplicate-collapse knobs
    its readers were already defaulting, so an old store and a rebuilt one produce the same
    mapping rather than two that differ only in what nobody wrote down.
    """
    return stated_sections(read_store_meta(path, registry))


def validated_chunk_rows(
    rows: Iterable[JsonObject], registry: ContractRegistry = DEFAULT_REGISTRY
) -> Iterator[JsonObject]:
    """Yield each row after validating it against the chunk contract, unchanged.

    The row is yielded as the producer built it rather than as the model dumps it: the contract
    decides whether the record is admissible, and the file keeps the compact form every reader of
    a built store already expects.
    """
    for index, row in enumerate(rows, start=1):
        registry.read_as(CHUNK_SCHEMA_ID, row, source=f"<chunk row {index}>")
        yield row


def read_chunk_rows(
    path: Path, registry: ContractRegistry = DEFAULT_REGISTRY
) -> tuple[ChunkRow, ...]:
    """Read one `chunks.jsonl` / `parents.jsonl` member through the chunk contract."""
    rows: list[ChunkRow] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        source = f"{path}#row-{index}"
        try:
            read = registry.read_as(CHUNK_SCHEMA_ID, json.loads(line), source=source)
        except json.JSONDecodeError as exc:
            raise DatasetReadError(f"{source}: {exc}") from exc
        assert isinstance(read, ChunkRow)
        rows.append(read)
    return tuple(rows)


def refuse_unreadable_store(store_dir: Path, registry: ContractRegistry = DEFAULT_REGISTRY) -> None:
    """The gate a query passes: the metadata resolves, and every declared index member still hashes.

    Both halves answer the same question before anything expensive happens -- can this build read
    what it is about to query? A meta from a future major hides every field a newer writer added,
    and an index member whose bytes moved since publication no longer matches the rows beside it.
    """
    meta_path = store_dir / META_FILE
    if not meta_path.is_file():
        return
    document = read_store_meta(meta_path, registry)
    refuse_changed_members(store_dir, list(document.index_members))


__all__ = [
    "read_chunk_rows",
    "read_store_meta",
    "readable_store_meta",
    "refuse_unreadable_store",
    "store_meta_document",
    "validated_chunk_rows",
    "write_store_meta",
]
