"""Stamp identity onto a record on the way out, and take it off on the way back.

Some producers build their contract model and write it whole; the ones here cannot. A store's
chunk rows and a graph's node rows are handed to retrieval as plain mappings, and every downstream
reader -- filters, fusion, the span metrics, the comparison lanes -- keys on the fields the chunker
wrote. So identity lives on DISK, where a reader that opened the file needs it, and is removed
again as the record enters the process: adding two keys to every in-memory chunk would change what
a hundred equality assertions and metadata merges see, for no reading anyone takes.

`encode` therefore writes `{schema_id, schema_version, ...record}` and `decode` validates through
the registry -- migrating an older row on the way -- and returns the logical record without its
identity and without the fields it states as absent.
"""

from collections.abc import Mapping
from typing import Any

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.registry import ContractRegistry

IDENTITY_KEYS = ("schema_id", "schema_version")


def encode(schema_id: str, version: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """The record as it is written: its identity first, then the fields it already had."""
    body = {key: value for key, value in record.items() if key not in IDENTITY_KEYS}
    return {"schema_id": schema_id, "schema_version": version, **body}


def decode(
    schema_id: str,
    record: object,
    *,
    version: str | None = None,
    source: str = "<record>",
    registry: ContractRegistry = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    """Read one record of a known family at the current contract, without its identity.

    A record carrying no identity is a file this project wrote before the family was registered;
    it is stamped with `version` when a binding supplies one and with the family's declared legacy
    version otherwise.

    A stated `null` is dropped along with the identity, at every level the contract models -- a
    retrieved span's absent `text_preview` reads the same as one an older writer never wrote. In
    these families `null` means "this producer recorded no such thing", which is what an absent key
    already means to every reader here, so dropping it makes one reading of a record however it was
    written: by an old producer that omitted the key, by a current one that omits it, or by an
    upgrade that wrote the absence out. An open body map (a benchmark cell, a stage result) is the
    producer's own and is passed through whole, nulls included.
    """
    normalized = registry.normalize(schema_id, record)
    model = registry.read_as(schema_id, normalized, version=version, source=source)
    dumped = model.model_dump(mode="json", exclude_none=True)
    return {key: value for key, value in dumped.items() if key not in IDENTITY_KEYS}
