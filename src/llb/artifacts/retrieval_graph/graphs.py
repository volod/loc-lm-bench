"""Read and write a graph store's project-owned records through their contracts.

Node and edge rows keep the compact form the store has always written -- one line per record, no
per-row identity -- because a generation holds tens of thousands of them and the file is bound at
its version by the store's own metadata. The metadata and the community summaries are single
documents, so each carries its identity and is resolved before the graph is queried.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.io import json_document
from llb.artifacts.registry import ContractRegistry
from llb.artifacts.serialization import stated_sections
from llb.core.contracts.common import JsonObject
from llb.core.contracts.retrieval_graph.graph import (
    COMMUNITY_SUMMARIES_SCHEMA_ID,
    GRAPH_EDGE_SCHEMA_ID,
    GRAPH_META_SCHEMA_ID,
    GRAPH_NODE_SCHEMA_ID,
    CommunitySummariesDocument,
    GraphMetaDocument,
)

SUMMARIES_KEY = "summaries"


def validated_graph_rows(
    rows: Iterable[JsonObject], schema_id: str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> Iterator[JsonObject]:
    """Yield each node or edge row after validating it against `schema_id`, unchanged."""
    for index, row in enumerate(rows, start=1):
        registry.read_as(schema_id, row, source=f"<{schema_id} row {index}>")
        yield row


def validated_nodes(
    rows: Iterable[JsonObject], registry: ContractRegistry = DEFAULT_REGISTRY
) -> Iterator[JsonObject]:
    return validated_graph_rows(rows, GRAPH_NODE_SCHEMA_ID, registry)


def validated_edges(
    rows: Iterable[JsonObject], registry: ContractRegistry = DEFAULT_REGISTRY
) -> Iterator[JsonObject]:
    return validated_graph_rows(rows, GRAPH_EDGE_SCHEMA_ID, registry)


def write_graph_meta(
    path: Path, meta: JsonObject, registry: ContractRegistry = DEFAULT_REGISTRY
) -> None:
    """Validate one graph generation's metadata and publish it with its contract identity."""
    document = read_meta_record(meta, source="<graph meta>", registry=registry)
    path.write_text(
        json.dumps(stated_sections(document), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_graph_meta(path: Path, registry: ContractRegistry = DEFAULT_REGISTRY) -> GraphMetaDocument:
    """Read one `graph_meta.json`, current or pre-contract, at the current contract version."""
    return read_meta_record(json_document(path), source=str(path), registry=registry)


def readable_graph_meta(path: Path, registry: ContractRegistry = DEFAULT_REGISTRY) -> JsonObject:
    """The graph metadata as its consumers read it: current, with absent sections omitted."""
    return stated_sections(read_graph_meta(path, registry))


def read_meta_record(
    record: JsonObject, *, source: str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> GraphMetaDocument:
    """One graph-meta mapping at the current version, whether or not it names its own identity."""
    read = registry.read_as(GRAPH_META_SCHEMA_ID, record, source=source)
    assert isinstance(read, GraphMetaDocument)
    return read


def write_community_summaries(path: Path, summaries: dict[str, str]) -> None:
    """Publish the diagnostic community summaries with their contract identity."""
    document = CommunitySummariesDocument(
        schema_id="llb.graph-community-summaries",
        schema_version="1.0.0",
        summaries=dict(summaries),
    )
    path.write_text(
        json.dumps(stated_sections(document), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_community_summaries(
    path: Path, registry: ContractRegistry = DEFAULT_REGISTRY
) -> dict[str, str]:
    """Read the summaries member in either form: the identified document, or the bare mapping.

    A generation written before the contract existed holds `{community_id: summary}` and nothing
    else, so the absence of an identity is what tells the two apart: a published document always
    names its own, and a bare mapping of community ids never can.
    """
    record = json_document(path)
    payload = record if "schema_id" in record else {SUMMARIES_KEY: record}
    read = registry.read_as(COMMUNITY_SUMMARIES_SCHEMA_ID, payload, source=str(path))
    assert isinstance(read, CommunitySummariesDocument)
    return dict(read.summaries)
