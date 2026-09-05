"""Retrieval, graph, and prompt-system artifacts read through their registered contracts.

The pairs under `samples/artifact_contracts/retrieval_graph/` are the same generation written two
ways: `current/` as this build publishes it, `legacy/` as this project wrote it before the
registry existed. Every test below asks the same question of one of them -- does the old form come
back as the record a current writer would produce, and does a form this build cannot read refuse
before anything acts on it.
"""

import json
import shutil
from pathlib import Path

import pytest

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.errors import (
    ArtifactContractError,
    DatasetReadError,
    UnsupportedFutureVersionError,
)
from llb.artifacts.retrieval_graph.datasets import (
    GRAPH_STORE_MEMBERS,
    PROMPT_SYSTEM_MEMBERS,
    VECTOR_STORE_MEMBERS,
    graph_store_manifest,
    prompt_system_manifest,
    vector_store_manifest,
)
from llb.artifacts.retrieval_graph.graphs import read_community_summaries, readable_graph_meta
from llb.artifacts.retrieval_graph.prompt_systems import (
    PACKAGE_FIELDS,
    read_member,
    read_prompt_system_manifest,
)
from llb.artifacts.retrieval_graph.sidecars import read_sidecar, write_sidecar
from llb.artifacts.retrieval_graph.stores import (
    read_chunk_rows,
    readable_store_meta,
    refuse_unreadable_store,
)
from llb.artifacts.retrieval_graph.survey import survey_generation
from llb.core.contracts.retrieval_graph.calibration import ROUTING_CALIBRATION_SCHEMA_ID
from llb.core.contracts.retrieval_graph.comparison import RETRIEVAL_COMPARISON_SCHEMA_ID

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = PROJECT_ROOT / "samples" / "artifact_contracts" / "retrieval_graph"
STORE = FIXTURES / "store"
GRAPH = FIXTURES / "graph"
PROMPT_SYSTEM = FIXTURES / "prompt_system"
SIDECARS = FIXTURES / "sidecars"


def test_pre_contract_store_meta_reads_as_the_record_a_current_writer_produces() -> None:
    """The old form differs only in the two absences it could not record."""
    current = readable_store_meta(STORE / "current" / "store_meta.json")
    legacy = readable_store_meta(STORE / "legacy" / "store_meta.json")

    assert legacy["collapse_duplicates"] is True  # the value its readers were already defaulting
    assert legacy["duplicate_tier"] == "exact"
    assert legacy["index_members"] == []  # an old generation never recorded its opaque members
    assert legacy == {**current, "index_members": []}


def test_chunk_rows_read_identically_from_both_generations() -> None:
    current = read_chunk_rows(STORE / "current" / "chunks.jsonl")
    legacy = read_chunk_rows(STORE / "legacy" / "chunks.jsonl")

    assert [row.model_dump() for row in current] == [row.model_dump() for row in legacy]
    assert current[0].chunk_id == "poryadok.md::000"
    assert current[0].metadata == {"pages": [1], "section_title": "Загальні положення"}


def test_a_store_from_an_unsupported_future_major_refuses_before_query_execution() -> None:
    with pytest.raises(UnsupportedFutureVersionError):
        refuse_unreadable_store(STORE / "unsupported-future")


def test_a_changed_index_member_refuses_before_query_execution(tmp_path: Path) -> None:
    store = tmp_path / "store"
    shutil.copytree(STORE / "current", store)
    refuse_unreadable_store(store)  # published state: every declared member still hashes

    (store / "index.faiss").write_bytes(b"a different faiss index\n")

    with pytest.raises(DatasetReadError, match="changed since publication"):
        refuse_unreadable_store(store)


def test_a_missing_index_member_refuses_before_query_execution(tmp_path: Path) -> None:
    store = tmp_path / "store"
    shutil.copytree(STORE / "current", store)
    (store / "lexical_index.json").unlink()

    with pytest.raises(DatasetReadError, match="is missing"):
        refuse_unreadable_store(store)


def test_the_store_dataset_binds_every_member_and_names_each_opaque_owner() -> None:
    manifest = vector_store_manifest(STORE / "current")
    by_id = {member.member_id: member for member in manifest.members}

    assert set(by_id) == {"store-meta", "chunks", "parents", "vector-index", "lexical-index"}
    assert by_id["chunks"].record_contract is not None
    assert by_id["chunks"].record_contract.schema_id == "llb.rag-chunk"
    assert by_id["vector-index"].opaque_binding is not None
    assert by_id["vector-index"].opaque_binding.owner == "faiss"
    assert by_id["vector-index"].opaque_binding.format_version == "1.9.0"
    assert by_id["lexical-index"].opaque_binding is not None
    assert by_id["lexical-index"].opaque_binding.format == "bm25-postings"


def test_a_pre_contract_store_binds_its_members_at_the_version_it_was_written_at() -> None:
    manifest = vector_store_manifest(STORE / "legacy")
    meta = next(member for member in manifest.members if member.member_id == "store-meta")

    assert meta.record_contract is not None
    assert meta.record_contract.schema_version == "1.0.0"
    assert [reading.refusal for reading in survey_generation(STORE / "legacy", manifest)] == [
        ""
    ] * 3


@pytest.mark.parametrize("generation", ["current", "legacy"])
def test_every_store_member_is_readable_in_both_generations(generation: str) -> None:
    root = STORE / generation
    readings = survey_generation(root, vector_store_manifest(root))

    assert [reading.refusal for reading in readings] == [""] * len(readings)
    assert {reading.member_id: reading.records for reading in readings}["chunks"] == 2


def test_graph_generations_read_to_the_same_records() -> None:
    current_meta = readable_graph_meta(GRAPH / "current" / "graph_meta.json")
    legacy_meta = readable_graph_meta(GRAPH / "legacy" / "graph_meta.json")
    current_summaries = read_community_summaries(GRAPH / "current" / "community_summaries.json")
    legacy_summaries = read_community_summaries(GRAPH / "legacy" / "community_summaries.json")

    assert current_meta == legacy_meta
    assert current_meta["n_nodes"] == 2 and current_meta["khop_depth"] == 2
    assert current_summaries == legacy_summaries
    assert set(current_summaries) == {"0"}


@pytest.mark.parametrize("generation", ["current", "legacy"])
def test_every_graph_member_is_readable_in_both_generations(generation: str) -> None:
    root = GRAPH / generation
    readings = survey_generation(root, graph_store_manifest(root))

    assert [reading.refusal for reading in readings] == [""] * len(readings)
    assert {reading.member_id: reading.records for reading in readings}["edges"] == 1


def test_prompt_system_packages_read_to_the_same_records() -> None:
    for schema_id in PACKAGE_FIELDS:
        name = {
            "llb.prompt-system-anthology": "anthology.json",
            "llb.prompt-system-doc-metadata": "doc_metadata.json",
            "llb.prompt-system-mapping": "graph_rag_mapping.json",
            "llb.prompt-system-candidates": "candidates.json",
        }[schema_id]
        current = read_member(PROMPT_SYSTEM / "current" / name, schema_id)
        legacy = read_member(PROMPT_SYSTEM / "legacy" / name, schema_id)
        assert current == legacy, name

    current_manifest = read_prompt_system_manifest(PROMPT_SYSTEM / "current" / "manifest.json")
    legacy_manifest = read_prompt_system_manifest(PROMPT_SYSTEM / "legacy" / "manifest.json")
    assert current_manifest == legacy_manifest
    assert current_manifest.candidates[0].status == "pinned"


@pytest.mark.parametrize("generation", ["current", "legacy"])
def test_every_prompt_system_member_is_readable_in_both_generations(generation: str) -> None:
    root = PROMPT_SYSTEM / generation
    readings = survey_generation(root, prompt_system_manifest(root))

    assert [reading.refusal for reading in readings] == [""] * len(readings)
    assert {reading.member_id: reading.records for reading in readings}["candidates"] == 1


@pytest.mark.parametrize(
    ("name", "schema_id"),
    [
        ("retrieval-comparison.json", RETRIEVAL_COMPARISON_SCHEMA_ID),
        ("routing-calibration.json", ROUTING_CALIBRATION_SCHEMA_ID),
    ],
)
def test_a_sidecar_reads_with_and_without_its_identity(
    name: str, schema_id: str, tmp_path: Path
) -> None:
    """A sidecar archived before the contract carries no identity; it reads to the same record."""
    published = read_sidecar(SIDECARS / name, schema_id)

    record = json.loads((SIDECARS / name).read_text(encoding="utf-8"))
    pre_contract = tmp_path / name
    pre_contract.write_text(
        json.dumps({k: v for k, v in record.items() if not k.startswith("schema_")}),
        encoding="utf-8",
    )

    assert read_sidecar(pre_contract, schema_id) == published


def test_a_sidecar_this_build_cannot_read_refuses_before_publication(tmp_path: Path) -> None:
    record = json.loads((SIDECARS / "retrieval-comparison.json").read_text(encoding="utf-8"))
    record["backends"]["dense"]["recall_at_k"] = "one"

    with pytest.raises(ArtifactContractError):
        write_sidecar(tmp_path / "report.json", RETRIEVAL_COMPARISON_SCHEMA_ID, record)

    assert not (tmp_path / "report.json").exists()


def test_every_described_member_names_a_registered_contract_or_an_owner() -> None:
    """The registry accounts for every project-owned member; the rest name whose format they are."""
    for member in (*VECTOR_STORE_MEMBERS, *GRAPH_STORE_MEMBERS, *PROMPT_SYSTEM_MEMBERS):
        if member.schema_id is None:
            assert member.opaque_binding is not None, member.member_id
            assert member.opaque_binding.owner and member.opaque_binding.format_version
        else:
            assert DEFAULT_REGISTRY.definition(member.schema_id).current_version
