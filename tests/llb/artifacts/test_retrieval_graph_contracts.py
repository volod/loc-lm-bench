"""The retrieval, graph, and prompt-system surface reads through the registry, at both its forms.

The fixtures under `samples/artifact_contracts/retrieval_graph/` are the two shapes that matter:
a store, graph, and package written by this build, and the pre-contract form the same three had
before the registry existed. Every assertion here is that the two reach the same logical records,
that an opaque member names its owner instead of being modelled, or that a form this build cannot
read is refused before anything expensive runs.
"""

import json
import shutil
from pathlib import Path

import pytest

from llb.artifacts.dataset_reading import survey_dataset, upgrade_dataset
from llb.artifacts.datasets import (
    DATASET_MANIFEST_FILE,
    load_dataset_manifest,
    member_digest_problems,
)
from llb.artifacts.errors import DatasetReadError, UnsupportedFutureVersionError
from llb.artifacts.gates import (
    ArtifactCompatibilityError,
    refuse_tampered_dataset,
    refuse_unreadable_prompt_system,
)
from llb.artifacts.retrieval.datasets import (
    graph_dataset_manifest,
    prompt_system_dataset_manifest,
    store_dataset_manifest,
)
from llb.artifacts.retrieval.families import retrieval_definitions
from llb.graph.store_io import (
    read_community_summaries,
    read_edge_rows,
    read_graph_meta,
    read_node_rows,
)
from llb.prompt_system.review import load_candidates
from llb.rag.comparison.sidecar import read_sidecar, write_sidecar
from llb.rag.query_prep.glossary import Glossary
from llb.rag.vector_store.persistence import read_store_chunks, read_store_meta

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = PROJECT_ROOT / "samples" / "artifact_contracts" / "retrieval_graph"
STORE = FIXTURES / "store"
GRAPH = FIXTURES / "graph"
PACKAGE = FIXTURES / "prompt-system"
LEGACY = FIXTURES / "legacy"
FUTURE = FIXTURES / "unsupported-future"


def test_every_retrieval_family_is_at_one_initial_version_with_a_legacy_read() -> None:
    """No release has shipped any of these, so a second version would be a form nobody wrote."""
    for definition in retrieval_definitions():
        assert list(definition.models) == ["1.0.0"] == [definition.current_version]
        assert definition.migrations == () and definition.refusals == ()
        # Every one of these families has files this project already wrote, so each must say
        # which version a reader that knows the family may assume.
        assert definition.legacy_version == "1.0.0"


def test_current_and_pre_contract_stores_load_to_identical_chunk_records() -> None:
    current_chunks = read_store_chunks(STORE / "chunks.jsonl")
    legacy_chunks = read_store_chunks(LEGACY / "store" / "chunks.jsonl")
    assert current_chunks == legacy_chunks
    assert current_chunks[0]["chunk_id"].endswith("#0000")


def test_a_store_that_recorded_no_duplicate_knobs_reads_as_having_recorded_none() -> None:
    """Stores in exactly this state are on disk, so the contract states the absence, not a value."""
    current = read_store_meta(STORE / "store_meta.json")
    legacy = read_store_meta(LEGACY / "store" / "store_meta.json")

    assert current["collapse_duplicates"] is True and current["duplicate_tier"] == "exact"
    assert "collapse_duplicates" not in legacy and "duplicate_tier" not in legacy
    # Everything the older writer DID record reads identically.
    assert legacy == {key: value for key, value in current.items() if key in legacy}


def test_loaded_chunk_records_carry_no_contract_identity() -> None:
    """Identity belongs to the file; a chunk in memory is what the chunker built."""
    for chunk in read_store_chunks(STORE / "chunks.jsonl"):
        assert "schema_id" not in chunk and "schema_version" not in chunk


def test_current_and_pre_contract_graphs_load_to_the_same_rows() -> None:
    assert read_node_rows(GRAPH / "nodes.jsonl") == read_node_rows(LEGACY / "graph" / "nodes.jsonl")
    assert read_edge_rows(GRAPH / "edges.jsonl") == read_edge_rows(LEGACY / "graph" / "edges.jsonl")
    assert read_community_summaries(GRAPH / "community_summaries.json") == (
        read_community_summaries(LEGACY / "graph" / "community_summaries.json")
    )
    current = read_graph_meta(GRAPH / "graph_meta.json")
    legacy = read_graph_meta(LEGACY / "graph" / "graph_meta.json")
    assert current == legacy
    assert current["khop_depth"] == 2 and current["backend"] == "graph"


def test_current_and_pre_contract_packages_load_to_the_same_candidates() -> None:
    current = load_candidates(PACKAGE / "candidates.json")
    legacy = load_candidates(LEGACY / "prompt-system" / "candidates.json")
    assert current == legacy
    assert current[0].prompt_system_id and current[0].fields.anthology_size == 1


def test_a_no_tree_control_reads_the_same_whether_it_wrote_none_or_the_empty_object() -> None:
    """Every tree-enabled grid writes controls, so the empty object is the common old form."""
    control, tree_candidate = load_candidates(LEGACY / "prompt-system" / "candidates.json")
    assert control.knowledge_tree == {}
    assert tree_candidate.knowledge_tree["baseline_prompt_system_id"] == control.prompt_system_id
    assert tree_candidate.knowledge_tree["used_tokens"] == 96


def test_store_manifest_binds_opaque_members_by_owner_and_format_version() -> None:
    manifest = store_dataset_manifest(STORE)
    opaque = {
        member.member_id: member.opaque_binding
        for member in manifest.members
        if member.opaque_binding is not None
    }
    assert opaque["vector-index-faiss"].owner == "faiss"
    assert opaque["vector-index-faiss"].format_version == "IndexFlatIP/1"
    assert opaque["lexical-index"].format_version == "bm25-uk-v2"
    structured = [m for m in manifest.members if m.record_contract is not None]
    assert {m.record_contract.schema_id for m in structured} == {
        "llb.rag-store-meta",
        "llb.rag-chunk",
    }


def test_published_store_manifest_matches_the_committed_bytes() -> None:
    published = load_dataset_manifest(STORE)
    assert published is not None and published.schema_version == "1.0.0"
    assert member_digest_problems(STORE, published) == ()


def test_a_changed_index_refuses_before_the_store_is_read(tmp_path: Path) -> None:
    target = tmp_path / "store"
    shutil.copytree(STORE, target)
    (target / "index.faiss").write_text("a different index entirely\n", encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="digest mismatch"):
        refuse_tampered_dataset(target)


def test_a_store_published_without_a_manifest_is_not_a_refusal() -> None:
    assert not (LEGACY / "store" / DATASET_MANIFEST_FILE).exists()
    refuse_tampered_dataset(LEGACY / "store")


def test_a_future_store_or_graph_refuses_before_query_execution() -> None:
    with pytest.raises(UnsupportedFutureVersionError):
        read_store_meta(FUTURE / "store_meta.json")
    with pytest.raises(UnsupportedFutureVersionError):
        read_graph_meta(FUTURE / "graph_meta.json")


def test_a_future_package_member_refuses_at_the_prompt_system_gate(tmp_path: Path) -> None:
    target = tmp_path / "package"
    shutil.copytree(PACKAGE, target)
    record = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    (target / "manifest.json").write_text(
        json.dumps({**record, "schema_version": "9.0.0"}), encoding="utf-8"
    )
    with pytest.raises(ArtifactCompatibilityError, match="prompt-system package cannot be read"):
        refuse_unreadable_prompt_system(target)


def test_the_prompt_system_gate_passes_a_pre_contract_package() -> None:
    refuse_unreadable_prompt_system(LEGACY / "prompt-system")


def test_a_pre_contract_store_needs_no_upgrade_while_every_family_is_at_its_initial_version(
    tmp_path: Path,
) -> None:
    """Nothing here has been released, so a pre-contract file is read at the only version there is."""
    target = tmp_path / "store"
    shutil.copytree(LEGACY / "store", target)
    readings = survey_dataset(target, store_dataset_manifest(target))
    assert readings and not any(reading.needs_upgrade for reading in readings)

    before = {path.name: path.read_bytes() for path in sorted(target.iterdir())}
    assert upgrade_dataset(target, store_dataset_manifest(target)) == ()
    assert {path.name: path.read_bytes() for path in sorted(target.iterdir())} == before


def test_every_graph_and_package_member_reads_at_the_current_contract() -> None:
    for root, manifest in (
        (GRAPH, graph_dataset_manifest(GRAPH)),
        (PACKAGE, prompt_system_dataset_manifest(PACKAGE)),
    ):
        readings = survey_dataset(root, manifest)
        assert readings and all(not reading.refusal for reading in readings)
        assert all(reading.records in (1, 2) for reading in readings)


def test_a_comparison_sidecar_round_trips_and_reads_its_pre_contract_form(tmp_path: Path) -> None:
    body = {"k": 10, "rows": {"vector": {"recall_at_k": 0.5}}}
    path = write_sidecar(tmp_path / "comparison.json", "comparison", "compare-retrieval", body)
    record = read_sidecar(path)
    assert record["kind"] == "comparison" and record["produced_by"] == "compare-retrieval"
    assert record["report"] == body

    bare = tmp_path / "archived.json"
    bare.write_text(json.dumps(body), encoding="utf-8")
    archived = read_sidecar(bare)
    assert archived["report"] == body
    # An archived sidecar recorded no producer, and the envelope says so rather than inventing one.
    assert archived["produced_by"] == "unrecorded"


def test_a_query_glossary_round_trips_through_its_contract(tmp_path: Path) -> None:
    path = tmp_path / "query_glossary.json"
    record = {
        "schema_id": "llb.query-glossary",
        "schema_version": "1.0.0",
        "version": "query-glossary-v1",
        "entries": [{"canonical": "akt", "aliases": ["act"]}],
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    current = Glossary.load(path)

    legacy = tmp_path / "legacy_glossary.json"
    legacy.write_text(
        json.dumps({"version": "query-glossary-v1", "entries": record["entries"]}),
        encoding="utf-8",
    )
    assert Glossary.load(legacy) == current


def test_a_structured_member_of_the_wrong_family_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "store"
    shutil.copytree(STORE, target)
    rows = (STORE / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(rows[0])
    (target / "chunks.jsonl").write_text(
        json.dumps({**record, "schema_id": "llb.graph-node"}) + "\n", encoding="utf-8"
    )
    manifest = store_dataset_manifest(target)
    member = next(m for m in manifest.members if m.member_id == "chunks")
    from llb.artifacts.default_registry import DEFAULT_REGISTRY
    from llb.artifacts.io import read_bound_member

    with pytest.raises(DatasetReadError, match="expected schema_id"):
        read_bound_member(target, member, DEFAULT_REGISTRY)
