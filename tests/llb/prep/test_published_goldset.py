import hashlib
import json

import pytest

from llb.goldset.schema import load_goldset
from llb.goldset.validate import validate_items
from llb.prep.published_goldset import SOURCE_SHA256, build_fixture, select_context_diverse
from llb.prep.ua_squad_source import DATASET_ID, DATASET_REVISION, DATASET_SPLIT
from llb.rag.chunking.corpus import chunk_corpus
from llb.core.paths import PROJECT_ROOT

FIXTURE_ROOT = PROJECT_ROOT / "samples" / "goldsets" / "ua_squad_postedited_v1"


def test_committed_published_fixture_is_canonical_and_verified():
    items = load_goldset(FIXTURE_ROOT / "goldset.jsonl")
    assert len(items) == 250
    assert len({item.id for item in items}) == 250
    assert all(item.lang == "uk" and item.provenance == "public-reused" for item in items)
    assert all(item.verified for item in items)
    assert validate_items(items, FIXTURE_ROOT / "corpus")["errors"] == []
    metadata = json.loads((FIXTURE_ROOT / "source.json").read_text(encoding="utf-8"))
    assert metadata["dataset"] == DATASET_ID
    assert metadata["revision"] == DATASET_REVISION
    assert metadata["split"] == DATASET_SPLIT
    assert metadata["source_sha256"] == SOURCE_SHA256
    assert metadata["items"] == 250 and metadata["documents"] == 250


def test_committed_fixture_runs_through_real_chunking_pipeline():
    chunks = chunk_corpus(FIXTURE_ROOT / "corpus", "fixed", size=800, overlap=120)
    assert len(chunks) >= 250
    assert len({chunk["doc_id"] for chunk in chunks}) == 250
    assert all(chunk["text"] for chunk in chunks)


def test_select_context_diverse_skips_ungrounded_and_duplicate_contexts():
    records = [
        {
            "id": "bad",
            "context": "abc",
            "question": "q",
            "answers": {"text": ["x"], "answer_start": [0]},
        },
        {
            "id": "a",
            "context": "abc",
            "question": "q",
            "answers": {"text": ["a"], "answer_start": [0]},
        },
        {
            "id": "dup",
            "context": "abc",
            "question": "q2",
            "answers": {"text": ["b"], "answer_start": [1]},
        },
        {
            "id": "b",
            "context": "def",
            "question": "q",
            "answers": {"text": ["d"], "answer_start": [0]},
        },
    ]
    assert [row["id"] for row in select_context_diverse(records, 2)] == ["a", "b"]


def test_fixture_builder_rejects_unpinned_source(tmp_path):
    source = tmp_path / "val.json"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected val.json SHA-256"):
        build_fixture(source, tmp_path / "out", 1)
    assert hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256
