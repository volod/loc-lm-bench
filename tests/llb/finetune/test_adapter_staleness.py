"""Tests for adapter staleness."""

from pathlib import Path
import pytest
from llb.finetune.guard import validate_adapter_for_eval
from llb.finetune.registry.io import load_registry, registry_path
from llb.finetune.registry.model import (
    VERDICT_CURRENT,
    VERDICT_STALE,
    VERDICT_UNKNOWN,
)
from llb.finetune.registry.register import register_adapter
from llb.finetune.registry.staleness import staleness
from llb.goldset.schema import dump_goldset
from adapter_registry_gc_helpers import _store_meta
from adapter_registry_helpers import (
    FIXTURE_REGISTRY,
    LAUNDERED_ADAPTER,
    POISONED_ADAPTER,
    STALE_FIXTURE_ID,
    _corpus,
    _goldset,
    _item,
    _trained_adapter,
)


def test_staleness_flips_when_the_goldset_digest_changes(tmp_path: Path):
    goldset = _goldset(tmp_path, _item("tune-1", "tuning"))
    corpus = _corpus(tmp_path)
    entry = register_adapter(
        registry=registry_path(tmp_path),
        adapter_dir=_trained_adapter(tmp_path),
        goldset_path=goldset,
        corpus_root=corpus,
        index_dir=_store_meta(tmp_path),
    )
    assert staleness(entry).verdict == VERDICT_CURRENT

    dump_goldset([_item("tune-1", "tuning"), _item("tune-2", "tuning")], goldset)

    report = staleness(entry)
    assert report.verdict == VERDICT_STALE
    assert report.is_stale
    assert report.reasons == ("goldset changed since training",)


def test_staleness_is_unknown_when_a_digest_was_never_recorded(tmp_path: Path):
    """A missing corpus digest can never read as `current` -- absence of evidence is not evidence."""
    entry = register_adapter(
        registry=registry_path(tmp_path),
        adapter_dir=_trained_adapter(tmp_path),
        goldset_path=_goldset(tmp_path),
        index_dir=_store_meta(tmp_path),
    )

    report = staleness(entry)

    assert report.verdict == VERDICT_UNKNOWN
    assert report.reasons == ("corpus digest unavailable",)


def test_staleness_flips_when_the_store_embedder_changes(tmp_path: Path):
    """Same corpus fingerprint, different retrieval knobs -> the training contexts are gone."""
    goldset = _goldset(tmp_path, _item("tune-1", "tuning"))
    corpus = _corpus(tmp_path)
    index_dir = _store_meta(tmp_path)
    entry = register_adapter(
        registry=registry_path(tmp_path),
        adapter_dir=_trained_adapter(tmp_path),
        goldset_path=goldset,
        corpus_root=corpus,
        index_dir=index_dir,
    )
    assert staleness(entry).verdict == VERDICT_CURRENT

    _store_meta(tmp_path, embedding_model="other/embedder")  # rebuild with another embedder

    report = staleness(entry)
    assert report.verdict == VERDICT_STALE
    assert report.reasons == (
        "retrieval embedding_model changed since training "
        "(intfloat/multilingual-e5-base -> other/embedder)",
    )

    _store_meta(tmp_path, size=400)  # rechunk instead: a different knob is named
    rechunked = staleness(entry)
    assert rechunked.verdict == VERDICT_STALE
    assert any("chunk_size" in reason for reason in rechunked.reasons)


def test_staleness_without_an_index_fingerprint_reads_unknown(tmp_path: Path):
    """An entry without an index cannot claim that its retrieval evidence is current."""
    goldset = _goldset(tmp_path, _item("tune-1", "tuning"))
    corpus = _corpus(tmp_path)
    entry = register_adapter(
        registry=registry_path(tmp_path),
        adapter_dir=_trained_adapter(tmp_path),
        goldset_path=goldset,
        corpus_root=corpus,
    )
    report = staleness(entry)
    assert report.verdict == VERDICT_UNKNOWN
    assert report.reasons == ("retrieval fingerprint unavailable",)


def test_retrieval_fingerprint_round_trips_through_the_event_log(tmp_path: Path):
    index_dir = _store_meta(tmp_path)
    entry = register_adapter(
        registry=registry_path(tmp_path),
        adapter_dir=_trained_adapter(tmp_path),
        goldset_path=_goldset(tmp_path),
        index_dir=index_dir,
    )
    loaded = load_registry(registry_path(tmp_path))[entry.adapter_id]
    assert loaded.retrieval_fingerprint == {
        "embedding_model": "intfloat/multilingual-e5-base",
        "strategy": "markdown",
        "chunk_size": 800,
        "chunk_overlap": 120,
        "retrieval_mode": "flat",
    }
    assert loaded.index_dir == str(index_dir)


def test_committed_fixture_stamps_the_stale_entry():
    entries = load_registry(FIXTURE_REGISTRY)
    stale = entries[STALE_FIXTURE_ID]

    report = staleness(stale)

    assert report.verdict == VERDICT_STALE
    assert [merge["backend"] for merge in stale.merges] == ["ollama"]


def test_guard_reads_recorded_digests_not_the_adapter_manifest():
    """The laundered manifest claims a clean tuning set; the registry records the final-split ids."""
    protected = [_item("sample-final-item", "final")]

    validate_adapter_for_eval(adapter_path=LAUNDERED_ADAPTER, items=protected, model="sample/base")

    with pytest.raises(SystemExit, match="sample-final-item"):
        validate_adapter_for_eval(
            adapter_path=LAUNDERED_ADAPTER,
            items=protected,
            model="sample/base",
            registry=FIXTURE_REGISTRY,
        )


def test_guard_still_refuses_an_unregistered_poisoned_manifest():
    with pytest.raises(SystemExit, match="sample-final-item"):
        validate_adapter_for_eval(
            adapter_path=POISONED_ADAPTER,
            items=[_item("sample-final-item", "final")],
            model="sample/base",
            registry=FIXTURE_REGISTRY,
        )
