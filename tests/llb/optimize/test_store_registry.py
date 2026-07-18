"""Unit tests for multi-objective store prewarm / fingerprint reuse."""

from types import SimpleNamespace

from llb.core.config import RunConfig
from llb.optimize.store_registry import (
    StoreRegistry,
    fingerprint_slug,
    store_fingerprint,
    study_stores_dir,
)


def _fake_builder(calls: list[tuple]) -> object:
    """Return a builder that records each embed and yields attribute-friendly fakes."""

    def build(config: RunConfig) -> SimpleNamespace:
        key = store_fingerprint(config)
        calls.append(key)
        return SimpleNamespace(
            embedding_model=config.embedding_model,
            strategy=config.strategy,
            chunk_size=config.chunk_size,
            meta={"embedding_model": config.embedding_model},
        )

    return build


def test_store_registry_second_get_issues_zero_embeds():
    """Acceptance: two embedders, fixed chunking -- reuse issues zero new embeds."""
    calls: list[tuple] = []
    embedders = ["intfloat/multilingual-e5-base", "BAAI/bge-m3"]
    base = RunConfig(strategy="recursive", chunk_size=512, chunk_overlap=64)
    registry = StoreRegistry(embedders=embedders, builder=_fake_builder(calls))

    built = registry.prewarm(base, embedders)
    assert built == 2
    assert registry.embed_calls == 2
    assert len(calls) == 2

    # Second pass over the same fingerprints (as a later Optuna trial would).
    for model in embedders:
        cfg = base.with_overrides(embedding_model=model)
        store = registry.get(cfg)
        assert store.embedding_model == model
        assert store.fusion_weight == cfg.fusion_weight

    assert registry.embed_calls == 2
    assert len(calls) == 2


def test_store_registry_fanout_on_new_chunking_fingerprint():
    """First sight of a chunking shape builds every shortlist embedder once."""
    calls: list[tuple] = []
    embedders = ["e5-base", "bge-m3"]
    registry = StoreRegistry(embedders=embedders, builder=_fake_builder(calls))
    cfg = RunConfig(
        embedding_model="e5-base",
        strategy="fixed",
        chunk_size=256,
        chunk_overlap=32,
    )
    registry.get(cfg)
    assert registry.embed_calls == 2
    assert {c[0] for c in calls} == set(embedders)

    # Same chunking, other embedder: no new embeds.
    other = cfg.with_overrides(embedding_model="bge-m3")
    registry.get(other)
    assert registry.embed_calls == 2


def test_store_registry_disk_cache_skips_second_embed(tmp_path, monkeypatch):
    """A second registry with the same cache_dir reloads instead of re-embedding."""
    calls: list[tuple] = []
    embedders = ["e5-base", "bge-m3"]
    cache_dir = study_stores_dir(tmp_path, "mo-disk")
    cache_dir.mkdir(parents=True)

    class _DiskFake(SimpleNamespace):
        def save(self, index_dir) -> None:
            from pathlib import Path

            path = Path(index_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "store_marker.txt").write_text(self.embedding_model, encoding="utf-8")

    def builder(config: RunConfig) -> _DiskFake:
        calls.append(store_fingerprint(config))
        return _DiskFake(embedding_model=config.embedding_model, meta={})

    first = StoreRegistry(cache_dir=cache_dir, embedders=embedders, builder=builder)
    base = RunConfig(embedding_model="e5-base", chunk_size=400, chunk_overlap=40)
    assert first.prewarm(base, embedders) == 2
    assert first.embed_calls == 2

    class _FakeRagStore:
        @staticmethod
        def load(index_dir):
            from pathlib import Path

            model = (Path(index_dir) / "store_marker.txt").read_text(encoding="utf-8")
            return SimpleNamespace(embedding_model=model, meta={"embedding_model": model})

    import llb.rag.store as rag_store

    monkeypatch.setattr(rag_store, "RagStore", _FakeRagStore)

    second = StoreRegistry(cache_dir=cache_dir, embedders=embedders, builder=builder)
    assert second.prewarm(base, embedders) == 0
    assert second.embed_calls == 0
    assert len(calls) == 2  # only the first registry embedded


def test_fingerprint_slug_stable():
    key = ("BAAI/bge-m3", "recursive", 512, 64, "flat", 400, False)
    assert fingerprint_slug(key) == fingerprint_slug(key)
    assert len(fingerprint_slug(key)) == 16
