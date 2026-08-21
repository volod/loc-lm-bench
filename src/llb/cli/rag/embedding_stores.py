"""Store builders used by the embedding bake-off command."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from llb.cli.rag.compare_stores import _dir_size_bytes
from llb.rag.encoders.throughput import (
    DEFAULT_MAX_WARM_PASSES,
    DEFAULT_MAX_WARM_SECONDS,
    DEFAULT_MIN_WARM_PASSES,
    DEFAULT_TARGET_RELATIVE_PRECISION,
    ThroughputProfile,
)
from llb.rag.encoders.throughput_profile import profile_local_embedder

if TYPE_CHECKING:
    from llb.core.config import RunConfig
    from llb.prep.frontier.telemetry import ProvenanceLog
    from llb.rag.embedding_bakeoff.models import StoreBuilder

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncoderThroughputOptions:
    """Warm/cold decomposition knobs for `compare-embeddings --encoder-throughput`."""

    enabled: bool = False
    devices: tuple[str, ...] = ()
    target_relative_precision: float = DEFAULT_TARGET_RELATIVE_PRECISION
    min_warm_passes: int = DEFAULT_MIN_WARM_PASSES
    max_warm_passes: int = DEFAULT_MAX_WARM_PASSES
    max_warm_seconds: float = DEFAULT_MAX_WARM_SECONDS
    compare_cpu: bool = False


def local_store_builder(
    cfg: "RunConfig",
    stores_dir: Path,
    *,
    throughput: EncoderThroughputOptions | None = None,
    vram_reader: Callable[[], int] | None = None,
    power_reader: Callable[[], float | None] | None = None,
    profiles_out: list[ThroughputProfile] | None = None,
) -> "StoreBuilder":
    """Build and persist one local store per embedding model.

    When `throughput.enabled`, also runs the cold/warm encoder decomposition over the indexed
    chunk texts and attaches the primary-device profile to the `BuiltStore`. Extra device profiles
    (e.g. CPU beside CUDA) append to `profiles_out` for the host summary.
    """
    from llb.rag.embedding_bakeoff.models import BuiltStore, slugify_model
    from llb.rag.vector_store.store import RagStore

    options = throughput or EncoderThroughputOptions()

    def build(model: str) -> "BuiltStore":
        started = time.perf_counter()
        store = RagStore.build(
            cfg.corpus_root,
            cfg.strategy,
            cfg.chunk_size,
            cfg.chunk_overlap,
            model,
            mode=cfg.retrieval_mode,
            child_size=cfg.child_chunk_size,
            lexical_lemmas=cfg.lexical_lemmas,
        )
        embed_seconds = time.perf_counter() - started
        out_dir = stores_dir / slugify_model(model)
        store.save(out_dir)
        resolve = getattr(store.embedder, "_resolve_device", None)
        device = resolve() if callable(resolve) else None
        # Read the precision the weights are actually held at BEFORE releasing them: a row's
        # throughput is only comparable beside the dtype it was measured at.
        read_dtype = getattr(store.embedder, "effective_dtype", None)
        dtype = read_dtype() if callable(read_dtype) else None
        # Release build weights before the cold/warm profile so peak VRAM is per-candidate,
        # not stacked across the bake-off roster (retrieve() lazy-reloads when scoring).
        release = getattr(store.embedder, "release", None)
        if callable(release):
            release()
        profile = None
        if options.enabled:
            texts = [str(chunk["text"]) for chunk in store.chunks]
            profile = _attach_throughput_profiles(
                model,
                texts,
                resolved_device=device,
                options=options,
                vram_reader=vram_reader,
                power_reader=power_reader,
                profiles_out=profiles_out,
            )
        return BuiltStore(
            store=store,
            embed_seconds=embed_seconds,
            index_bytes=_dir_size_bytes(out_dir),
            device=device,
            dtype=dtype,
            throughput_profile=profile,
        )

    return build


def _attach_throughput_profiles(
    model: str,
    texts: Sequence[str],
    *,
    resolved_device: str | None,
    options: EncoderThroughputOptions,
    vram_reader: Callable[[], int] | None,
    power_reader: Callable[[], float | None] | None,
    profiles_out: list[ThroughputProfile] | None,
) -> ThroughputProfile:
    """Profile the primary device (and optional CPU twin); return the primary profile."""
    primary = _primary_device(options.devices, resolved_device)
    devices = [primary]
    if options.compare_cpu and primary != "cpu" and "cpu" not in devices:
        devices.append("cpu")
    primary_profile: ThroughputProfile | None = None
    for device in devices:
        _LOG.info("[compare-embeddings] encoder-throughput model=%s device=%s", model, device)
        profile = profile_local_embedder(
            model,
            texts,
            device=device,
            target_relative_precision=options.target_relative_precision,
            min_warm_passes=options.min_warm_passes,
            max_warm_passes=options.max_warm_passes,
            max_warm_seconds=options.max_warm_seconds,
            vram_reader=vram_reader if device == "cuda" else None,
            power_reader=power_reader if device == "cuda" else None,
        )
        if profiles_out is not None:
            profiles_out.append(profile)
        if device == primary:
            primary_profile = profile
    assert primary_profile is not None
    return primary_profile


def _primary_device(requested: tuple[str, ...], resolved: str | None) -> str:
    if requested:
        return requested[0]
    return resolved or "cpu"


def api_store_builder(
    cfg: "RunConfig", stores_dir: Path, log: "ProvenanceLog", max_usd: Optional[float]
) -> "StoreBuilder":
    """Build and persist the API-embedded store while recording its cost."""
    from llb.rag.encoders.api import ApiEmbedder, litellm_embed
    from llb.rag.embedding_bakeoff.models import KIND_API, BuiltStore, slugify_model
    from llb.rag.vector_store.store import RagStore

    def build(model: str) -> "BuiltStore":
        embedder = ApiEmbedder(model, litellm_embed(model, log=log, max_usd=max_usd))
        started = time.perf_counter()
        store = RagStore.build(
            cfg.corpus_root,
            cfg.strategy,
            cfg.chunk_size,
            cfg.chunk_overlap,
            model,
            mode=cfg.retrieval_mode,
            child_size=cfg.child_chunk_size,
            embedder=embedder,
            lexical_lemmas=cfg.lexical_lemmas,
        )
        embed_seconds = time.perf_counter() - started
        out_dir = stores_dir / slugify_model(model)
        store.save(out_dir)
        return BuiltStore(
            store=store,
            embed_seconds=embed_seconds,
            index_bytes=_dir_size_bytes(out_dir),
            kind=KIND_API,
            cost_usd=log.summary()["total_cost_usd"],
        )

    return build
