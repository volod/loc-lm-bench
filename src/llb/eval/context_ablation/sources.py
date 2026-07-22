"""Prompt context for the two lanes that do not retrieve (rag-vs-long-context-ablation).

A context source replaces store retrieval inside the ordinary eval graph and returns exactly what
the retrieve node would (`retrieved` / `context`, plus an optional terminal `status`). Everything
downstream -- scoring, the retrieval sidecar, the manifest, the bundle layout -- is unchanged, so
each lane persists an ORDINARY `run-eval` bundle that reproduces from its own config.

The long-context lane is deliberately oracle-grounded: it lays the item's own gold source
document(s) into the prompt, because the question this lane asks is whether whole-document
stuffing beats chunked retrieval WHEN the right document is known. That makes it a diagnostic
ceiling, never a leaderboard row.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple, cast

from llb.core.config import RunConfig
from llb.core.contracts.models import ModelSpec
from llb.core.contracts.rag import ChunkRecord
from llb.eval import common as eval_common
from llb.eval.context_ablation.models import (
    LANE_CLOSED_BOOK,
    LANE_LONG_CONTEXT,
    LANE_RAG,
)
from llb.eval.graph import CLOSED_BOOK_TEMPLATE, ContextSource, RagState

# True when a context of that many characters fits the model's usable window.
FitsContext = Callable[[int], bool]

DEFAULT_MODELS_MANIFEST = Path("samples/configs/models_uk.yaml")


class ContextLane(NamedTuple):
    """How a non-RAG context strategy fills the prompt: its context source and its prompt id."""

    source: ContextSource
    template_id: str | None


def closed_book_source() -> ContextSource:
    """No context at all -- the model answers from its weights.

    An empty context is the POINT of this lane, so it must not raise `retrieval_miss`: that status
    short-circuits generation, and a lane that never calls the model measures nothing.
    """

    def source(state: RagState) -> RagState:
        return {"retrieved": [], "context": "", "retrieve_latency_s": 0.0}

    return source


def whole_document_chunk(doc_id: str, text: str) -> ChunkRecord:
    """The item's whole source document as one offset-exact chunk."""
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}#{LANE_LONG_CONTEXT}",
        "char_start": 0,
        "char_end": len(text),
        "text": text,
        "strategy": LANE_LONG_CONTEXT,
        "metadata": {"context_strategy": LANE_LONG_CONTEXT},
    }


def long_context_source(documents: Mapping[str, str], fits: FitsContext) -> ContextSource:
    """The item's whole gold document(s), or a `context_overflow` skip when they do not fit.

    A document that exceeds the usable window is SKIPPED rather than truncated: a truncated
    document is a different (and unstated) retrieval policy, and crediting its answer to
    "long context" would measure whichever slice of the document happened to survive the cut.
    """

    def source(state: RagState) -> RagState:
        doc_ids = list(dict.fromkeys(str(span["doc_id"]) for span in state.get("gold_spans", [])))
        if not doc_ids:
            return {"retrieved": [], "context": "", "status": eval_common.RETRIEVAL_MISS}
        missing = [doc_id for doc_id in doc_ids if doc_id not in documents]
        if missing:
            raise SystemExit(
                f"[context-ablation] the long_context lane needs every gold document in the "
                f"corpus, but {', '.join(missing[:3])} is not there; point --corpus-root at the "
                f"corpus this gold set was labeled against"
            )
        chunks = [whole_document_chunk(doc_id, documents[doc_id]) for doc_id in doc_ids]
        total_chars = sum(len(documents[doc_id]) for doc_id in doc_ids)
        if not fits(total_chars):
            return {"retrieved": [], "context": "", "status": eval_common.CONTEXT_OVERFLOW}
        return {
            "retrieved": chunks,
            "context": eval_common.format_context(chunks),
            "retrieve_latency_s": 0.0,
        }

    return source


def load_corpus_documents(corpus_root: Path) -> dict[str, str]:
    """Every corpus document keyed by the `doc_id` a gold span names (its relative path)."""
    from llb.rag.chunking.corpus import iter_docs

    root = Path(corpus_root)
    if not root.is_dir():
        raise SystemExit(f"[context-ablation] corpus not found: {root}")
    documents = dict(iter_docs(root))
    if not documents:
        raise SystemExit(f"[context-ablation] no .txt/.md document under {root}")
    return documents


def resolve_model_spec(
    model: str, backend: str | None = None, manifest: Path = DEFAULT_MODELS_MANIFEST
) -> ModelSpec | None:
    """Best-effort planning spec for the SERVED artifact `model` (None when the manifest has none).

    A roster entry names its per-backend artifacts under `sources` -- the run is served by an
    Ollama GGUF tag, not by the entry's headline HF repo id -- so the lookup goes through
    `candidate_sources` and returns the spec priced for the artifact that actually runs.

    None is not a failure: without a spec only an explicit `context_budget` / `max_model_len` can
    bound the prompt, so an unlisted model skips nothing instead of skipping everything.
    """
    from llb.backends.prepare.manifest import load_manifest
    from llb.backends.resolver_sources import candidate_sources

    try:
        specs = load_manifest(manifest)
    except (OSError, ValueError):
        return None
    for spec in specs:
        if spec.get("name") == model:
            return spec
        for source_backend, record in candidate_sources(spec):
            if record.get("source") != model:
                continue
            if backend is not None and source_backend != backend:
                continue
            return cast(ModelSpec, {**spec, "backend": source_backend, **record})
    return None


def context_fit_check(
    config: RunConfig,
    *,
    model_spec: ModelSpec | None = None,
    vram_mib: int | None = None,
    ram_mib: int | None = None,
) -> FitsContext:
    """A `fits(context_chars)` predicate for this run, resolved once per lane, not per item."""
    from llb.backends.hardware import detect_gpus, detect_ram_mb, max_vram_mb
    from llb.optimize.tuning_space import fits_context_chars

    spec = (
        model_spec if model_spec is not None else resolve_model_spec(config.model, config.backend)
    )
    vram = vram_mib if vram_mib is not None else max_vram_mb(detect_gpus())
    ram = ram_mib if ram_mib is not None else detect_ram_mb()

    def fits(context_chars: int) -> bool:
        return fits_context_chars(config, spec, vram, ram, context_chars)

    return fits


def build_context_lane(config: RunConfig, fits: FitsContext | None = None) -> ContextLane | None:
    """The context source + generation prompt for `config.context_strategy` (None for `rag`)."""
    if config.context_strategy == LANE_RAG:
        return None
    if config.context_strategy == LANE_CLOSED_BOOK:
        return ContextLane(closed_book_source(), CLOSED_BOOK_TEMPLATE)
    if config.context_strategy == LANE_LONG_CONTEXT:
        documents = load_corpus_documents(config.corpus_root)
        return ContextLane(long_context_source(documents, fits or context_fit_check(config)), None)
    raise ValueError(f"unknown context strategy: {config.context_strategy!r}")
