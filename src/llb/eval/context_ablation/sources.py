"""Prompt context for the lanes that do not build it from ranked chunks.

Part of rag-vs-long-context-ablation.

Two seams, because the lanes split two ways. A context SOURCE replaces store retrieval entirely
and returns exactly what the retrieve node would (`retrieved` / `context`, plus an optional
terminal `status`) -- that is `closed_book` and `long_context`, neither of which retrieves. A
context REFINER runs after ordinary retrieval and rewrites what the prompt carries -- that is
`retrieved_document`, which retrieves exactly as the `rag` lane does and only widens the unit of
context afterwards. Everything downstream -- scoring, the retrieval sidecar, the manifest, the
bundle layout -- is unchanged, so each lane persists an ORDINARY `run-eval` bundle that reproduces
from its own config.

The long-context lane is deliberately oracle-grounded: it lays the item's own gold source
document(s) into the prompt, because the question this lane asks is whether whole-document
stuffing beats chunked retrieval WHEN the right document is known. That makes it a diagnostic
ceiling, never a leaderboard row. `retrieved_document` is the shippable sibling that answers the
operator's version of the same question -- send the document, but the one RETRIEVAL picked -- so
the oracle gap splits into a part an operator can capture and a part that was the gold label.
"""

from collections.abc import Callable, Mapping, Sequence
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
    LANE_RETRIEVED_DOCUMENT,
)
from llb.eval.graph import CLOSED_BOOK_TEMPLATE
from llb.eval.graph_contracts import ContextRefiner, ContextSource, RagState

# True when a context of that many characters fits the model's usable window.
FitsContext = Callable[[int], bool]

DEFAULT_MODELS_MANIFEST = Path("samples/configs/models_uk.yaml")


class ContextLane(NamedTuple):
    """How a non-RAG context strategy fills the prompt.

    A lane sets exactly one of the two seams: `source` for a lane that does not retrieve at all,
    `refiner` for a lane that retrieves and then rewrites the context. `template_id` overrides the
    generation prompt (only `closed_book` needs to, so the other lanes' deltas stay attributable
    to the context rather than to prompt wording).
    """

    source: ContextSource | None = None
    template_id: str | None = None
    refiner: ContextRefiner | None = None


def closed_book_source() -> ContextSource:
    """No context at all -- the model answers from its weights.

    An empty context is the POINT of this lane, so it must not raise `retrieval_miss`: that status
    short-circuits generation, and a lane that never calls the model measures nothing.
    """

    def source(state: RagState) -> RagState:
        return {"retrieved": [], "context": "", "retrieve_latency_s": 0.0}

    return source


def whole_document_chunk(doc_id: str, text: str, strategy: str = LANE_LONG_CONTEXT) -> ChunkRecord:
    """A whole source document as one offset-exact chunk, tagged with the lane that laid it in."""
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}#{strategy}",
        "char_start": 0,
        "char_end": len(text),
        "text": text,
        "strategy": strategy,
        "metadata": {"context_strategy": strategy},
    }


def document_context(
    doc_ids: Sequence[str], documents: Mapping[str, str], fits: FitsContext, strategy: str
) -> RagState:
    """`doc_ids` laid into the prompt whole, or a `context_overflow` skip when they do not fit.

    A document that exceeds the usable window is SKIPPED rather than truncated: a truncated
    document is a different (and unstated) retrieval policy, and crediting its answer to a
    document lane would measure whichever slice of the document happened to survive the cut. Both
    document lanes share this rule, so their deltas differ only in how the documents were CHOSEN.
    """
    missing = [doc_id for doc_id in doc_ids if doc_id not in documents]
    if missing:
        raise SystemExit(
            f"[context-ablation] the {strategy} lane needs every named document in the corpus, "
            f"but {', '.join(missing[:3])} is not there; point --corpus-root at the corpus this "
            f"gold set was labeled against and this index was built from"
        )
    if not fits(sum(len(documents[doc_id]) for doc_id in doc_ids)):
        return {"retrieved": [], "context": "", "status": eval_common.CONTEXT_OVERFLOW}
    chunks = [whole_document_chunk(doc_id, documents[doc_id], strategy) for doc_id in doc_ids]
    return {"retrieved": chunks, "context": eval_common.format_context(chunks)}


def long_context_source(documents: Mapping[str, str], fits: FitsContext) -> ContextSource:
    """The item's whole GOLD document(s) -- the oracle ceiling, never a shippable policy."""

    def source(state: RagState) -> RagState:
        doc_ids = list(dict.fromkeys(str(span["doc_id"]) for span in state.get("gold_spans", [])))
        if not doc_ids:
            return {"retrieved": [], "context": "", "status": eval_common.RETRIEVAL_MISS}
        return {
            **document_context(doc_ids, documents, fits, LANE_LONG_CONTEXT),
            "retrieve_latency_s": 0.0,
        }

    return source


def ranked_doc_ids(chunks: Sequence[ChunkRecord], top_n: int) -> list[str]:
    """The first `top_n` DISTINCT documents in the ranked chunk list, best-first.

    De-duplicating by document is what makes `top_n` a document budget rather than a chunk budget:
    three chunks of one document are one document, and asking for two documents keeps walking the
    ranking until a second one appears.
    """
    ordered = list(dict.fromkeys(str(chunk["doc_id"]) for chunk in chunks))
    return ordered[:top_n]


def retrieved_document_refiner(
    documents: Mapping[str, str], fits: FitsContext, top_n: int = 1
) -> ContextRefiner:
    """Replace the retrieved chunks in the prompt with the whole documents they came from.

    The gold label is nowhere in this path: the documents are selected by the SAME ranking the
    `rag` lane answers from, so whatever this lane gains over `rag` is a gain an operator can
    capture by widening the unit of retrieval. `retrieved` is rewritten to the documents actually
    laid in, which makes the lane's recall@k read document-level -- how often the selection rule
    picked a document that holds the answer -- and that is exactly this lane's own ceiling.
    """

    def refine(state: RagState, update: RagState) -> RagState:
        doc_ids = ranked_doc_ids(update.get("retrieved", []), top_n)
        if not doc_ids:
            return update
        refined = {
            **update,
            **document_context(doc_ids, documents, fits, LANE_RETRIEVED_DOCUMENT),
            # Whole documents replace the chunk block outright, so no restored chunk and no
            # restored header reaches the prompt; the accounting must not claim one did.
            "table_headers_restored": 0,
            "table_header_chars": 0,
        }
        refined.pop("prompt_chunks", None)
        return cast(RagState, refined)

    return refine


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
    """The context seam + generation prompt for `config.context_strategy` (None for `rag`)."""
    if config.context_strategy == LANE_RAG:
        return None
    if config.context_strategy == LANE_CLOSED_BOOK:
        return ContextLane(source=closed_book_source(), template_id=CLOSED_BOOK_TEMPLATE)
    if config.context_strategy == LANE_LONG_CONTEXT:
        documents = load_corpus_documents(config.corpus_root)
        return ContextLane(source=long_context_source(documents, fits or context_fit_check(config)))
    if config.context_strategy == LANE_RETRIEVED_DOCUMENT:
        documents = load_corpus_documents(config.corpus_root)
        return ContextLane(
            refiner=retrieved_document_refiner(
                documents, fits or context_fit_check(config), config.retrieved_document_top_n
            )
        )
    raise ValueError(f"unknown context strategy: {config.context_strategy!r}")
