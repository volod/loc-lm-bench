"""Focused store validation implementation."""

from pathlib import Path
from llb.core.config_validation import (
    DEFAULT_EMBEDDING_MODEL,
)
from llb.core.contracts.rag import RagStoreMeta
from llb.prep.corpus.fingerprints import corpus_fingerprint
from llb.rag.encoders.tuned import embedder_fingerprint


def store_embedder_mismatch(meta: RagStoreMeta, expected_model: str) -> str | None:
    """Return the store's built encoder identity when it differs from `expected_model`, else None.

    A store is embedded and queried by the SAME encoder (recorded in `store_meta.json`), so a
    config that names a different `embedding_model` than the store on disk would silently score
    the wrong encoder. Callers refuse the run with this signal (embedding bake-off fingerprint).

    For a locally fine-tuned encoder the id is a DIRECTORY, and a directory is not an identity:
    retraining into the same path leaves the string equal while the weights change. So when either
    side carries a recorded fingerprint (`llb.rag.encoders.tuned`), the identities are compared
    rather than the ids, and a same-path/different-training pair is refused with the fingerprint
    the store was built under.
    """
    built = str(meta.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    built_fingerprint = meta.get("embedder_fingerprint")
    if built != expected_model:
        return built
    if not isinstance(built_fingerprint, str) or not built_fingerprint:
        return None  # a store written before the field existed: the id comparison is all there is
    expected_fingerprint = embedder_fingerprint(expected_model)
    if built_fingerprint == expected_fingerprint:
        return None
    return f"{built} [{built_fingerprint}]"


def stale_store_message(
    meta: RagStoreMeta, corpus_root: Path | str, index_dir: Path | str
) -> str | None:
    """Return a rebuild message when the store fingerprint differs from the current corpus."""
    built = meta.get("corpus_fingerprint")
    if not isinstance(built, str):
        return None
    current = corpus_fingerprint(corpus_root)
    if built == current:
        return None
    return (
        f"[rag] stale store at {index_dir}: corpus manifest fingerprint changed. "
        "Refresh incrementally with `llb refresh-index` (changed documents only) or rebuild "
        "with `llb build-index --corpus-root <corpus-dir>` so removed sources and "
        "governance metadata propagate into chunks."
    )
