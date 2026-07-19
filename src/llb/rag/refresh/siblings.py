"""Refresh the per-strategy comparison stores nested under the main index directory.

`compare-retrieval` persists each candidate store under ``<index-dir>/<strategy>/`` (for
example ``$DATA_DIR/llb/rag/sentence/`` or ``.../hybrid/``). Those siblings record the same
per-doc fingerprints as the main store, so after corpus edits each one is refreshed through the
ordinary `refresh_vector_store` path -- otherwise a later `compare-retrieval` rerun would
silently serve stale sibling stores. The ``generations/`` child is the main store's own refresh
history, never a sibling.
"""

from pathlib import Path
from typing import Any

from llb.core.store_generations import GENERATIONS_DIRNAME, resolve_store_dir
from llb.rag.lexical import Lemmatizer
from llb.rag.refresh.store_refresh import VectorRefreshResult, refresh_vector_store
from llb.rag.store_build import META_FILE


def sibling_store_dirs(index_dir: Path | str) -> list[Path]:
    """Direct children of `index_dir` that resolve to a built store, sorted by name."""
    base = Path(index_dir)
    if not base.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name == GENERATIONS_DIRNAME:
            continue
        live = resolve_store_dir(child, META_FILE)
        if (live / META_FILE).is_file():
            out.append(child)
    return out


def refresh_sibling_stores(
    index_dir: Path | str,
    corpus_root: Path | str,
    *,
    embedder: Any = None,
    lemmatizer: Lemmatizer | None = None,
    timestamp: str | None = None,
) -> list[tuple[str, VectorRefreshResult]]:
    """Refresh every sibling comparison store against the corpus; per-store no-ops are fine.

    Each sibling diffs its own recorded fingerprints, so an already-current store returns an
    unrefreshed result. With `embedder=None` every store loads the embedding model its meta
    records (tests inject one fake for all).
    """
    results: list[tuple[str, VectorRefreshResult]] = []
    for store_dir in sibling_store_dirs(index_dir):
        result = refresh_vector_store(
            store_dir,
            corpus_root,
            embedder=embedder,
            lemmatizer=lemmatizer,
            timestamp=timestamp,
        )
        results.append((store_dir.name, result))
    return results
