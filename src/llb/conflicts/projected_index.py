"""Persistence and reuse for the PCA-projected semantic conflict index."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.conflicts.constants import (
    PROJECTION_FILE,
    TREE_BOUND_EPSILON,
    TREE_DIR,
    TREE_FILE,
    TREE_META_FILE,
)
from llb.conflicts.projection import PCAProjection, fit_pca_projection
from llb.conflicts.store_access import StoreView
from llb.conflicts.tree import SemanticPrefixTree
from llb.conflicts.tree_refresh import tree_meta
from llb.conflicts.vectorops import VectorSet
from llb.core.contracts.common import JsonObject


@dataclass(frozen=True)
class ProjectedIndex:
    """The projected vectors/tree and their report metadata."""

    vectors: VectorSet
    tree: SemanticPrefixTree
    meta: JsonObject


def exact_projected_pairs(
    tree: SemanticPrefixTree,
    vectors: VectorSet,
    distance_threshold: float,
) -> tuple[list[tuple[int, int]], str]:
    """Exact radius pairs from SciPy's C kd-tree, with the persisted tree as fallback."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover - scipy is a declared rag dependency
        return (
            tree.candidate_pairs_within(distance_threshold, vectors),
            "semantic-prefix-tree",
        )
    rows = cKDTree(vectors.numpy_matrix()).query_pairs(
        distance_threshold + TREE_BOUND_EPSILON,
        eps=0.0,
        output_type="ndarray",
    )
    return sorted((int(left), int(right)) for left, right in rows.tolist()), "scipy-ckdtree"


def prepare_projected_index(
    store: StoreView,
    source_vectors: VectorSet,
    *,
    dims: int,
    leaf_size: int,
    centered: bool,
) -> ProjectedIndex:
    """Load a matching projection/tree or fit and persist a replacement."""
    resolved_dims = min(dims, source_vectors.dim)
    source_fingerprint = _source_fingerprint(store, centered=centered)
    directory = store.index_dir / TREE_DIR
    projection_path = directory / PROJECTION_FILE
    tree_path = directory / TREE_FILE
    meta_path = directory / TREE_META_FILE

    projection = _load_compatible_projection(
        projection_path,
        embedding_model=store.embedding_model,
        source_dim=source_vectors.dim,
        dims=resolved_dims,
        centered=centered,
    )
    fitted = projection is None
    if projection is None:
        projection = fit_pca_projection(
            source_vectors,
            resolved_dims,
            embedding_model=store.embedding_model,
            centered=centered,
            source_fingerprint=source_fingerprint,
        )
    projected = projection.transform(source_vectors)

    previous_meta = _load_json(meta_path)
    reusable = (
        not fitted
        and previous_meta.get("source_fingerprint") == source_fingerprint
        and previous_meta.get("projection_fingerprint") == projection.fingerprint
        and previous_meta.get("leaf_size") == leaf_size
        and tree_path.is_file()
    )
    tree = _load_tree(tree_path) if reusable else None
    if tree is None:
        reusable = False
        tree = SemanticPrefixTree.build(projected, leaf_size=leaf_size)
    action = "reused" if reusable else "built"
    meta: JsonObject = {
        **tree_meta(
            tree,
            embedding_model=store.embedding_model,
            dim=store.dim,
            corpus_fingerprint=str(store.meta.get("corpus_fingerprint", "")),
            doc_fingerprints=store.doc_fingerprints,
            cos_threshold=0.0,
        ),
        "source_fingerprint": source_fingerprint,
        "projection_fingerprint": projection.fingerprint,
        "project_dims": resolved_dims,
        "source_dim": source_vectors.dim,
        "centered": centered,
        "index_action": action,
    }
    if not reusable:
        directory.mkdir(parents=True, exist_ok=True)
        projection.save(projection_path)
        tree.save(tree_path)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return ProjectedIndex(vectors=projected, tree=tree, meta=meta)


def _load_compatible_projection(
    path: Path,
    *,
    embedding_model: str,
    source_dim: int,
    dims: int,
    centered: bool,
) -> PCAProjection | None:
    if not path.is_file():
        return None
    try:
        projection = PCAProjection.load(path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return (
        projection
        if projection.compatible(
            embedding_model=embedding_model,
            source_dim=source_dim,
            dims=dims,
            centered=centered,
        )
        else None
    )


def _source_fingerprint(store: StoreView, *, centered: bool) -> str:
    payload: dict[str, Any] = {
        "embedding_model": store.embedding_model,
        "dim": store.dim,
        "centered": centered,
        "corpus_fingerprint": store.meta.get("corpus_fingerprint", ""),
        "doc_fingerprints": store.doc_fingerprints,
        "chunks": [
            [
                chunk.get("chunk_id", ""),
                chunk["doc_id"],
                chunk["char_start"],
                chunk["char_end"],
            ]
            for chunk in store.chunks
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> JsonObject:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_tree(path: Path) -> SemanticPrefixTree | None:
    try:
        return SemanticPrefixTree.load(path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, SystemExit):
        return None
