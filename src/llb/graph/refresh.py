"""Incremental GraphRAG store refresh: re-extract changed documents only.

`build_graph` is pure and deterministic from (extractions, docs, ontology); the expensive part
is the model-backed per-document extraction. Refresh keeps the persisted extractions of
unchanged documents, takes updated extraction rows for added/modified documents from the
operator-provided update file (produced by re-extracting just those documents), drops deleted
documents entirely, and rebuilds the graph -- so mention/edge deletion propagates exactly and
the result equals a from-scratch build over the merged extraction set.

To make that possible the graph inputs (`extraction.jsonl` + `ontology.json`) are persisted
beside every graph store (`save_graph_inputs`, called by `build-graph` and by each refresh), and
the graph meta records per-doc content hashes (`doc_fingerprints`). Refreshed stores publish as
immutable ``generations/<utc-timestamp>/`` children of the graph dir, like the vector store.
Tagged-diagnostic community summaries are not carried over (they would describe the old graph);
re-run `build-graph --summarize` if needed.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from llb.core.store_generations import (
    generation_timestamp,
    new_generation_paths,
    publish_generation,
    resolve_store_dir,
)
from llb.graph.constants import DEFAULT_KHOP_DEPTH, META_FILE
from llb.graph.ingest import load_extractions, load_ontology
from llb.graph.store import GraphStore
from llb.prep.ontology.constants import EXTRACTION_FILENAME, ONTOLOGY_FILENAME
from llb.prep.ontology.inventory import inventory_corpus
from llb.prep.ontology.models import DocExtraction, OntologyCandidate
from llb.rag.refresh.diff import ManifestDiff, diff_fingerprints

_LOG = logging.getLogger(__name__)


@dataclass
class GraphRefreshResult:
    """Outcome of one graph-store refresh (`refreshed=False` == corpus unchanged, no-op)."""

    diff: ManifestDiff
    refreshed: bool
    source_dir: Path
    generation_dir: Path | None = None
    store: GraphStore | None = None


def save_graph_inputs(
    graph_dir: Path | str,
    extractions: list[DocExtraction],
    ontology: OntologyCandidate | None,
) -> None:
    """Persist the graph's build inputs beside the store so a later refresh can chain."""
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    with (graph_dir / EXTRACTION_FILENAME).open("w", encoding="utf-8") as fh:
        for extraction in extractions:
            fh.write(json.dumps(extraction.model_dump(), ensure_ascii=False) + "\n")
    if ontology is not None:
        (graph_dir / ONTOLOGY_FILENAME).write_text(
            json.dumps(ontology.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def refresh_graph_store(
    graph_dir: Path | str,
    corpus_root: Path | str,
    *,
    extraction_update: list[DocExtraction] | None = None,
    timestamp: str | None = None,
) -> GraphRefreshResult:
    """Diff the corpus against the live graph store and publish a refreshed generation.

    `extraction_update` supplies the extraction rows for added/modified documents; a refresh
    with changed documents and no matching rows refuses with the list of documents that need
    re-extraction (deletion-only refreshes need no update rows).
    """
    base_dir = Path(graph_dir)
    live_dir = resolve_store_dir(base_dir, META_FILE)
    meta_path = live_dir / META_FILE
    if not meta_path.is_file():
        raise SystemExit(
            f"[refresh] no graph store at {base_dir}; build one first with `llb build-graph`"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    docs = inventory_corpus(corpus_root)
    current = {doc.doc_id: doc.sha256 for doc in docs}
    recorded_raw = meta.get("doc_fingerprints")
    if not isinstance(recorded_raw, dict):
        _LOG.warning(
            "[refresh] graph store at %s records no doc_fingerprints (built before refresh "
            "support); treating every document as changed",
            live_dir,
        )
        recorded_raw = {}
    recorded = {str(doc_id): str(sha) for doc_id, sha in recorded_raw.items()}
    diff = diff_fingerprints(recorded, current)
    if recorded and not diff.has_changes:
        return GraphRefreshResult(diff=diff, refreshed=False, source_dir=live_dir)

    extraction_path = live_dir / EXTRACTION_FILENAME
    if not extraction_path.is_file():
        raise SystemExit(
            f"[refresh] the graph store at {live_dir} has no persisted {EXTRACTION_FILENAME}; "
            "rebuild it once with `llb build-graph` (which now saves its inputs) to enable "
            "incremental refresh"
        )
    old_extractions = load_extractions(extraction_path)
    updates = {extraction.doc_id: extraction for extraction in (extraction_update or [])}
    missing = sorted(doc_id for doc_id in diff.changed if doc_id not in updates)
    if missing:
        raise SystemExit(
            "[refresh] extraction rows missing for changed documents: "
            + ", ".join(missing)
            + "; re-extract just those documents and pass the file via --graph-extraction, "
            "or skip the graph store with --skip-graph"
        )
    unchanged = set(diff.unchanged)
    kept = {e.doc_id: e for e in old_extractions if e.doc_id in unchanged}
    merged: list[DocExtraction] = []
    for doc in docs:  # inventory order == the deterministic from-scratch build order
        if doc.doc_id in diff.changed:
            merged.append(updates[doc.doc_id])
        elif doc.doc_id in kept:
            merged.append(kept[doc.doc_id])
    ontology = load_ontology(live_dir / ONTOLOGY_FILENAME)
    store = GraphStore.build(
        merged, docs, ontology, khop_depth=int(meta.get("khop_depth", DEFAULT_KHOP_DEPTH))
    )
    staging_dir, final_dir = new_generation_paths(base_dir, timestamp or generation_timestamp())
    store.save(staging_dir)
    save_graph_inputs(staging_dir, merged, ontology)
    generation_dir = publish_generation(staging_dir, final_dir)
    _LOG.info("[refresh] graph %s -> %s: %s", live_dir, generation_dir, diff.summary())
    return GraphRefreshResult(
        diff=diff,
        refreshed=True,
        source_dir=live_dir,
        generation_dir=generation_dir,
        store=store,
    )
