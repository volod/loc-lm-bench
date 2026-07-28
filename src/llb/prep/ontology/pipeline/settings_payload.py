"""Serialization helpers for ontology-draft settings."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llb.prep.ontology.pipeline.settings import DraftSettings


def _opt_str(value: Path | str | None) -> str | None:
    return str(value) if value is not None else None


def _multi_hop_settings(settings: "DraftSettings") -> dict[str, object]:
    return {
        "multi_hop": settings.multi_hop,
        "multi_hop_only": settings.multi_hop_only,
        "chains": settings.chains,
        "multi_hop_max_paths": settings.multi_hop_max_paths,
        "multi_hop_bridge_fill": settings.multi_hop_bridge_fill,
        "multi_hop_path_stratified": settings.multi_hop_path_stratified,
        "multi_hop_relation_pair_target": settings.multi_hop_relation_pair_target,
        "multi_hop_document_mode_target": settings.multi_hop_document_mode_target,
        "multi_hop_source_document_target": settings.multi_hop_source_document_target,
        "dedup_against": (
            [str(path) for path in settings.dedup_against] if settings.dedup_against else None
        ),
        "carry_forward_multi_hop": settings.carry_forward_multi_hop,
        "graph_dir": _opt_str(settings.graph_dir),
    }


def pinned_payload(settings: "DraftSettings") -> dict[str, object]:
    """Determinism-critical settings recorded in the journal meta sidecar."""
    return {
        "corpus_root": settings.corpus_root,
        "seed": settings.seed,
        "max_items": settings.max_items,
        "doc_limit": settings.doc_limit,
        "extract_max_chars": settings.resolved_extract_max_chars,
        "extract_chunk_overlap": settings.resolved_extract_overlap,
        "extract_concurrency": settings.resolved_extract_concurrency,
        "reuse_extraction_bundle": _opt_str(settings.reuse_extraction_bundle),
        "retrieval_index_dir": _opt_str(settings.retrieval_index_dir),
        "retrieval_k": settings.retrieval_k,
        "drop_nonretrievable_needles": settings.drop_nonretrievable_needles,
        "coverage_target": settings.coverage_target,
        **_multi_hop_settings(settings),
        "rejection_feedback": _opt_str(settings.rejection_feedback),
    }


def provenance_settings(settings: "DraftSettings", *, resumed: bool) -> dict[str, object]:
    """The settings block recorded in a completed bundle's provenance."""
    return {
        "max_items": settings.max_items,
        "seed": settings.seed,
        "doc_limit": settings.doc_limit,
        "extract_max_chars": settings.resolved_extract_max_chars,
        "extract_chunk_overlap": settings.resolved_extract_overlap,
        "extract_concurrency": settings.resolved_extract_concurrency,
        "reuse_extraction_bundle": _opt_str(settings.reuse_extraction_bundle),
        "coverage_target": settings.coverage_target,
        **_multi_hop_settings(settings),
        "rejection_feedback": _opt_str(settings.rejection_feedback),
        "needle_retrieval_index_dir": _opt_str(settings.retrieval_index_dir),
        "needle_retrieval_k": settings.retrieval_k,
        "drop_nonretrievable_needles": settings.drop_nonretrievable_needles,
        "resumed": resumed,
    }
