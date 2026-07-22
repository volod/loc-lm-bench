"""The drafting stages between extraction and bundle emit: seed selection + QA drafting (stage 4-5),
the optional graph-walk stages (multi-hop items, ordered question chains), cross-bundle dedup, and
the rejection-feedback prompt adjustment.

Each stage is a pure function of its inputs + `DraftSettings`, so `draft_goldset` reads as a linear
composition of them.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from llb.goldset.chains import ChainItem
from llb.goldset.schema import GoldItem
from llb.prep.frontier_telemetry import LLMComplete
from llb.prep.ontology.coverage import build_seeds, select_seeds
from llb.prep.ontology.coverage_report import coverage_report
from llb.prep.ontology.dedup import QuestionEmbedder
from llb.prep.ontology.draft import draft_items
from llb.prep.ontology.induce import ontology_constraints
from llb.prep.ontology.models import (
    DocExtraction,
    DocRecord,
    ItemLabels,
    OntologyCandidate,
)
from llb.prep.ontology.pipeline.settings import DraftSettings
from llb.prep.ontology.refine import refine_drafts_labeled

if TYPE_CHECKING:
    from llb.graph.model import KnowledgeGraph

_LOG = logging.getLogger(__name__)


def _load_path_graph(
    graph_dir: Path | str | None,
    extractions: list[DocExtraction],
    docs: list[DocRecord],
    ontology: OntologyCandidate,
) -> "KnowledgeGraph":
    """The knowledge graph the multi-hop walker reads: a persisted store, else built in-run."""
    if graph_dir is not None:
        from llb.graph.store import GraphStore

        return GraphStore.load(graph_dir).graph
    from llb.graph.build import build_graph

    return build_graph(extractions, docs, ontology)


def _multi_hop_stage(
    complete: LLMComplete,
    docs: list[DocRecord],
    extractions: list[DocExtraction],
    ontology: OntologyCandidate,
    *,
    graph_dir: Path | str | None,
    max_paths: int,
    seed: int,
    bridge_fill: bool = False,
    graph: "KnowledgeGraph | None" = None,
) -> tuple[list[GoldItem], dict[str, ItemLabels]]:
    """Walk 2-hop graph paths and draft multi-span multi-hop chain items (yield-max).

    Strict directed `A -r1-> B -r2-> C` chains are the default, but extracted graphs are usually
    too sparse in object-to-subject links to supply a measurable slice (a 625-node, 213-edge
    Ukrainian PDF graph yields ONE). `bridge_fill` keeps directed paths first and then fills with
    the same shared-bridge fact pairs the chain lane already uses: two distinct facts incident on
    one entity, cited from two distinct spans, which is exactly the >= 2-span retrieval problem a
    multi-hop slice has to measure.

    `graph` is the injection seam: pass a built `KnowledgeGraph` to skip the store load entirely
    (tests drive the stage with no DuckDB, no store on disk, and no extraction).
    """
    from llb.prep.ontology.graph_paths import walk_chain_paths, walk_two_hop_paths
    from llb.prep.ontology.multi_hop import build_multi_hop_items, draft_multi_hop

    if graph is None:
        graph = _load_path_graph(graph_dir, extractions, docs, ontology)
    walk = walk_chain_paths if bridge_fill else walk_two_hop_paths
    seeds = walk(graph, max_paths=max_paths, seed=seed)
    raw = draft_multi_hop(complete, docs, seeds)
    return build_multi_hop_items(docs, seeds, raw)


def _chain_stage(
    docs: list[DocRecord],
    extractions: list[DocExtraction],
    ontology: OntologyCandidate,
    *,
    graph_dir: Path | str | None,
    max_paths: int,
    seed: int,
) -> list[ChainItem]:
    """Walk 2-hop graph paths and emit ordered chain-of-questions items."""
    from llb.prep.ontology.chains import build_chain_items
    from llb.prep.ontology.graph_paths import walk_chain_paths

    graph = _load_path_graph(graph_dir, extractions, docs, ontology)
    seeds = walk_chain_paths(graph, max_paths=max_paths, seed=seed)
    return build_chain_items(docs, seeds)


def _dedup_stage(
    items: list[GoldItem],
    labels: dict[str, ItemLabels],
    *,
    dedup_against: list[Path | str],
    embedder: QuestionEmbedder | None,
) -> tuple[list[GoldItem], dict[str, ItemLabels], dict[str, object]]:
    """Drop near-duplicates of prior-bundle questions (pinned E5); prune their labels (yield-max)."""
    from llb.prep.ontology.dedup import (
        E5QuestionEmbedder,
        NearDuplicateFilter,
        load_prior_questions,
    )

    prior = load_prior_questions(dedup_against)
    resolved = embedder if embedder is not None else E5QuestionEmbedder()
    kept, report = NearDuplicateFilter(prior, resolved).filter(items)
    kept_ids = {item.id for item in kept}
    kept_labels = {item_id: label for item_id, label in labels.items() if item_id in kept_ids}
    report["prior_bundles"] = [str(path) for path in dedup_against]
    return kept, kept_labels, report


def _feedback_adjusted_hint(
    draft_hint: str, rejection_feedback: Path | str
) -> tuple[str, dict[str, object]]:
    """Tighten the draft prompt with verify-gate rejection feedback; return the applied block."""
    from llb.prep.ontology.feedback import (
        applied_feedback_block,
        feedback_hint_text,
        feedback_hints,
        load_rejection_feedback,
    )

    hints = feedback_hints(load_rejection_feedback(rejection_feedback))
    applied = applied_feedback_block(rejection_feedback, hints)
    hint_text = feedback_hint_text(hints)
    if hint_text:
        draft_hint = f"{draft_hint}\n{hint_text}" if draft_hint else hint_text
        _LOG.info(
            "[ontology] applying rejection feedback (%d hint(s)) from %s",
            len(hints),
            rejection_feedback,
        )
    return draft_hint, applied


def _draft_stage(
    complete: LLMComplete,
    docs: list[DocRecord],
    extractions: list[DocExtraction],
    ontology: OntologyCandidate,
    settings: DraftSettings,
) -> tuple[list[GoldItem], dict[str, ItemLabels], dict[str, object], dict[str, object] | None]:
    """Stages 4-5: seed selection + QA drafting. Returns items, labels, coverage, feedback."""
    pool = build_seeds(docs, extractions)
    seeds = select_seeds(
        pool,
        max_items=settings.max_items,
        seed=settings.seed,
        coverage_target=settings.coverage_target,
    )
    cov_report = coverage_report(
        pool, seeds, coverage_target=settings.coverage_target, max_items=settings.max_items
    )
    draft_hint = ontology_constraints(ontology)
    applied_feedback: dict[str, object] | None = None
    if settings.rejection_feedback is not None:
        draft_hint, applied_feedback = _feedback_adjusted_hint(
            draft_hint, settings.rejection_feedback
        )
    raw_drafts = draft_items(complete, docs, seeds, draft_hint)
    items, item_labels = refine_drafts_labeled(docs, raw_drafts)
    return (
        items,
        item_labels,
        {"seeds": seeds, "coverage": cov_report, "draft_parsed": len(raw_drafts)},
        applied_feedback,
    )


def _graph_stages(
    complete: LLMComplete,
    docs: list[DocRecord],
    extractions: list[DocExtraction],
    ontology: OntologyCandidate,
    settings: DraftSettings,
    items: list[GoldItem],
    item_labels: dict[str, ItemLabels],
) -> tuple[list[GoldItem], dict[str, ItemLabels], list[ChainItem]]:
    """Optional graph-walk stages: multi-hop items and ordered question chains."""
    if settings.multi_hop:
        mh_items, mh_labels = _multi_hop_stage(
            complete,
            docs,
            extractions,
            ontology,
            graph_dir=settings.graph_dir,
            max_paths=settings.multi_hop_max_paths,
            seed=settings.seed,
            bridge_fill=settings.multi_hop_bridge_fill,
        )
        items = items + mh_items
        item_labels = {**item_labels, **mh_labels}
    chain_items: list[ChainItem] = []
    if settings.chains:
        chain_items = _chain_stage(
            docs,
            extractions,
            ontology,
            graph_dir=settings.graph_dir,
            max_paths=settings.multi_hop_max_paths,
            seed=settings.seed,
        )
    return items, item_labels, chain_items
