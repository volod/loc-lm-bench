"""Pipeline policy for suppressing prior-bundle question duplicates.

The shipped policy is the cosine constant and nothing else. The optional shadow lane fits the
gold-item linkage model over the same records and reports what a probability cut WOULD have
decided, so a reviewer can move the operating point with its consequence visible; it never
changes which items this function drops.
"""

from pathlib import Path
from typing import cast

from llb.goldset.schema import GoldItem
from llb.prep.ontology.constants import MULTI_HOP_NEAR_DUP_ANSWER_COSINE_THRESHOLD
from llb.prep.ontology.extraction.dedup import QuestionEmbedder
from llb.prep.ontology.models import ItemLabels


def _normalized_question(text: str) -> str:
    return " ".join(text.casefold().split())


def _drop_intra_batch_exact_questions(
    items: list[GoldItem],
) -> tuple[list[GoldItem], list[dict[str, object]]]:
    """Keep the first exact normalized question and explain every later rejection."""
    first_by_question: dict[str, GoldItem] = {}
    kept: list[GoldItem] = []
    dropped: list[dict[str, object]] = []
    for item in items:
        normalized = _normalized_question(item.question)
        first = first_by_question.get(normalized)
        if first is None:
            first_by_question[normalized] = item
            kept.append(item)
            continue
        dropped.append(
            {
                "id": item.id,
                "reason": "intra-batch-exact-question",
                "nearest_batch_id": first.id,
                "nearest_batch_question": first.question,
                "candidate_question": item.question,
            }
        )
    return kept, dropped


def _partition_items(
    items: list[GoldItem], labels: dict[str, ItemLabels]
) -> tuple[list[GoldItem], list[GoldItem]]:
    multi_hop_ids = {
        item.id
        for item in items
        if (label := labels.get(item.id)) and label.question_type == "multi-hop"
    }
    multi_hop_items = [item for item in items if item.id in multi_hop_ids]
    flat_items = [item for item in items if item.id not in multi_hop_ids]
    return flat_items, multi_hop_items


def _shadow_lane(
    items: list[GoldItem],
    labels: dict[str, ItemLabels],
    prior_items: list[GoldItem],
    embedder: QuestionEmbedder,
    dropped_detail: list[dict[str, object]],
    bundle_dir: Path | None,
) -> dict[str, object]:
    """Score the linkage model beside the decisions just taken and enrich each drop row."""
    from llb.prep.ontology.linkage.shadow import score_shadow_linkage

    scoring = score_shadow_linkage(
        prior_items=prior_items,
        candidates=items,
        candidate_labels=labels,
        dropped_detail=dropped_detail,
        embedder=embedder,
        bundle_dir=bundle_dir,
    )
    for row in dropped_detail:
        payload = scoring.per_drop.get(str(row.get("id", "")))
        if payload is not None:
            row.update(payload)
    return scoring.report


def deduplicate_drafts(
    items: list[GoldItem],
    labels: dict[str, ItemLabels],
    *,
    dedup_against: list[Path | str],
    embedder: QuestionEmbedder | None,
    linkage_shadow: bool = False,
    bundle_dir: Path | None = None,
) -> tuple[list[GoldItem], dict[str, ItemLabels], dict[str, object]]:
    """Apply flat and multi-hop duplicate policies and prune labels for rejected rows."""
    from llb.prep.ontology.extraction.dedup import (
        E5QuestionEmbedder,
        NearDuplicateFilter,
        load_prior_items,
    )

    prior_items = load_prior_items(dedup_against)
    prior_questions = [item.question for item in prior_items]
    prior_answers = [item.reference_answer for item in prior_items]
    resolved = embedder if embedder is not None else E5QuestionEmbedder()
    unique_items, intra_batch_dropped = _drop_intra_batch_exact_questions(items)
    flat_items, multi_hop_items = _partition_items(unique_items, labels)
    prior_ids = [item.id for item in prior_items]
    kept_flat, flat_report = NearDuplicateFilter(
        prior_questions, resolved, prior_ids=prior_ids
    ).filter(flat_items)
    kept_multi_hop, multi_hop_report = NearDuplicateFilter(
        prior_questions,
        resolved,
        prior_answers=prior_answers,
        answer_threshold=MULTI_HOP_NEAR_DUP_ANSWER_COSINE_THRESHOLD,
        prior_ids=prior_ids,
    ).filter(multi_hop_items)
    kept_ids = {item.id for item in [*kept_flat, *kept_multi_hop]}
    kept = [item for item in items if item.id in kept_ids]
    dropped_detail = [
        *intra_batch_dropped,
        *cast(list[dict[str, object]], flat_report.get("dropped_detail", [])),
        *cast(list[dict[str, object]], multi_hop_report.get("dropped_detail", [])),
    ]
    report: dict[str, object] = {
        "enabled": True,
        "threshold": flat_report["threshold"],
        "prior_questions": len(prior_questions),
        "checked": len(items),
        "dropped": len(dropped_detail),
        "dropped_ids": [row["id"] for row in dropped_detail],
        "dropped_detail": dropped_detail,
        "intra_batch_exact_dropped": len(intra_batch_dropped),
        "question_only_items": len(flat_items),
        "question_answer_items": len(multi_hop_items),
        "question_answer_threshold": multi_hop_report.get("answer_threshold"),
        "prior_bundles": [str(path) for path in dedup_against],
    }
    if linkage_shadow:
        report["linkage_shadow"] = _shadow_lane(
            items, labels, prior_items, resolved, dropped_detail, bundle_dir
        )
    kept_labels = {item_id: label for item_id, label in labels.items() if item_id in kept_ids}
    return kept, kept_labels, report
