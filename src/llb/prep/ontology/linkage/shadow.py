"""Score the linkage model beside the shipped drop constant, without changing any drop.

The lane runs after the constant has decided: it fits the gold-item specification over the prior
bundles plus the drafted batch, looks up the pair behind every drop the constant made, and reports
where a probability cut and the constant would disagree. Nothing here removes or restores an item
-- adopting the model as the drop policy needs the reviewer-labelled operating point, which this
report is the input to.
"""

import logging
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.goldset.schema import GoldItem
from llb.linkage.model import LinkageResult
from llb.prep.ontology.extraction.dedup import QuestionEmbedder
from llb.prep.ontology.linkage.agreements import (
    adjacency,
    level_labels,
    pair_index,
    pair_key,
    pair_payload,
)
from llb.prep.ontology.linkage.constants import (
    MAX_SHADOW_RECORDS,
    MIN_SHADOW_RECORDS,
    ROLE_CANDIDATE,
    ROLE_PRIOR,
    SHADOW_MODE,
)
from llb.prep.ontology.linkage.records import build_gold_item_spec, build_records, embed_columns
from llb.prep.ontology.linkage.verdicts import ShadowDecisions, operating_points
from llb.prep.ontology.models import ItemLabels

_LOG = logging.getLogger(__name__)

_REQUIRED_PACKAGES = ("splink", "duckdb")


@dataclass(frozen=True)
class ShadowScoring:
    """What the shadow lane produced: the report block, and one payload per shipped drop."""

    report: JsonObject
    per_drop: dict[str, JsonObject] = field(default_factory=dict)


def _declined(prior_items: list[GoldItem], candidates: list[GoldItem]) -> str | None:
    """Why the lane is not running, in the words the report will carry (None = it runs)."""
    missing = [name for name in _REQUIRED_PACKAGES if find_spec(name) is None]
    if missing:
        return f"the linkage extra is not installed (missing: {', '.join(missing)})"
    if not candidates:
        return "the batch drafted no items to score"
    total = len(prior_items) + len(candidates)
    if total < MIN_SHADOW_RECORDS:
        return (
            f"{total} records are below the {MIN_SHADOW_RECORDS}-record floor a Fellegi-Sunter "
            "fit needs"
        )
    if total > MAX_SHADOW_RECORDS:
        return (
            f"{total} records exceed the {MAX_SHADOW_RECORDS}-record cap of the all-pairs "
            "blocking rule"
        )
    return None


def _record_lookup(records: list[JsonObject]) -> tuple[dict[tuple[str, str], str], dict[str, int]]:
    """`(role, item id) -> record id` and `record id -> table position`, first occurrence wins."""
    by_item: dict[tuple[str, str], str] = {}
    positions: dict[str, int] = {}
    for position, record in enumerate(records):
        identifier = str(record["unique_id"])
        positions[identifier] = position
        by_item.setdefault((str(record["role"]), str(record["item_id"])), identifier)
    return by_item, positions


def _shipped_pair(
    row: dict[str, object], by_item: dict[tuple[str, str], str]
) -> tuple[str, str] | None:
    """The record pair one drop row was decided on, or None when its partner is not in the table."""
    candidate = by_item.get((ROLE_CANDIDATE, str(row.get("id", ""))))
    prior = row.get("nearest_prior_id")
    batch = row.get("nearest_batch_id")
    partner = (
        by_item.get((ROLE_PRIOR, str(prior)))
        if prior is not None
        else by_item.get((ROLE_CANDIDATE, str(batch)))
        if batch is not None
        else None
    )
    if candidate is None or partner is None:
        return None
    return pair_key(candidate, partner)


def _summary(result: LinkageResult, decisions: ShadowDecisions, records: int) -> JsonObject:
    return {
        "n_records": records,
        "n_scored_pairs": len(result.pairs),
        "n_shipped_drops": decisions.n_shipped,
        "n_shipped_drops_scored": decisions.n_scored,
        "n_untrained_levels": len(result.untrained_levels),
        "trained_from_labels": result.trained_from_labels,
    }


def score_shadow_linkage(
    *,
    prior_items: list[GoldItem],
    candidates: list[GoldItem],
    candidate_labels: dict[str, ItemLabels],
    dropped_detail: list[dict[str, object]],
    embedder: QuestionEmbedder,
    bundle_dir: Path | None = None,
) -> ShadowScoring:
    """Fit the gold-item linkage model over prior + drafted items and score today's decisions."""
    reason = _declined(prior_items, candidates)
    if reason is not None:
        _LOG.info("[ontology] dedup shadow linkage not run: %s", reason)
        return ShadowScoring(report={"enabled": False, "reason": reason})

    from llb.linkage.engine import run_linkage

    questions, answers = embed_columns(embedder, [*prior_items, *candidates])
    if not questions or not questions[0] or not answers[0]:
        return ShadowScoring(report={"enabled": False, "reason": "the embedder returned no width"})
    records, item_of = build_records(prior_items, candidates, candidate_labels, questions, answers)
    spec = build_gold_item_spec(len(questions[0]), len(answers[0]))
    result = run_linkage(records, spec)

    by_item, positions = _record_lookup(records)
    labels = level_labels(result.trained_model)
    candidate_ids = {item.id for item in candidates}
    shipped = {
        str(row["id"]): _shipped_pair(row, by_item)
        for row in dropped_detail
        if str(row.get("id", "")) in candidate_ids
    }
    decisions = ShadowDecisions.build(
        shipped=shipped,
        candidates=candidates,
        by_item=by_item,
        positions=positions,
        pairs=pair_index(result.pairs),
        neighbours=adjacency(result.pairs),
        item_of=item_of,
        labels=labels,
    )
    report: JsonObject = {
        "enabled": True,
        "mode": SHADOW_MODE,
        "n_prior_items": len(prior_items),
        "n_candidates": len(candidates),
        **_summary(result, decisions, len(records)),
        **decisions.thresholds_payload(),
        "operating_points": operating_points(decisions),
    }
    if bundle_dir is not None:
        from llb.linkage.artifacts import write_linkage_artifacts

        report["artifacts"] = write_linkage_artifacts(
            result,
            Path(bundle_dir),
            {
                "mode": SHADOW_MODE,
                "n_prior_items": len(prior_items),
                "n_candidates": len(candidates),
                "provisional_match_weight": decisions.provisional_weight,
                "policy": "shadow only -- the shipped constant decided every drop in this run",
            },
        )
    _LOG.info(
        "[ontology] dedup shadow linkage: %d records, %d pairs, provisional match weight %.4f",
        len(records),
        len(result.pairs),
        decisions.provisional_weight,
    )
    return ShadowScoring(
        report=report,
        per_drop={
            item_id: pair_payload(pair, labels) for item_id, pair in decisions.scored_drops.items()
        },
    )
