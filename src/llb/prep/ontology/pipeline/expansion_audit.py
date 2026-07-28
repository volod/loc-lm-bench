"""Readiness audit for a widened multi-hop draft ledger."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from llb.goldset.schema import GoldItem, load_goldset
from llb.goldset.validate import validate_items
from llb.prep.ontology.constants import (
    MULTI_HOP_NEAR_DUP_ANSWER_COSINE_THRESHOLD,
    NEAR_DUP_COSINE_THRESHOLD,
)
from llb.prep.ontology.inventory import inventory_corpus
from llb.prep.ontology.language import is_ukrainian_dominant
from llb.prep.ontology.pipeline.expansion import labeled_multi_hop_ids


def minimum_combined_items(carried_items: int, minimum_headroom_fraction: float) -> int:
    """Translate a corpus-relative headroom requirement into a combined-ledger size."""
    if carried_items < 0:
        raise ValueError("carried_items must be non-negative")
    if not 0.0 <= minimum_headroom_fraction <= 1.0:
        raise ValueError("minimum_headroom_fraction must be between zero and one")
    return math.ceil(carried_items * (1.0 + minimum_headroom_fraction))


@dataclass
class _ExpansionAudit:
    """Accumulate independent readiness checks, then build the public report."""

    root: Path
    items: list[GoldItem]
    labeled_ids: set[str]
    carry: object
    dedup: object
    path_strata: object
    path_stratified: bool
    validation_errors: list[str]
    texts: dict[str, str]
    errors: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, root: Path) -> "_ExpansionAudit":
        items = load_goldset(root / "goldset.jsonl")
        provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        settings = provenance.get("settings")
        return cls(
            root=root,
            items=items,
            labeled_ids=labeled_multi_hop_ids(root),
            carry=provenance.get("multi_hop_carry_forward", {}),
            dedup=provenance.get("dedup", {}),
            path_strata=provenance.get("multi_hop_path_strata", {}),
            path_stratified=bool(
                settings.get("multi_hop_path_stratified") if isinstance(settings, dict) else False
            ),
            validation_errors=list(validate_items(items, root / "corpus")["errors"]),
            texts={doc.doc_id: doc.text for doc in inventory_corpus(root / "corpus")},
        )

    def check_item_contract(self) -> None:
        self.errors.extend(self.validation_errors)
        if {item.id for item in self.items} != self.labeled_ids:
            self.errors.append("goldset and labeled multi-hop sidecar ids differ")
        if any(self._distinct_span_count(item) < 2 for item in self.items):
            self.errors.append(
                "one or more multi-hop rows have fewer than two distinct exact spans"
            )

    def check_language_and_duplicates(self) -> None:
        if not self.all_ukrainian:
            self.errors.append("one or more multi-hop rows fail the Ukrainian output gate")
        normalized = [" ".join(item.question.casefold().split()) for item in self.items]
        if len(normalized) != len(set(normalized)):
            self.errors.append("the combined ledger has exact duplicate questions")

    def check_dedup_accounting(self) -> None:
        if not isinstance(self.dedup, dict) or int(self.dedup.get("prior_questions", 0)) == 0:
            self.errors.append("the expansion has no recorded prior-question dedup")
            return
        if not isinstance(self.carry, dict):
            return
        checked = int(self.dedup.get("checked", 0))
        dropped = int(self.dedup.get("dropped", 0))
        new_items = int(self.carry.get("new_items", 0))
        if checked != dropped + new_items:
            self.errors.append("dedup accounting does not cover every newly drafted item")

    def check_headroom(self, minimum_headroom_fraction: float) -> None:
        carry = self.carry if isinstance(self.carry, dict) else {}
        carried = int(carry.get("carried_items", 0))
        if carried <= 0:
            self.errors.append("the expansion has no carried review baseline")
            return
        required = minimum_combined_items(carried, minimum_headroom_fraction)
        if len(self.items) < required:
            self.errors.append(
                f"drafted multi-hop size {len(self.items)} is below the relative "
                f"headroom requirement {required}"
            )

    def check_path_strata(self) -> None:
        if not self.path_stratified:
            return
        if not isinstance(self.path_strata, dict):
            self.errors.append("stratified expansion has no path-strata report")
            return
        if not self.path_strata.get("all_requested_covered_or_exhausted"):
            self.errors.append(
                "one or more requested path strata are neither covered nor exhausted"
            )

    @property
    def all_span_exact(self) -> bool:
        return all(
            self.texts.get(span.doc_id, "")[span.char_start : span.char_end] == span.text
            for item in self.items
            for span in item.source_spans
        )

    @property
    def all_ukrainian(self) -> bool:
        return all(
            is_ukrainian_dominant(text)
            for item in self.items
            for text in (item.question, item.reference_answer)
        )

    @staticmethod
    def _distinct_span_count(item: GoldItem) -> int:
        return len({(span.doc_id, span.char_start, span.char_end) for span in item.source_spans})

    def report(self, *, minimum_headroom_fraction: float) -> dict[str, object]:
        drafted = len(self.items)
        has_carry = isinstance(self.carry, dict)
        carry = self.carry if isinstance(self.carry, dict) else {}
        dedup = self.dedup if isinstance(self.dedup, dict) else {}
        path_strata = self.path_strata if isinstance(self.path_strata, dict) else {}
        carried = int(carry.get("carried_items", 0))
        added = max(0, drafted - carried)
        return {
            "kind": "multihop-draft-expansion-audit",
            "bundle": str(self.root),
            "drafted_multi_hop_items": drafted,
            "carried_items": carried,
            "added_items": added,
            "review_headroom_fraction": (added / carried) if carried else 0.0,
            "minimum_headroom_fraction": minimum_headroom_fraction,
            "minimum_combined_items": (
                minimum_combined_items(carried, minimum_headroom_fraction) if carried else None
            ),
            "new_items": int(carry.get("new_items", 0)) if has_carry else drafted,
            "dedup_prior_questions": int(dedup.get("prior_questions", 0)),
            "dedup_dropped": int(dedup.get("dropped", 0)),
            "dedup_question_threshold": NEAR_DUP_COSINE_THRESHOLD,
            "dedup_answer_threshold": MULTI_HOP_NEAR_DUP_ANSWER_COSINE_THRESHOLD,
            "all_rows_span_exact": self.all_span_exact and not self.validation_errors,
            "all_rows_ukrainian": self.all_ukrainian,
            "path_stratified": self.path_stratified,
            "path_strata_ready": (
                bool(path_strata.get("all_requested_covered_or_exhausted"))
                if self.path_stratified
                else None
            ),
            "ready_for_human_review": not self.errors,
            "errors": self.errors,
        }


def audit_multihop_expansion(
    bundle: Path | str, *, minimum_headroom_fraction: float
) -> dict[str, object]:
    """Audit the widened review ledger and return a machine-readable readiness report."""
    minimum_combined_items(0, minimum_headroom_fraction)
    audit = _ExpansionAudit.load(Path(bundle))
    audit.check_item_contract()
    audit.check_language_and_duplicates()
    audit.check_dedup_accounting()
    audit.check_headroom(minimum_headroom_fraction)
    audit.check_path_strata()
    return audit.report(minimum_headroom_fraction=minimum_headroom_fraction)
