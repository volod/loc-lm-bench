"""Text-analysis scoring schema (M5.0) -- objective recovery of planted labels.

This module is the executable form of the proposal in
`docs/design/text-analysis-schema.md` (the artifact a human signs off via MH.2). It defines:

  * the text-analysis SUB-TASKS (the unit of credit per sub-task) -- spec Appendix D;
  * the PLANTED-LABEL taxonomy `prepare-synthetic-corpus` must emit (`PlantedLabel`);
  * the OBJECTIVE vs JUDGED split (`OBJECTIVE_KINDS` / `JUDGED_KINDS`);
  * the MATCHING engine: label-ID matching by exact/normalized surface form, then pinned-
    embedder COSINE as the secondary signal, with explicit thresholds + partial credit.

The matching basis is the MH.2-decided engine: planted-label-ID matching + embedder cosine,
NOT lemmatization and NOT LLM-entailment. The engine is PURE -- the cosine similarity is
INJECTED as a `similarity(a, b) -> float` callable, so scoring is unit-testable without the
embedder; `embedder_similarity()` supplies the production default over the pinned embedder.

Free-form sub-tasks (narrative / insight / long-doc) carry an objective floor here but their
headline quality is the GATED judge (`llb.scoring.judge`), entering only when trusted.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from llb.contracts import PlantedLabelRecord, SubtaskScore

Similarity = Callable[[str, str], float]

# --- sub-task taxonomy (units of credit; spec Appendix D "Text Analysis") -----------------

KEY_FACT = "key_fact"  # a planted atomic fact the answer must recover
ENTITY = "entity"  # a named entity present in the doc
TOPIC = "topic"  # a planted topic/theme of the doc
TREND = "trend"  # a planted directional trend (attrs: subject, direction)
RISK = "risk"  # a planted risk/problem
DECISION = "decision"  # a planted decision/action item
CONTRADICTION = "contradiction"  # a planted internal contradiction (attrs: span ids)
NARRATIVE = "narrative"  # the doc's overarching narrative (free-form quality -> judged)
INSIGHT = "insight"  # a non-stated inference (free-form quality -> judged)
LONG_DOC = "long_doc"  # long-doc comprehension answer (map-reduce; correctness/judge)

# Recovery of these is scored OBJECTIVELY by planted-label matching (set precision/recall/F1).
OBJECTIVE_KINDS = frozenset({KEY_FACT, ENTITY, TOPIC, TREND, RISK, DECISION, CONTRADICTION})
# These are scored by the GATED judge for free-form quality (objective match is a floor only).
JUDGED_KINDS = frozenset({NARRATIVE, INSIGHT, LONG_DOC})
ALL_KINDS = OBJECTIVE_KINDS | JUDGED_KINDS

# --- matching thresholds (PROPOSAL values; signed off / tuned via MH.2) --------------------

# Exact or normalized surface match -> full credit (1.0). Otherwise the pinned-embedder cosine
# decides: at/above TAU_FULL is a paraphrase/morphology match (full credit); in the partial band
# [TAU_PARTIAL, TAU_FULL) earns PARTIAL_CREDIT; below TAU_PARTIAL is no match.
TAU_FULL = 0.85
TAU_PARTIAL = 0.70
PARTIAL_CREDIT = 0.5

_PUNCT_STRIP = " \t\r\n.,;:!?\"'`«»“”()[]{}-–—"


def normalize_surface(text: str) -> str:
    """Casefold, collapse whitespace, and strip surrounding punctuation -- the canonical form
    used for exact label-ID surface matching (deliberately NOT lemmatization, per MH.2)."""
    return " ".join(text.casefold().split()).strip(_PUNCT_STRIP)


@dataclass(frozen=True)
class PlantedLabel:
    """A planted ground-truth label for one text-analysis sub-task (see `PlantedLabelRecord`)."""

    label_id: str
    kind: str
    value: str
    aliases: tuple[str, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)
    scoring: str = ""

    @property
    def surfaces(self) -> tuple[str, ...]:
        """All accepted surface forms (value + aliases)."""
        return (self.value, *self.aliases)

    @property
    def is_objective(self) -> bool:
        if self.scoring:
            return self.scoring == "objective"
        return self.kind in OBJECTIVE_KINDS

    @classmethod
    def from_record(cls, record: PlantedLabelRecord) -> "PlantedLabel":
        kind = record["kind"]
        if kind not in ALL_KINDS:
            raise ValueError(f"unknown text-analysis label kind: {kind!r}")
        return cls(
            label_id=record["label_id"],
            kind=kind,
            value=record["value"],
            aliases=tuple(record.get("aliases", []) or ()),
            attrs=dict(record.get("attrs", {}) or {}),
            scoring=record.get("scoring", ""),
        )


def load_planted_labels(records: list[PlantedLabelRecord]) -> list[PlantedLabel]:
    """Build `PlantedLabel`s from the planter's emitted records, rejecting unknown kinds."""
    return [PlantedLabel.from_record(r) for r in records]


# --- matching + per-sub-task scoring -------------------------------------------------------


def _credit(prediction: str, label: PlantedLabel, similarity: Similarity) -> float:
    """Credit a single prediction earns against one label: 1.0 for an exact/normalized surface
    match, else cosine-banded credit over the label's surfaces."""
    norm_pred = normalize_surface(prediction)
    if not norm_pred:
        return 0.0
    if any(norm_pred == normalize_surface(s) for s in label.surfaces):
        return 1.0
    best = max((similarity(prediction, s) for s in label.surfaces), default=0.0)
    if best >= TAU_FULL:
        return 1.0
    if best >= TAU_PARTIAL:
        return PARTIAL_CREDIT
    return 0.0


def score_subtask(
    predictions: list[str],
    labels: list[PlantedLabel],
    similarity: Similarity,
) -> SubtaskScore:
    """Score one sub-task over one document: greedily match predictions to planted labels
    (each prediction and each label used at most once, highest-credit pairs first), then report
    set precision / recall / F1 weighted by matched credit. Unmatched predictions are false
    positives (they lower precision -> hallucinated extractions are penalized)."""
    kind = labels[0].kind if labels else ""
    objective = labels[0].is_objective if labels else True

    # All scoring pairs with positive credit, best first; ties broken deterministically by index.
    pairs: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(predictions):
        for li, label in enumerate(labels):
            credit = _credit(pred, label, similarity)
            if credit > 0.0:
                pairs.append((credit, pi, li))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))

    used_pred: set[int] = set()
    used_label: set[int] = set()
    matched: list[tuple[str, float]] = []
    total_credit = 0.0
    for credit, pi, li in pairs:
        if pi in used_pred or li in used_label:
            continue
        used_pred.add(pi)
        used_label.add(li)
        matched.append((labels[li].label_id, credit))
        total_credit += credit

    n_labels = len(labels)
    n_pred = len(predictions)
    recall = total_credit / n_labels if n_labels else 0.0
    precision = total_credit / n_pred if n_pred else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    matched.sort(key=lambda m: m[0])
    return SubtaskScore(
        kind=kind,
        objective=objective,
        n_labels=n_labels,
        n_pred=n_pred,
        matched=matched,
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
    )


def score_document(
    predictions_by_kind: dict[str, list[str]],
    labels: list[PlantedLabel],
    similarity: Similarity,
) -> dict[str, Any]:
    """Score every sub-task present in `labels` for one document.

    `predictions_by_kind` maps a sub-task kind to the candidate's extracted surface strings for
    that kind. Returns per-sub-task `SubtaskScore`s plus the document objective headline (mean
    F1 over the OBJECTIVE sub-tasks that have planted labels). Judged sub-tasks are scored (as a
    floor) but kept out of the objective headline, which the gated judge owns.
    """
    by_kind: dict[str, list[PlantedLabel]] = {}
    for label in labels:
        by_kind.setdefault(label.kind, []).append(label)

    subtasks: dict[str, SubtaskScore] = {}
    for kind, kind_labels in by_kind.items():
        subtasks[kind] = score_subtask(predictions_by_kind.get(kind, []), kind_labels, similarity)

    objective_f1s = [s["f1"] for k, s in subtasks.items() if k in OBJECTIVE_KINDS]
    objective_score = sum(objective_f1s) / len(objective_f1s) if objective_f1s else 0.0
    return {
        "subtasks": subtasks,
        "objective_score": round(objective_score, 6),
        "n_objective_subtasks": len(objective_f1s),
    }


# --- default production similarity (pinned embedder cosine) --------------------------------


def embedder_similarity(embedder: Any = None) -> Similarity:
    """Production `similarity`: cosine over the PINNED embedder (the MH.2 matching basis).

    Vectors are L2-normalized by the `Embedder`, so cosine is their dot product. Heavy imports
    (the embedder, numpy) stay lazy; the returned callable caches encodings per surface string so
    a label's surfaces are embedded once across many predictions.
    """
    if embedder is None:
        from llb.rag.embedding import Embedder

        embedder = Embedder()
    cache: dict[str, Any] = {}

    def _vec(text: str) -> Any:
        if text not in cache:
            cache[text] = embedder.encode_queries([text])[0]
        return cache[text]

    def similarity(a: str, b: str) -> float:
        import numpy as np

        return float(np.dot(_vec(a), _vec(b)))

    return similarity
