"""Classify each scored case into exactly one miss class (span-overlap based), cluster the misses
by document / topic / question type, and assemble the `MissAnalysis` for one run bundle.

`analyze_run` is the entry point: it loads the bundle, classifies + clusters, then attaches the
ranked recommendations built in `recommendations.py`.
"""

import re
from collections import Counter
from pathlib import Path

from llb.board.miss_analysis.load import load_scored_bundle
from llb.board.miss_analysis.model import (
    ARTIFACT_STATUSES,
    CLUSTER_DIMENSIONS,
    DEFAULT_MISS_THRESHOLD,
    DEFAULT_QUESTION_TYPE,
    JUDGE_AGREEMENT_MIN,
    MISS_ARTIFACT,
    MISS_CLASSES,
    MISS_GENERATION,
    MISS_JUDGE,
    MISS_REFUSAL,
    MISS_RETRIEVAL,
    RAG_CONFIG_KEYS,
    _QUESTION_TYPE_MARKERS,
    _TOPIC_MIN_TOKEN_CHARS,
    _TOPIC_STOPWORDS,
    ClusterRow,
    MissAnalysis,
    MissRecord,
)
from llb.board.miss_analysis.recommendations import build_recommendations
from llb.core.contracts import JsonObject
from llb.eval import common as eval_common
from llb.goldset.schema import GoldItem
from llb.rag.retrieval import chunk_hits_any


def retrieval_hit_from_record(record: JsonObject) -> bool:
    """Span-overlap hit check over a persisted `retrieval.jsonl` record."""
    gold_spans = [dict(span) for span in record.get("gold_spans", [])]
    return any(
        chunk_hits_any(dict(chunk), gold_spans)  # type: ignore[arg-type]
        for chunk in record.get("retrieved", [])
    )


def _case_retrieval_hit(row: JsonObject, record: JsonObject | None) -> bool:
    if record is not None:
        return retrieval_hit_from_record(record)
    return float(row.get("retrieval_hit", 0.0) or 0.0) > 0.0


def classify_case(
    row: JsonObject,
    retrieval_record: JsonObject | None,
    *,
    threshold: float = DEFAULT_MISS_THRESHOLD,
    judge_agreement_min: float = JUDGE_AGREEMENT_MIN,
) -> str | None:
    """Map one scored case to its miss class, or None when the case is not a miss.

    Precedence keeps the classes disjoint (zero cross-class leakage): refusal, then
    format/transport artifact, then retrieval miss (typed status or span-overlap absence of the
    gold span), then judge disagreement, then generation miss.
    """
    status = str(row.get("status", eval_common.OK))
    if status == eval_common.REFUSAL:
        return MISS_REFUSAL
    if status in ARTIFACT_STATUSES:
        return MISS_ARTIFACT
    if status == eval_common.RETRIEVAL_MISS:
        return MISS_RETRIEVAL
    if float(row.get("objective_score", 0.0)) >= threshold:
        return None
    if not _case_retrieval_hit(row, retrieval_record):
        return MISS_RETRIEVAL
    judge = row.get("judge_score")
    if judge is not None and float(judge) >= judge_agreement_min:
        return MISS_JUDGE
    return MISS_GENERATION


def question_type_of(question: str, provenance_row: JsonObject | None) -> str:
    """Drafted `question_type` label when the sidecar has one, else the interrogative heuristic."""
    if provenance_row is not None and provenance_row.get("question_type"):
        return str(provenance_row["question_type"])
    tokens = re.findall(r"\w+", question.casefold())
    for token in tokens[:3]:  # the interrogative leads a question (allow a short preamble)
        for marker, qtype in _QUESTION_TYPE_MARKERS:
            if token == marker:
                return qtype
    return DEFAULT_QUESTION_TYPE


def topic_of(question: str, provenance_row: JsonObject | None) -> str:
    """Drafted `topic` label when present, else the longest content token of the question.

    The heuristic token is lemmatized best-effort (identity when the `[lex]` extra is absent),
    so Ukrainian case forms of one topic ("начальник" / "начальника") collapse into a single
    cluster key instead of splitting the same topic across inflections.
    """
    if provenance_row is not None and provenance_row.get("topic"):
        return str(provenance_row["topic"])
    tokens = re.findall(r"\w+", question.casefold())
    candidates = [
        token
        for token in tokens
        if len(token) >= _TOPIC_MIN_TOKEN_CHARS and token not in _TOPIC_STOPWORDS
    ]
    if not candidates:
        return DEFAULT_QUESTION_TYPE
    from llb.rag.lexical import best_effort_lemma

    return best_effort_lemma(max(candidates, key=len))


def _cluster_keys(item: GoldItem | None, provenance_row: JsonObject | None) -> dict[str, str]:
    if item is None:
        return {dimension: "?" for dimension in CLUSTER_DIMENSIONS}
    return {
        "document": item.source_doc_id,
        "topic": topic_of(item.question, provenance_row),
        "question_type": question_type_of(item.question, provenance_row),
    }


def _build_clusters(
    rows: list[JsonObject],
    miss_ids: set[str],
    keys_by_item: dict[str, dict[str, str]],
) -> dict[str, list[ClusterRow]]:
    """Miss density per cluster key, per dimension, over ALL scored cases (so rates are
    relative to how often the group was asked, not just to the miss pile)."""
    clusters: dict[str, list[ClusterRow]] = {}
    for dimension in CLUSTER_DIMENSIONS:
        totals: Counter[str] = Counter()
        missed: Counter[str] = Counter()
        for row in rows:
            item_id = str(row.get("item_id"))
            key = keys_by_item.get(item_id, {}).get(dimension, "?")
            totals[key] += 1
            if item_id in miss_ids:
                missed[key] += 1
        ranked = [
            ClusterRow(key=key, n_misses=missed[key], n_cases=totals[key])
            for key in totals
            if missed[key] > 0
        ]
        ranked.sort(key=lambda c: (c.n_misses, c.miss_rate), reverse=True)
        clusters[dimension] = ranked
    return clusters


def analyze_run(
    run_dir: Path | str,
    items: list[GoldItem],
    *,
    threshold: float = DEFAULT_MISS_THRESHOLD,
    judge_agreement_min: float = JUDGE_AGREEMENT_MIN,
    provenance: dict[str, JsonObject] | None = None,
    alternatives: list[tuple[str, float]] | None = None,
) -> MissAnalysis:
    """Classify and cluster one finalized run bundle's misses and build recommendations.

    `items` is the goldset the run scored (for question / document / label metadata);
    `alternatives` is `[(model, objective_score), ...]` of comparable sibling runs, so the
    "try the named alternative model" recommendation can cite measured numbers.
    """
    manifest, rows, retrieval = load_scored_bundle(run_dir)
    provenance = provenance or {}
    items_by_id = {item.id: item for item in items}

    misses: list[MissRecord] = []
    keys_by_item: dict[str, dict[str, str]] = {}
    for row in rows:
        item_id = str(row.get("item_id"))
        item = items_by_id.get(item_id)
        provenance_row = provenance.get(item_id)
        keys_by_item[item_id] = _cluster_keys(item, provenance_row)
        record = retrieval.get(item_id)
        miss_class = classify_case(
            row, record, threshold=threshold, judge_agreement_min=judge_agreement_min
        )
        if miss_class is None:
            continue
        judge = row.get("judge_score")
        rank = row.get("first_hit_rank")
        misses.append(
            MissRecord(
                item_id=item_id,
                miss_class=miss_class,
                status=str(row.get("status", "")),
                objective_score=float(row.get("objective_score", 0.0)),
                judge_score=float(judge) if judge is not None else None,
                retrieval_hit=_case_retrieval_hit(row, record),
                first_hit_rank=int(rank) if rank is not None else None,
                question=item.question if item else "",
                source_doc_id=keys_by_item[item_id]["document"],
                topic=keys_by_item[item_id]["topic"],
                question_type=keys_by_item[item_id]["question_type"],
                answer_preview=str(row.get("answer_preview", "")),
            )
        )

    config = manifest.get("config") or {}
    class_counts = {cls: 0 for cls in MISS_CLASSES}
    for miss in misses:
        class_counts[miss.miss_class] += 1
    analysis = MissAnalysis(
        run_dir=str(run_dir),
        model=str(config.get("model", "?")),
        backend=str(config.get("backend", "?")),
        split=str(manifest.get("split", "?")),
        n_cases=len(rows),
        threshold=threshold,
        rag_config={key: config.get(key) for key in RAG_CONFIG_KEYS},
        misses=misses,
        class_counts=class_counts,
        clusters=_build_clusters(rows, {m.item_id for m in misses}, keys_by_item),
    )
    analysis.recommendations = build_recommendations(analysis, alternatives=alternatives or [])
    return analysis
