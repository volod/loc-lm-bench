"""Explain a finalized run's wrong answers (miss analysis).

After any run or sweep, `llb analyze-misses` classifies every miss of one run bundle into
exactly one class -- retrieval miss (gold span absent from the retrieved context), generation
miss (evidence present, answer wrong), refusal, format/scoring artifact, or judge disagreement
-- clusters the misses by document, topic, and question type, and emits ranked, evidence-backed
recommendations (raise or lower `top_k`, change chunking, add prompt-system dictionary terms,
try the named alternative model). Every recommendation line names its numeric evidence.

Classification is span-overlap based: it reads the additive per-case `retrieval.jsonl` record
the runner persists beside `scores.jsonl` (falling back to the scored `retrieval_hit` for
legacy bundles). Everything here is pure and file-driven -- no endpoint, GPU, or store -- so the
whole classifier is unit-testable over a synthetic scored bundle. The bounded probe mode that
re-runs the miss subset at alternative retrieval depths lives in `miss_probe.py`; run bundles
are never mutated.
"""

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from llb.backends.base import ERR_BACKEND, ERR_TIMEOUT
from llb.core.contracts import JsonObject
from llb.eval import common as eval_common
from llb.goldset.schema import GoldItem
from llb.prompts import render_text
from llb.rag.retrieval import chunk_hits_any

_LOG = logging.getLogger(__name__)

# Miss classes (each miss lands in exactly ONE, decided in precedence order; see classify_case).
MISS_RETRIEVAL = "retrieval_miss"
MISS_GENERATION = "generation_miss"
MISS_REFUSAL = "refusal"
MISS_ARTIFACT = "format_artifact"
MISS_JUDGE = "judge_disagreement"
MISS_CLASSES = (MISS_RETRIEVAL, MISS_GENERATION, MISS_REFUSAL, MISS_ARTIFACT, MISS_JUDGE)

# A scoreable (status=ok) case below this objective score counts as a miss.
DEFAULT_MISS_THRESHOLD = 0.5
# A low-objective case the trusted judge rated at/above this is a judge DISAGREEMENT, not a
# generation miss -- the two signals conflict, so a human (or the calibration gate) should look.
JUDGE_AGREEMENT_MIN = 0.7
# Terminal statuses that are output/transport artifacts rather than model knowledge failures.
ARTIFACT_STATUSES = frozenset({eval_common.EMPTY, eval_common.MALFORMED, ERR_TIMEOUT, ERR_BACKEND})

# Probe interpretation thresholds: a deeper probe CONFIRMS the retrieval hypothesis when it
# recovers at least this fraction of the retrieval misses; a shallower probe recommends
# lowering top_k only when its mean objective beats the subset baseline by at least this much.
PROBE_CONFIRM_MIN = 0.5
PROBE_MIN_OBJECTIVE_GAIN = 0.05

# A generation-miss cluster this large (and holding at least this share of generation misses)
# earns a "prompt-system dictionary terms" recommendation for the clustered document/topic.
DICTIONARY_CLUSTER_MIN = 2
DICTIONARY_CLUSTER_SHARE = 0.5

MISS_ANALYSIS_METHOD = "miss-analysis"
RETRIEVAL_FILENAME = "retrieval.jsonl"
MISSES_FILENAME = "misses.jsonl"
REPORT_FILENAME = "report.md"
ANALYSIS_FILENAME = "analysis.json"
ITEM_PROVENANCE_FILENAME = "item_provenance.jsonl"
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

# The RAG knobs cited as recommendation evidence (mirrors board/recommend RAG_CONFIG_KEYS).
RAG_CONFIG_KEYS = ("strategy", "chunk_size", "chunk_overlap", "top_k", "retrieval_mode")

# Question-type heuristic: leading interrogative -> coarse type (UA + EN). Used only when the
# goldset has no `item_provenance.jsonl` sidecar carrying drafted `question_type` labels.
_QUESTION_TYPE_MARKERS = (
    ("хто", "who"),
    ("кого", "who"),
    ("коли", "when"),
    ("де", "where"),
    ("куди", "where"),
    ("скільки", "how_many"),
    ("чому", "why"),
    ("навіщо", "why"),
    ("як", "how"),
    ("який", "which"),
    ("яка", "which"),
    ("яке", "which"),
    ("які", "which"),
    ("якого", "which"),
    ("якої", "which"),
    ("що", "what"),
    ("чим", "what"),
    ("чого", "what"),
    ("who", "who"),
    ("when", "when"),
    ("where", "where"),
    ("why", "why"),
    ("how", "how"),
    ("which", "which"),
    ("what", "what"),
)
DEFAULT_QUESTION_TYPE = "other"

# Topic heuristic: the longest content token of the question (casefolded), skipping
# interrogatives/particles. Coarse but deterministic; a provenance sidecar `topic` wins.
_TOPIC_MIN_TOKEN_CHARS = 4
_TOPIC_STOPWORDS = frozenset(marker for marker, _ in _QUESTION_TYPE_MARKERS) | frozenset(
    {
        "було",
        "буде",
        "цього",
        "цієї",
        "року",
        "році",
        "щодо",
        "через",
        "після",
        "перед",
        "тому",
        "того",
        "може",
        "does",
        "with",
        "from",
    }
)

CLUSTER_DIMENSIONS = ("document", "topic", "question_type")


def _t(name: str, **values: object) -> str:
    """Render a `board.miss.<name>` text template (report prose lives in template files)."""
    return render_text(f"board.miss.{name}", values)


@dataclass(slots=True)
class MissRecord:
    """One classified miss (a `misses.jsonl` line)."""

    item_id: str
    miss_class: str
    status: str
    objective_score: float
    judge_score: float | None
    retrieval_hit: bool
    first_hit_rank: int | None
    question: str
    source_doc_id: str
    topic: str
    question_type: str
    answer_preview: str

    def as_dict(self) -> JsonObject:
        return {
            "item_id": self.item_id,
            "miss_class": self.miss_class,
            "status": self.status,
            "objective_score": self.objective_score,
            "judge_score": self.judge_score,
            "retrieval_hit": self.retrieval_hit,
            "first_hit_rank": self.first_hit_rank,
            "question": self.question,
            "source_doc_id": self.source_doc_id,
            "topic": self.topic,
            "question_type": self.question_type,
            "answer_preview": self.answer_preview,
        }


@dataclass(slots=True)
class ClusterRow:
    """Miss density for one cluster key (document / topic / question type)."""

    key: str
    n_misses: int
    n_cases: int

    @property
    def miss_rate(self) -> float:
        return self.n_misses / self.n_cases if self.n_cases else 0.0

    def as_dict(self) -> JsonObject:
        return {
            "key": self.key,
            "n_misses": self.n_misses,
            "n_cases": self.n_cases,
            "miss_rate": round(self.miss_rate, 4),
        }


@dataclass
class MissAnalysis:
    """The full analysis of one run bundle: classified misses, clusters, recommendations."""

    run_dir: str
    model: str
    backend: str
    split: str
    n_cases: int
    threshold: float
    rag_config: JsonObject
    misses: list[MissRecord]
    class_counts: dict[str, int]
    clusters: dict[str, list[ClusterRow]]
    recommendations: list[JsonObject] = field(default_factory=list)
    probes: list[JsonObject] = field(default_factory=list)


# --------------------------------------------------------------------------- bundle loading


def load_scored_bundle(
    run_dir: Path | str,
) -> tuple[JsonObject, list[JsonObject], dict[str, JsonObject]]:
    """Read a finalized run bundle: (manifest, score rows, retrieval records by item id).

    The retrieval map is empty for legacy bundles that predate `retrieval.jsonl`; the
    classifier then falls back to the scored `retrieval_hit` signal.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    scores_path = run_dir / "scores.jsonl"
    if not manifest_path.is_file() or not scores_path.is_file():
        raise SystemExit(
            f"[analyze-misses] {run_dir} is not a finalized run bundle "
            "(manifest.json + scores.jsonl required)"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(scores_path)
    retrieval_path = run_dir / RETRIEVAL_FILENAME
    retrieval: dict[str, JsonObject] = {}
    if retrieval_path.is_file():
        retrieval = {str(rec["item_id"]): rec for rec in _read_jsonl(retrieval_path)}
    else:
        _LOG.warning(
            "[analyze-misses] %s has no %s (older bundle); span-overlap classification "
            "falls back to the scored retrieval_hit signal",
            run_dir,
            RETRIEVAL_FILENAME,
        )
    return manifest, rows, retrieval


def _read_jsonl(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_item_provenance(goldset_path: Path | str) -> dict[str, JsonObject]:
    """Draft-bundle sidecar labels (`item_provenance.jsonl` beside the goldset), keyed by item
    id. Soft input: absent for plain goldsets, then heuristics label question type and topic."""
    sidecar = Path(goldset_path).parent / ITEM_PROVENANCE_FILENAME
    if not sidecar.is_file():
        return {}
    return {str(row.get("id")): row for row in _read_jsonl(sidecar)}


# --------------------------------------------------------------------------- classification


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


# --------------------------------------------------------------------------- recommendations


def _fmt_rate(value: float) -> str:
    return f"{value:.0%}"


def _probe_note(analysis: MissAnalysis) -> tuple[str, JsonObject | None]:
    """The raise-top_k evidence fragment from the deepest above-current probe, when one ran."""
    top_k = analysis.rag_config.get("top_k") or 0
    deeper = [p for p in analysis.probes if int(p["top_k"]) > int(top_k)]
    if not deeper:
        return "", None
    probe = max(deeper, key=lambda p: int(p["top_k"]))
    n_retrieval = int(probe.get("n_retrieval_misses", 0))
    recovered = int(probe.get("recovered_retrieval", 0))
    if n_retrieval and recovered / n_retrieval >= PROBE_CONFIRM_MIN:
        note = _t("probe_confirmed", k=probe["top_k"], recovered=recovered, n=n_retrieval)
    else:
        note = _t("probe_rejected", k=probe["top_k"], recovered=recovered, n=n_retrieval)
    return note, probe


def _retrieval_recommendations(analysis: MissAnalysis) -> list[JsonObject]:
    n_retrieval = analysis.class_counts[MISS_RETRIEVAL]
    if not n_retrieval:
        return []
    recs: list[JsonObject] = []
    top_k = analysis.rag_config.get("top_k")
    note, probe = _probe_note(analysis)
    recs.append(
        {
            "action": "raise_top_k",
            "weight": n_retrieval,
            "line": _t(
                "rec_raise_top_k",
                top_k=top_k,
                n_retrieval=n_retrieval,
                n_misses=len(analysis.misses),
                probe_note=note,
            ),
        }
    )
    if probe is not None:
        unrecovered = int(probe["n_retrieval_misses"]) - int(probe["recovered_retrieval"])
        if unrecovered > 0:
            recs.append(
                {
                    "action": "change_chunking",
                    "weight": unrecovered,
                    "line": _t(
                        "rec_chunking_probed",
                        strategy=analysis.rag_config.get("strategy"),
                        size=analysis.rag_config.get("chunk_size"),
                        overlap=analysis.rag_config.get("chunk_overlap"),
                        n_unrecovered=unrecovered,
                        n_retrieval=probe["n_retrieval_misses"],
                        max_k=probe["top_k"],
                    ),
                }
            )
    else:
        recs.append(
            {
                "action": "change_chunking",
                "weight": n_retrieval,
                "line": _t(
                    "rec_chunking_unprobed",
                    strategy=analysis.rag_config.get("strategy"),
                    size=analysis.rag_config.get("chunk_size"),
                    overlap=analysis.rag_config.get("chunk_overlap"),
                    n_retrieval=n_retrieval,
                    top_k=top_k,
                ),
            }
        )
    return recs


def _lower_top_k_recommendation(analysis: MissAnalysis) -> list[JsonObject]:
    """Recommend a SHALLOWER context only when a below-current probe measurably beat the miss
    subset's original mean objective (fewer distractors helped)."""
    top_k = analysis.rag_config.get("top_k") or 0
    shallower = [p for p in analysis.probes if int(p["top_k"]) < int(top_k)]
    if not shallower:
        return []
    best = max(shallower, key=lambda p: float(p.get("mean_objective", 0.0)))
    base = float(best.get("base_mean_objective", 0.0))
    gain = float(best.get("mean_objective", 0.0)) - base
    if gain < PROBE_MIN_OBJECTIVE_GAIN:
        return []
    return [
        {
            "action": "lower_top_k",
            "weight": len(analysis.misses),
            "line": _t(
                "rec_lower_top_k",
                k=best["top_k"],
                top_k=top_k,
                probe_obj=f"{float(best['mean_objective']):.3f}",
                base_obj=f"{base:.3f}",
            ),
        }
    ]


def _generation_recommendations(
    analysis: MissAnalysis, alternatives: list[tuple[str, float]]
) -> list[JsonObject]:
    n_generation = analysis.class_counts[MISS_GENERATION]
    if not n_generation:
        return []
    recs: list[JsonObject] = []
    generation_misses = [m for m in analysis.misses if m.miss_class == MISS_GENERATION]
    cluster = _dominant_generation_cluster(analysis, generation_misses)
    if cluster is not None:
        dimension, key, count = cluster
        recs.append(
            {
                "action": "dictionary_terms",
                "weight": count,
                "line": _t(
                    "rec_dictionary",
                    cluster=key,
                    dimension=dimension,
                    n_cluster=count,
                    n_generation=n_generation,
                ),
            }
        )
    better = [(m, obj) for m, obj in alternatives if m != analysis.model]
    if better:
        alt_model, alt_objective = max(better, key=lambda pair: pair[1])
        run_objective = _run_mean_objective(analysis)
        if alt_objective > run_objective:
            recs.append(
                {
                    "action": "alternative_model",
                    "weight": n_generation,
                    "line": _t(
                        "rec_alternative_model",
                        alt_model=alt_model,
                        alt_objective=f"{alt_objective:.3f}",
                        objective=f"{run_objective:.3f}",
                        model=analysis.model,
                        split=analysis.split,
                        n_generation=n_generation,
                    ),
                }
            )
    return recs


def _dominant_generation_cluster(
    analysis: MissAnalysis, generation_misses: list[MissRecord]
) -> tuple[str, str, int] | None:
    """The (dimension, key, count) of the densest document/topic cluster of generation misses,
    when it is big enough to suggest missing domain vocabulary."""
    best: tuple[str, str, int] | None = None
    for dimension, attribute in (("document", "source_doc_id"), ("topic", "topic")):
        counts = Counter(getattr(m, attribute) for m in generation_misses)
        if not counts:
            continue
        key, count = counts.most_common(1)[0]
        if count >= DICTIONARY_CLUSTER_MIN and count / len(generation_misses) >= (
            DICTIONARY_CLUSTER_SHARE
        ):
            if best is None or count > best[2]:
                best = (dimension, key, count)
    return best


def _run_mean_objective(analysis: MissAnalysis) -> float:
    """Mean objective of the analyzed run, recovered from its manifest via the bundle path."""
    try:
        manifest = json.loads(
            (Path(analysis.run_dir) / "manifest.json").read_text(encoding="utf-8")
        )
        return float((manifest.get("metrics") or {}).get("objective_score", 0.0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def _status_recommendations(analysis: MissAnalysis) -> list[JsonObject]:
    recs: list[JsonObject] = []
    n_refusal = analysis.class_counts[MISS_REFUSAL]
    if n_refusal:
        recs.append(
            {
                "action": "refusal_review",
                "weight": n_refusal,
                "line": _t("rec_refusals", n_refusal=n_refusal, n_misses=len(analysis.misses)),
            }
        )
    n_artifact = analysis.class_counts[MISS_ARTIFACT]
    if n_artifact:
        breakdown = Counter(m.status for m in analysis.misses if m.miss_class == MISS_ARTIFACT)
        recs.append(
            {
                "action": "artifact_review",
                "weight": n_artifact,
                "line": _t(
                    "rec_artifacts",
                    n_artifact=n_artifact,
                    breakdown=", ".join(f"{status}={n}" for status, n in sorted(breakdown.items())),
                ),
            }
        )
    n_judge = analysis.class_counts[MISS_JUDGE]
    if n_judge:
        recs.append(
            {
                "action": "judge_review",
                "weight": n_judge,
                "line": _t(
                    "rec_judge",
                    n_judge=n_judge,
                    judge_min=JUDGE_AGREEMENT_MIN,
                    threshold=analysis.threshold,
                ),
            }
        )
    return recs


def build_recommendations(
    analysis: MissAnalysis, *, alternatives: list[tuple[str, float]] | None = None
) -> list[JsonObject]:
    """Ranked, evidence-backed recommendation lines; heaviest miss pile first."""
    recs = (
        _retrieval_recommendations(analysis)
        + _lower_top_k_recommendation(analysis)
        + _generation_recommendations(analysis, alternatives or [])
        + _status_recommendations(analysis)
    )
    return sorted(recs, key=lambda rec: int(rec["weight"]), reverse=True)


def refresh_recommendations(
    analysis: MissAnalysis, *, alternatives: list[tuple[str, float]] | None = None
) -> None:
    """Rebuild the ranked recommendations (call after attaching probe outcomes)."""
    analysis.recommendations = build_recommendations(analysis, alternatives=alternatives)


# --------------------------------------------------------------------------- report + artifacts


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _class_table(analysis: MissAnalysis) -> str:
    n_misses = len(analysis.misses) or 1
    rows = [
        [cls, str(count), _fmt_rate(count / n_misses)]
        for cls, count in analysis.class_counts.items()
        if count
    ]
    return _md_table(["miss class", "n", "share of misses"], rows)


def _cluster_table(rows: list[ClusterRow]) -> str:
    return _md_table(
        ["key", "misses", "cases", "miss rate"],
        [[c.key, str(c.n_misses), str(c.n_cases), _fmt_rate(c.miss_rate)] for c in rows],
    )


def _probe_table(probes: list[JsonObject]) -> str:
    rows = [
        [
            str(p["top_k"]),
            f"{float(p.get('mean_objective', 0.0)):.3f}",
            f"{float(p.get('base_mean_objective', 0.0)):.3f}",
            f"{p.get('recovered_retrieval', 0)}/{p.get('n_retrieval_misses', 0)}",
            str(p.get("run_dir", "?")),
        ]
        for p in sorted(probes, key=lambda p: int(p["top_k"]))
    ]
    return _md_table(
        [
            "probe top_k",
            "mean objective",
            "baseline objective",
            "retrieval misses recovered",
            "run",
        ],
        rows,
    )


def format_report_md(analysis: MissAnalysis) -> str:
    """Render the analysis as the Markdown report written beside `misses.jsonl`."""
    lines = [
        "# loc-lm-bench miss analysis",
        "",
        _t(
            "header_line",
            run_dir=analysis.run_dir,
            model=analysis.model,
            backend=analysis.backend,
            split=analysis.split,
            n_cases=analysis.n_cases,
            threshold=analysis.threshold,
        ),
    ]
    if not analysis.misses:
        lines += ["", _t("no_misses", threshold=analysis.threshold)]
        return "\n".join(lines)
    lines += [
        _t(
            "summary_line",
            n_misses=len(analysis.misses),
            n_cases=analysis.n_cases,
            pct=_fmt_rate(len(analysis.misses) / analysis.n_cases if analysis.n_cases else 0.0),
        ),
        "",
        "## Miss classes",
        "",
        _class_table(analysis),
    ]
    for dimension in CLUSTER_DIMENSIONS:
        rows = analysis.clusters.get(dimension, [])
        if rows:
            lines += ["", f"## Misses by {dimension.replace('_', ' ')}", "", _cluster_table(rows)]
    if analysis.probes:
        lines += [
            "",
            "## Retrieval-depth probes (miss subset only)",
            "",
            _probe_table(analysis.probes),
        ]
    if analysis.recommendations:
        lines += ["", "## Recommendations", ""]
        lines += [f"{rank}. {rec['line']}" for rank, rec in enumerate(analysis.recommendations, 1)]
    return "\n".join(lines)


def analysis_payload(analysis: MissAnalysis) -> JsonObject:
    """Machine-readable summary (`analysis.json`) consumed by `llb recommend`."""
    return {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": analysis.run_dir,
        "model": analysis.model,
        "backend": analysis.backend,
        "split": analysis.split,
        "n_cases": analysis.n_cases,
        "n_misses": len(analysis.misses),
        "threshold": analysis.threshold,
        "rag_config": analysis.rag_config,
        "class_counts": analysis.class_counts,
        "clusters": {
            dimension: [row.as_dict() for row in rows]
            for dimension, rows in analysis.clusters.items()
        },
        "probes": analysis.probes,
        "recommendations": analysis.recommendations,
    }


def write_analysis(analysis: MissAnalysis, out_dir: Path | str) -> dict[str, str]:
    """Persist report.md + misses.jsonl + analysis.json under one analysis directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / REPORT_FILENAME
    report_path.write_text(format_report_md(analysis) + "\n", encoding="utf-8")
    misses_path = out_dir / MISSES_FILENAME
    misses_path.write_text(
        "".join(json.dumps(m.as_dict(), ensure_ascii=False) + "\n" for m in analysis.misses),
        encoding="utf-8",
    )
    analysis_path = out_dir / ANALYSIS_FILENAME
    analysis_path.write_text(
        json.dumps(analysis_payload(analysis), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "report": str(report_path),
        "misses": str(misses_path),
        "analysis": str(analysis_path),
    }


def analysis_out_dir(data_dir: Path | str) -> Path:
    """A fresh `$DATA_DIR/miss-analysis/<timestamp>/` directory path (not yet created)."""
    stamp = datetime.now(timezone.utc).strftime(_TIMESTAMP_FORMAT)
    return Path(data_dir) / MISS_ANALYSIS_METHOD / stamp


def latest_analysis(data_dir: Path | str) -> JsonObject | None:
    """The newest persisted `analysis.json` under `$DATA_DIR/miss-analysis/`, with its report
    path attached -- or None when no analysis has ever run (recommend then omits the section)."""
    root = Path(data_dir) / MISS_ANALYSIS_METHOD
    if not root.is_dir():
        return None
    for candidate in sorted(root.iterdir(), reverse=True):
        payload_path = candidate / ANALYSIS_FILENAME
        if payload_path.is_file():
            try:
                payload: JsonObject = json.loads(payload_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            payload["report_path"] = str(candidate / REPORT_FILENAME)
            return payload
    return None
