"""text-analysis and category expansion scored text-analysis runner -- objective planted-label recovery.

Drives a candidate over a synthetic text-analysis bundle (the planter's `corpus/` docs plus
`text_analysis_labels.jsonl` of `PlantedLabelRecord`s), asks it to extract the per-sub-task
elements present in each document, scores recovery with the text-analysis sign-off matching engine
(`llb.scoring.text_analysis`), and aggregates one `ModelResult` under `TIER_TEXT_ANALYSIS` --
its OWN Tier, never cross-ranked with the RAG board. The per-document objective scores carry the
bootstrap CI; real-corpus and synthetic bundles are run + reported SEPARATELY (the planter tags
`synthetic: true` in provenance).

The model is reached through an injectable `complete` (prompt -> raw text) and the cosine is an
injectable `similarity`, so the whole flow runs from a FAKE endpoint with no GPU or embedder.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.bench.common import (
    DEFAULT_THRESHOLD,
    JudgeScorer,
    LLMComplete,
    Mirror,
    ThroughputMeter,
    category_result,
    mean,
    persist_category_run,
    render_board,
    run_gated_judge,
    verified_data_config,
)
from llb.core.contracts import (
    BoardRow,
    JudgeInputRecord,
    JudgeScore,
    JudgeStatus,
    RunMetrics,
    RunPaths,
    TextAnalysisCaseRow,
)
from llb.eval.common import EMPTY, MALFORMED, OK
from llb.eval.map_reduce import run_map_reduce_text
from llb.prep.frontier import parse_json_block
from llb.prompts import render_text, render_text_map
from llb.rag.chunking import iter_docs
from llb.scoring import text_analysis as ta
from llb.scoring.aggregate import TIER_TEXT_ANALYSIS, ModelResult, bootstrap_mean_ci
from llb.scoring.judge import JudgeOutcome

_LOG = logging.getLogger(__name__)

METHOD = "text-analysis"
TEXT_ANALYSIS_LABELS = "text_analysis_labels.jsonl"

# The judged free-form sub-tasks the gated judge owns (objective match is only a floor for them).
_JUDGED_EXTRACT_KINDS = (ta.NARRATIVE, ta.INSIGHT)
# A UA intent framing per judged sub-task: faithfulness scores grounding in the doc, answer-
# relevancy scores whether the free-form output addresses the sub-task.
_JUDGE_INTENT = render_text_map("bench.text_analysis.judge_intents")
_DEFAULT_LONG_DOC_QUESTION = render_text("bench.text_analysis.long_doc_default_question")

# UA instruction phrasing per sub-task kind (the candidate-facing extraction ask).
_KIND_UA = render_text_map("bench.text_analysis.kind_labels")


@dataclass(slots=True)
class TextAnalysisRun:
    """Outcome of one scored text-analysis run."""

    result: ModelResult
    rows: list[TextAnalysisCaseRow]
    board: list[BoardRow]
    table: str
    paths: RunPaths | None
    judged_quality: float | None = None  # mean gated-judge quality (None when not trusted/run)
    judged_quality_ci: tuple[float, float] | None = None
    judge_trusted: bool = False
    judge_reason: str = "no judge configured"


@dataclass(slots=True)
class _ScoredTextAnalysisDocs:
    doc_ids: list[str]
    rows: list[TextAnalysisCaseRow]
    case_objectives: list[float]
    judge_records: list[JudgeInputRecord]
    judge_row_index: list[int]
    n_ok: int


@dataclass(slots=True)
class _JudgeQualityResult:
    outcome: JudgeOutcome
    value: float | None
    ci: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class _JudgeConfig:
    model: str | None
    rho: float | None
    threshold: float
    scorer: JudgeScorer | None
    base_url: str | None


@dataclass(frozen=True, slots=True)
class _TextAnalysisPersistInput:
    data_dir: Path | str | None
    run_name: str
    model: str
    backend: str
    bundle: Path | str
    synthetic: bool
    n_docs: int
    result: ModelResult
    reliability: float
    rows: list[TextAnalysisCaseRow]
    judge_result: _JudgeQualityResult
    judge_config: _JudgeConfig
    verification_cfg: dict[str, object]
    tokens_per_s: float
    mirror: Mirror | None


def analysis_prompt(doc_id: str, text: str, kinds: Sequence[str]) -> str:
    """Ask the candidate to extract each requested sub-task's elements as a JSON object keyed by
    the sub-task name, value = a list of short Ukrainian strings."""
    bullets = "\n".join(f"- {kind}: {_KIND_UA.get(kind, kind)}" for kind in kinds)
    keys = ", ".join(kinds)
    return render_text(
        "bench.text_analysis.analysis",
        {"doc_id": doc_id, "text": text, "bullets": bullets, "keys": keys},
    )


def parse_predictions(raw: str, kinds: Sequence[str]) -> dict[str, list[str]]:
    """Parse the candidate's JSON into {kind: [surface strings]}; raises on a non-object payload."""
    payload = parse_json_block(raw)
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object keyed by sub-task")
    out: dict[str, list[str]] = {}
    for kind in kinds:
        value = payload.get(kind, [])
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            out[kind] = []
            continue
        out[kind] = [str(v).strip() for v in value if str(v).strip()]
    return out


def load_planted_by_doc(bundle: Path | str) -> dict[str, list[ta.PlantedLabel]]:
    """Load `text_analysis_labels.jsonl` and group the planted labels by their `doc_id`."""
    path = Path(bundle) / TEXT_ANALYSIS_LABELS
    by_doc: dict[str, list[ta.PlantedLabelRecord]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record: ta.PlantedLabelRecord = json.loads(line)
        by_doc.setdefault(str(record.get("doc_id", "")), []).append(record)
    return {doc_id: ta.load_planted_labels(records) for doc_id, records in by_doc.items()}


def _case_row(
    doc_id: str, status: str, scored: dict[str, Any], n_labels: int
) -> TextAnalysisCaseRow:
    f1_by_kind = {kind: subtask["f1"] for kind, subtask in scored["subtasks"].items()}
    return {
        "item_id": doc_id,
        "status": status,
        "objective_score": float(scored["objective_score"]),
        "n_objective_subtasks": int(scored["n_objective_subtasks"]),
        "n_labels": n_labels,
        "subtask_f1_json": json.dumps(f1_by_kind, ensure_ascii=False, sort_keys=True),
    }


def judged_quality(score: JudgeScore) -> float:
    """Collapse the judge's two G-Eval signals into one free-form quality scalar: the output is
    GROUNDED in the document (faithfulness) AND addresses the sub-task (answer_relevancy)."""
    return (float(score["faithfulness"]) + float(score["answer_relevancy"])) / 2.0


def long_doc_question(labels: list[ta.PlantedLabel]) -> str | None:
    """The comprehension question for a doc's `long_doc` label (`attrs.question`), else a default;
    None when the doc plants no long_doc label."""
    for label in labels:
        if label.kind == ta.LONG_DOC:
            return str(label.attrs.get("question") or "").strip() or _DEFAULT_LONG_DOC_QUESTION
    return None


def _judged_answer(predictions: dict[str, list[str]], kind: str) -> str:
    """Join a judged sub-task's extracted surfaces into one free-form answer for the judge."""
    return " ".join(predictions.get(kind, [])).strip()


def _load_corpus_docs(bundle: Path | str) -> dict[str, str]:
    # `iter_docs` ids are corpus-relative paths WITH the extension (e.g. "synth-000.md");
    # planted labels key by the planter's bare doc id ("synth-000"). Index both forms.
    docs: dict[str, str] = {}
    for rel, text in iter_docs(Path(bundle) / "corpus"):
        docs[rel] = text
        docs.setdefault(Path(rel).stem, text)
    return docs


def _matching_doc_ids(
    bundle: Path | str,
    labels_by_doc: dict[str, list[ta.PlantedLabel]],
    docs: dict[str, str],
    limit: int | None,
) -> list[str]:
    doc_ids = sorted(doc_id for doc_id in labels_by_doc if doc_id in docs)
    if not doc_ids:
        raise SystemExit(
            f"no text-analysis documents with planted labels under {bundle} "
            f"(need {TEXT_ANALYSIS_LABELS} + a matching corpus/)"
        )
    return doc_ids[:limit] if limit is not None else doc_ids


def _extract_kinds(labels: list[ta.PlantedLabel]) -> list[str]:
    return sorted({label.kind for label in labels if label.kind != ta.LONG_DOC})


def _predict_doc_extractions(
    doc_id: str,
    doc_text: str,
    labels: list[ta.PlantedLabel],
    complete: LLMComplete,
) -> tuple[str, dict[str, list[str]]]:
    extract_kinds = _extract_kinds(labels)
    raw = complete(analysis_prompt(doc_id, doc_text, extract_kinds)) if extract_kinds else ""
    if extract_kinds and not raw.strip():
        return EMPTY, {}
    if not extract_kinds:
        return OK, {}
    try:
        return OK, parse_predictions(raw, extract_kinds)
    except (ValueError, json.JSONDecodeError):
        return MALFORMED, {}


def _long_doc_judge_record(
    labels: list[ta.PlantedLabel],
    doc_text: str,
    complete: LLMComplete,
) -> tuple[str, JudgeInputRecord] | None:
    question = long_doc_question(labels)
    if question is None:
        return None
    answer = run_map_reduce_text(complete, question, doc_text)
    return answer, {"question": question, "answer": answer, "contexts": [doc_text]}


def _append_freeform_judge_records(
    predictions: dict[str, list[str]],
    doc_text: str,
    judge_records: list[JudgeInputRecord],
    judge_row_index: list[int],
    row_idx: int,
) -> None:
    for kind in _JUDGED_EXTRACT_KINDS:
        answer = _judged_answer(predictions, kind)
        if answer:
            judge_records.append(
                {
                    "question": _JUDGE_INTENT[kind],
                    "answer": answer,
                    "contexts": [doc_text],
                }
            )
            judge_row_index.append(row_idx)


def _score_doc_batch(
    doc_ids: list[str],
    labels_by_doc: dict[str, list[ta.PlantedLabel]],
    docs: dict[str, str],
    complete: LLMComplete,
    similarity: ta.Similarity,
) -> _ScoredTextAnalysisDocs:
    rows: list[TextAnalysisCaseRow] = []
    case_objectives: list[float] = []
    judge_records: list[JudgeInputRecord] = []
    judge_row_index: list[int] = []
    n_ok = 0

    for doc_id in doc_ids:
        labels = labels_by_doc[doc_id]
        doc_text = docs[doc_id]
        status, predictions = _predict_doc_extractions(doc_id, doc_text, labels, complete)
        scored = ta.score_document(predictions, labels, similarity)
        if status == OK:
            n_ok += 1
        case_objectives.append(float(scored["objective_score"]))
        row = _case_row(doc_id, status, scored, len(labels))
        row_idx = len(rows)
        long_doc_record = _long_doc_judge_record(labels, doc_text, complete)
        if long_doc_record is not None:
            answer, record = long_doc_record
            row["long_doc_answer"] = answer[:280]
            judge_records.append(record)
            judge_row_index.append(row_idx)
        _append_freeform_judge_records(
            predictions, doc_text, judge_records, judge_row_index, row_idx
        )
        rows.append(row)

    return _ScoredTextAnalysisDocs(
        doc_ids=doc_ids,
        rows=rows,
        case_objectives=case_objectives,
        judge_records=judge_records,
        judge_row_index=judge_row_index,
        n_ok=n_ok,
    )


def _attach_judged_quality(
    rows: list[TextAnalysisCaseRow],
    scores: list[JudgeScore],
    judge_row_index: list[int],
) -> tuple[float, tuple[float, float] | None]:
    per_record = [judged_quality(score) for score in scores]
    per_row: dict[int, list[float]] = {}
    for row_idx, value in zip(judge_row_index, per_record):
        per_row.setdefault(row_idx, []).append(value)
    for row_idx, values in per_row.items():
        rows[row_idx]["judged_quality"] = round(mean(values), 6)
    return round(mean(per_record), 6), bootstrap_mean_ci(per_record)


def _run_judged_quality(
    scored: _ScoredTextAnalysisDocs,
    config: _JudgeConfig,
) -> _JudgeQualityResult:
    outcome = run_gated_judge(
        scored.judge_records,
        judge_model=config.model,
        judge_rho=config.rho,
        threshold=config.threshold,
        scorer=config.scorer,
        base_url=config.base_url,
    )
    if outcome.trusted and outcome.scores:
        quality, quality_ci = _attach_judged_quality(
            scored.rows, outcome.scores, scored.judge_row_index
        )
        return _JudgeQualityResult(outcome=outcome, value=quality, ci=quality_ci)
    if config.model is not None:
        _LOG.info(
            "[text-analysis] judge demoted (%s); objective recovery ranks alone", outcome.reason
        )
    return _JudgeQualityResult(outcome=outcome, value=None, ci=None)


def _text_analysis_metrics(
    result: ModelResult, reliability: float, tokens_per_s: float
) -> RunMetrics:
    return {
        "objective_score": result.objective_score,
        "reliability": reliability,
        "tokens_per_s": tokens_per_s,
    }


def _text_analysis_config(request: _TextAnalysisPersistInput) -> dict[str, object]:
    return {
        "model": request.model,
        "backend": request.backend,
        "tier": TIER_TEXT_ANALYSIS,
        "category": "text_analysis",
        "bundle": str(request.bundle),
        "synthetic": request.synthetic,
        "n_docs": request.n_docs,
        "judge_trusted": request.judge_result.outcome.trusted,
        "judged_quality": request.judge_result.value,  # gated diagnostic, NOT the headline
        "judged_quality_ci": list(request.judge_result.ci) if request.judge_result.ci else None,
        "judge_diagnostics": request.judge_result.outcome.diagnostics,
        **request.verification_cfg,
    }


def _text_analysis_judge_status(
    judge_model: str | None,
    judge_rho: float | None,
    judge_threshold: float,
    outcome: JudgeOutcome,
) -> JudgeStatus | None:
    if judge_model is None:
        return None
    return {
        "calibration_rho": judge_rho,
        "threshold": judge_threshold,
        "trusted": outcome.trusted,
        "model": judge_model,
        "metrics": ["judged_quality"],
        "diagnostics": outcome.diagnostics,
    }


def _persist_text_analysis_run(request: _TextAnalysisPersistInput) -> RunPaths | None:
    if request.data_dir is None:
        return None
    paths = persist_category_run(
        method=METHOD,
        data_dir=request.data_dir,
        run_name=request.run_name,
        config=_text_analysis_config(request),
        metrics=_text_analysis_metrics(request.result, request.reliability, request.tokens_per_s),
        case_rows=request.rows,
        judge=_text_analysis_judge_status(
            request.judge_config.model,
            request.judge_config.rho,
            request.judge_config.threshold,
            request.judge_result.outcome,
        ),
        mirror=request.mirror,
    )
    _LOG.info(
        "[text-analysis] %s scored %d docs (objective=%.3f, judged-quality=%s) -> %s",
        request.model,
        request.n_docs,
        request.result.objective_score,
        f"{request.judge_result.value:.3f}" if request.judge_result.value is not None else "n/a",
        paths["manifest"],
    )
    return paths


def run_text_analysis(
    bundle: Path | str,
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    similarity: ta.Similarity | None = None,
    judge_model: str | None = None,
    judge_rho: float | None = None,
    judge_threshold: float = DEFAULT_THRESHOLD,
    judge_scorer: JudgeScorer | None = None,
    judge_base_url: str | None = None,
    data_dir: Path | str | None = None,
    run_name: str = "text-analysis",
    limit: int | None = None,
    synthetic: bool = True,
    persist: bool = True,
    mirror: Mirror | None = None,
    data_verified: bool = False,
    verification_ref: str | None = None,
    meter: ThroughputMeter | None = None,
) -> TextAnalysisRun:
    """Score one model over a text-analysis bundle and return its board under TIER_TEXT_ANALYSIS.

    Objective planted-label recovery is the headline. `long_doc` labels are answered through the
    map-reduce template (`run_map_reduce_text`). When a judge is configured AND trusted
    (`judge_rho >= judge_threshold`), an opt-in JUDGED-QUALITY signal over the free-form sub-tasks
    (narrative / insight / long_doc) is recorded ALONGSIDE (per-doc + mean + CI), never folded into
    the objective headline; otherwise the judge is demoted. `synthetic` flags planted vs real corpus
    so the two are never merged. `judge_scorer` / `similarity` are injectable for tests. A `meter`
    (populated by the endpoint `complete`) supplies the run's real generation tok/s.
    """
    verification_cfg = verified_data_config(
        data_verified=data_verified, verification_ref=verification_ref
    )
    labels_by_doc = load_planted_by_doc(bundle)
    docs = _load_corpus_docs(bundle)
    doc_ids = _matching_doc_ids(bundle, labels_by_doc, docs, limit)
    similarity_fn = similarity if similarity is not None else ta.embedder_similarity()
    scored_docs = _score_doc_batch(doc_ids, labels_by_doc, docs, complete, similarity_fn)
    judge_config = _JudgeConfig(
        model=judge_model,
        rho=judge_rho,
        threshold=judge_threshold,
        scorer=judge_scorer,
        base_url=judge_base_url,
    )
    judge_result = _run_judged_quality(scored_docs, judge_config)

    reliability = scored_docs.n_ok / len(scored_docs.doc_ids)
    tokens_per_s = meter.tokens_per_s if meter is not None else 0.0
    result = category_result(
        model=model,
        backend=backend,
        tier=TIER_TEXT_ANALYSIS,
        case_objectives=scored_docs.case_objectives,
        reliability=reliability,
        tokens_per_s=tokens_per_s,
    )
    board, table = render_board([result])
    paths = (
        _persist_text_analysis_run(
            _TextAnalysisPersistInput(
                data_dir=data_dir,
                run_name=run_name,
                model=model,
                backend=backend,
                bundle=bundle,
                synthetic=synthetic,
                n_docs=len(scored_docs.doc_ids),
                result=result,
                reliability=reliability,
                rows=scored_docs.rows,
                judge_result=judge_result,
                judge_config=judge_config,
                verification_cfg=verification_cfg,
                tokens_per_s=tokens_per_s,
                mirror=mirror,
            )
        )
        if persist
        else None
    )
    return TextAnalysisRun(
        result=result,
        rows=scored_docs.rows,
        board=board,
        table=table,
        paths=paths,
        judged_quality=judge_result.value,
        judged_quality_ci=judge_result.ci,
        judge_trusted=judge_result.outcome.trusted,
        judge_reason=judge_result.outcome.reason,
    )
