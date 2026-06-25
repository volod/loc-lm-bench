"""M5.0/M5.4 scored text-analysis runner -- objective planted-label recovery.

Drives a candidate over a synthetic text-analysis bundle (the planter's `corpus/` docs plus
`text_analysis_labels.jsonl` of `PlantedLabelRecord`s), asks it to extract the per-sub-task
elements present in each document, scores recovery with the MH.2 matching engine
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
    category_result,
    mean,
    persist_category_run,
    render_board,
    run_gated_judge,
)
from llb.contracts import (
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
from llb.rag.chunking import iter_docs
from llb.scoring import text_analysis as ta
from llb.scoring.aggregate import TIER_TEXT_ANALYSIS, ModelResult, bootstrap_mean_ci

_LOG = logging.getLogger(__name__)

METHOD = "text-analysis"
TEXT_ANALYSIS_LABELS = "text_analysis_labels.jsonl"

# The judged free-form sub-tasks the gated judge owns (objective match is only a floor for them).
_JUDGED_EXTRACT_KINDS = (ta.NARRATIVE, ta.INSIGHT)
# A UA intent framing per judged sub-task: faithfulness scores grounding in the doc, answer-
# relevancy scores whether the free-form output addresses the sub-task.
_JUDGE_INTENT: dict[str, str] = {
    ta.NARRATIVE: "Стисло й точно виклади загальну оповідь документа, не додаючи зайвого.",
    ta.INSIGHT: "Сформулюй обґрунтовані висновки, що випливають з документа.",
    ta.LONG_DOC: "Дай повну відповідь на питання, спираючись лише на документ.",
}
_DEFAULT_LONG_DOC_QUESTION = "Про що цей документ і які його ключові висновки?"

# UA instruction phrasing per sub-task kind (the candidate-facing extraction ask).
_KIND_UA: dict[str, str] = {
    ta.KEY_FACT: "ключові факти",
    ta.ENTITY: "іменовані сутності (особи, організації, місця)",
    ta.TOPIC: "теми документа",
    ta.TREND: "тенденції (вкажи напрям: зростання, спад або стабільність)",
    ta.RISK: "ризики або проблеми",
    ta.DECISION: "рішення або дії",
    ta.CONTRADICTION: "внутрішні суперечності",
    ta.NARRATIVE: "стислий виклад загальної оповіді",
    ta.INSIGHT: "висновки, прямо не зазначені в тексті",
}


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


def analysis_prompt(doc_id: str, text: str, kinds: Sequence[str]) -> str:
    """Ask the candidate to extract each requested sub-task's elements as a JSON object keyed by
    the sub-task name, value = a list of short Ukrainian strings."""
    bullets = "\n".join(f"- {kind}: {_KIND_UA.get(kind, kind)}" for kind in kinds)
    keys = ", ".join(kinds)
    return (
        "Ти аналізуєш україномовний документ. Витягни перелічені нижче елементи.\n"
        f"{bullets}\n\n"
        "Поверни ЛИШЕ JSON-об'єкт, де кожен ключ -- назва категорії (англійською, точно як "
        f"нижче), а значення -- масив коротких рядків українською. Категорії: {keys}.\n"
        "Якщо для категорії нічого немає, поверни порожній масив.\n\n"
        f"Документ [{doc_id}]:\n{text}\n"
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
    run_name: str = "m5-text-analysis",
    limit: int | None = None,
    synthetic: bool = True,
    persist: bool = True,
    mirror: Mirror | None = None,
) -> TextAnalysisRun:
    """Score one model over a text-analysis bundle and return its board under TIER_TEXT_ANALYSIS.

    Objective planted-label recovery is the headline. `long_doc` labels are answered through the
    map-reduce template (`run_map_reduce_text`). When a judge is configured AND trusted
    (`judge_rho >= judge_threshold`), an opt-in JUDGED-QUALITY signal over the free-form sub-tasks
    (narrative / insight / long_doc) is recorded ALONGSIDE (per-doc + mean + CI), never folded into
    the objective headline; otherwise the judge is demoted. `synthetic` flags planted vs real corpus
    so the two are never merged. `judge_scorer` / `similarity` are injectable for tests.
    """
    labels_by_doc = load_planted_by_doc(bundle)
    # `iter_docs` ids are corpus-relative paths WITH the extension (e.g. "synth-000.md"); planted
    # labels key by the planter's bare doc id ("synth-000"). Index both so either form resolves.
    docs: dict[str, str] = {}
    for rel, text in iter_docs(Path(bundle) / "corpus"):
        docs[rel] = text
        docs.setdefault(Path(rel).stem, text)
    if similarity is None:
        similarity = ta.embedder_similarity()

    doc_ids = sorted(doc_id for doc_id in labels_by_doc if doc_id in docs)
    if not doc_ids:
        raise SystemExit(
            f"no text-analysis documents with planted labels under {bundle} "
            f"(need {TEXT_ANALYSIS_LABELS} + a matching corpus/)"
        )
    if limit is not None:
        doc_ids = doc_ids[:limit]

    rows: list[TextAnalysisCaseRow] = []
    case_objectives: list[float] = []
    judge_records: list[JudgeInputRecord] = []
    judge_row_index: list[int] = []  # the row each judge record's quality attaches to
    n_ok = 0
    for doc_id in doc_ids:
        labels = labels_by_doc[doc_id]
        # long_doc is answered via map-reduce, not the single extraction prompt.
        extract_kinds = sorted({label.kind for label in labels if label.kind != ta.LONG_DOC})
        raw = (
            complete(analysis_prompt(doc_id, docs[doc_id], extract_kinds)) if extract_kinds else ""
        )
        predictions: dict[str, list[str]]
        if extract_kinds and not raw.strip():
            status, predictions = EMPTY, {}
        elif not extract_kinds:
            status, predictions = OK, {}
        else:
            try:
                predictions = parse_predictions(raw, extract_kinds)
                status = OK
            except (ValueError, json.JSONDecodeError):
                status, predictions = MALFORMED, {}
        scored = ta.score_document(predictions, labels, similarity)
        if status == OK:
            n_ok += 1
        case_objectives.append(float(scored["objective_score"]))
        row = _case_row(doc_id, status, scored, len(labels))

        # long-doc comprehension answer via the map-reduce template (the judged headline).
        question = long_doc_question(labels)
        if question is not None:
            answer = run_map_reduce_text(complete, question, docs[doc_id])
            row["long_doc_answer"] = answer[:280]
            judge_records.append(
                {"question": question, "answer": answer, "contexts": [docs[doc_id]]}
            )
            judge_row_index.append(len(rows))
        # narrative / insight free-form judged sub-tasks present on this doc.
        for kind in _JUDGED_EXTRACT_KINDS:
            answer = _judged_answer(predictions, kind)
            if answer:
                judge_records.append(
                    {
                        "question": _JUDGE_INTENT[kind],
                        "answer": answer,
                        "contexts": [docs[doc_id]],
                    }
                )
                judge_row_index.append(len(rows))
        rows.append(row)

    # Opt-in, gated judged-quality signal (objective recovery stays the headline).
    outcome = run_gated_judge(
        judge_records,
        judge_model=judge_model,
        judge_rho=judge_rho,
        threshold=judge_threshold,
        scorer=judge_scorer,
        base_url=judge_base_url,
    )
    quality: float | None = None
    quality_ci: tuple[float, float] | None = None
    if outcome.trusted and outcome.scores:
        per_record = [judged_quality(s) for s in outcome.scores]
        per_row: dict[int, list[float]] = {}
        for row_idx, value in zip(judge_row_index, per_record):
            per_row.setdefault(row_idx, []).append(value)
        for row_idx, values in per_row.items():
            rows[row_idx]["judged_quality"] = round(mean(values), 6)
        quality = round(mean(per_record), 6)
        quality_ci = bootstrap_mean_ci(per_record)
    elif judge_model is not None:
        _LOG.info(
            "[text-analysis] judge demoted (%s); objective recovery ranks alone", outcome.reason
        )

    reliability = n_ok / len(doc_ids) if doc_ids else 0.0
    result = category_result(
        model=model,
        backend=backend,
        tier=TIER_TEXT_ANALYSIS,
        case_objectives=case_objectives,
        reliability=reliability,
    )
    board, table = render_board([result])

    paths: RunPaths | None = None
    if persist and data_dir is not None:
        metrics: RunMetrics = {
            "objective_score": result.objective_score,
            "reliability": reliability,
            "tokens_per_s": 0.0,
        }
        config = {
            "model": model,
            "backend": backend,
            "tier": TIER_TEXT_ANALYSIS,
            "category": "text_analysis",
            "bundle": str(bundle),
            "synthetic": synthetic,
            "n_docs": len(doc_ids),
            "judge_trusted": outcome.trusted,
            "judged_quality": quality,  # gated diagnostic, NOT the headline
            "judged_quality_ci": list(quality_ci) if quality_ci else None,
        }
        judge_status: JudgeStatus | None = None
        if judge_model is not None:
            judge_status = {
                "calibration_rho": judge_rho,
                "threshold": judge_threshold,
                "trusted": outcome.trusted,
                "model": judge_model,
                "metrics": ["judged_quality"],
            }
        paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=run_name,
            config=config,
            metrics=metrics,
            case_rows=rows,
            judge=judge_status,
            mirror=mirror,
        )
        _LOG.info(
            "[text-analysis] %s scored %d docs (objective=%.3f, judged-quality=%s) -> %s",
            model,
            len(doc_ids),
            result.objective_score,
            f"{quality:.3f}" if quality is not None else "n/a",
            paths["manifest"],
        )
    return TextAnalysisRun(
        result=result,
        rows=rows,
        board=board,
        table=table,
        paths=paths,
        judged_quality=quality,
        judged_quality_ci=quality_ci,
        judge_trusted=outcome.trusted,
        judge_reason=outcome.reason,
    )
