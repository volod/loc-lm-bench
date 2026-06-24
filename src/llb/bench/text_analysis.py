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
    LLMComplete,
    Mirror,
    category_result,
    persist_category_run,
    render_board,
)
from llb.contracts import BoardRow, RunMetrics, RunPaths, TextAnalysisCaseRow
from llb.eval.common import EMPTY, MALFORMED, OK
from llb.prep.frontier import parse_json_block
from llb.rag.chunking import iter_docs
from llb.scoring import text_analysis as ta
from llb.scoring.aggregate import TIER_TEXT_ANALYSIS, ModelResult

_LOG = logging.getLogger(__name__)

METHOD = "text-analysis"
TEXT_ANALYSIS_LABELS = "text_analysis_labels.jsonl"

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


def run_text_analysis(
    bundle: Path | str,
    *,
    model: str,
    backend: str,
    complete: LLMComplete,
    similarity: ta.Similarity | None = None,
    data_dir: Path | str | None = None,
    run_name: str = "m5-text-analysis",
    limit: int | None = None,
    synthetic: bool = True,
    persist: bool = True,
    mirror: Mirror | None = None,
) -> TextAnalysisRun:
    """Score one model over a text-analysis bundle and return its board under TIER_TEXT_ANALYSIS.

    `synthetic` flags whether the bundle is planted (vs a real corpus); it is recorded so the two
    are never merged in reporting. `persist=True` writes a run bundle under
    `$DATA_DIR/text-analysis/<ts>/` when `data_dir` is given.
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
    n_ok = 0
    for doc_id in doc_ids:
        labels = labels_by_doc[doc_id]
        kinds = sorted({label.kind for label in labels})
        raw = complete(analysis_prompt(doc_id, docs[doc_id], kinds))
        if not raw.strip():
            status, predictions = EMPTY, {}
        else:
            try:
                predictions = parse_predictions(raw, kinds)
                status = OK
            except (ValueError, json.JSONDecodeError):
                status, predictions = MALFORMED, {}
        scored = ta.score_document(predictions, labels, similarity)
        if status == OK:
            n_ok += 1
        case_objectives.append(float(scored["objective_score"]))
        rows.append(_case_row(doc_id, status, scored, len(labels)))

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
        }
        paths = persist_category_run(
            method=METHOD,
            data_dir=data_dir,
            run_name=run_name,
            config=config,
            metrics=metrics,
            case_rows=rows,
            mirror=mirror,
        )
        _LOG.info(
            "[text-analysis] %s scored %d docs (objective=%.3f) -> %s",
            model,
            len(doc_ids),
            result.objective_score,
            paths["manifest"],
        )
    return TextAnalysisRun(result=result, rows=rows, board=board, table=table, paths=paths)
