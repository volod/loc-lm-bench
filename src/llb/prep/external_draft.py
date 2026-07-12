"""Import externally drafted corpus-grounded goldsets (external-draft contract Artifact B).

An operator who drafted test data OUTSIDE this repo with an AI provider service (Claude Projects,
NotebookLM, ChatGPT Projects) on OPEN corpus data imports the grounded-JSONL export into a standard,
unverified draft bundle with one command. Every `quote` is re-grounded against the LOCAL corpus (so a
label can never point at text that is not there), exact `source_spans` are computed, the external
service/model/classification is recorded, and `question_type`/`difficulty` labels are carried in
item provenance -- so externally drafted goldsets flow through the same cross-check + human
verification gate as local drafts.

Egress gate: the required `external_provenance.json` sidecar must be present and declare
`data_classification: "open"`; anything else is a hard refusal that writes NO bundle (uploading a
corpus to a provider publishes it -- restricted data never leaves the box). The sidecar records what
was uploaded and where. Reuses `frontier.ground_span`, the lenient `curation` loaders,
`goldset.splits.assign_splits`, and `goldset.validate.validate_items`; no network.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from llb.core.contracts import ValidationReport
from llb.goldset.schema import GoldItem, Provenance, SourceSpan, Split, dump_goldset
from llb.goldset.splits import assign_splits
from llb.goldset.validate import validate_items
from llb.prep.curation.common import load_corpus_texts, load_json_documents, load_jsonl_rows
from llb.prep.frontier import ground_span
from llb.prep.ontology.constants import DEFAULT_QUESTION_TYPE, QUESTION_TYPES
from llb.prep.ontology.coverage import classify_difficulty
from llb.prep.ontology.models import ItemLabels
from llb.prep.ontology.needles import NeedleRetriever, annotate_needle_retrieval
from llb.prep.ontology.question_types import classify_question_type

_LOG = logging.getLogger(__name__)

PROVENANCE_EXTERNAL: Provenance = "frontier-drafted"
DATA_CLASSIFICATION_OPEN = "open"
DIFFICULTIES = ("easy", "medium", "hard")

SIDECAR_FILENAME = "external_provenance.json"
GOLDSET_FILENAME = "goldset.jsonl"
CORPUS_DIRNAME = "corpus"
PROVENANCE_FILENAME = "provenance.json"
ITEM_PROVENANCE_FILENAME = "item_provenance.jsonl"
IMPORT_REPORT_FILENAME = "import_report.json"


@dataclass
class ImportReport:
    """Counts and per-row drop/repair reasons for one import (written beside the bundle)."""

    loaded: int = 0
    kept: int = 0
    dropped: list[dict[str, str]] = field(default_factory=list)
    repaired: list[dict[str, str]] = field(default_factory=list)

    def drop(self, item_id: str, reason: str) -> None:
        self.dropped.append({"id": item_id, "reason": reason})

    def repair(self, item_id: str, what: str) -> None:
        self.repaired.append({"id": item_id, "repair": what})

    def to_dict(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "kept": self.kept,
            "counts": {"dropped": len(self.dropped), "repaired": len(self.repaired)},
            "dropped": self.dropped,
            "repaired": self.repaired,
        }


def load_sidecar(sidecar: Path) -> dict[str, Any]:
    """Load + enforce the data-classification sidecar; a missing or non-open sidecar aborts.

    Raises SystemExit (no bundle written) so restricted/private material can never be imported as
    if it were cleared for third-party processing.
    """
    if not Path(sidecar).is_file():
        raise SystemExit(
            f"[import-external-draft] required sidecar {sidecar} is absent; an external draft "
            f"cannot be imported without its {SIDECAR_FILENAME} data-classification record."
        )
    try:
        data = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[import-external-draft] cannot read sidecar {sidecar}: {exc}") from exc
    if not isinstance(data, dict) or data.get("data_classification") != DATA_CLASSIFICATION_OPEN:
        raise SystemExit(
            "[import-external-draft] refused: the sidecar must declare "
            f'data_classification: "{DATA_CLASSIFICATION_OPEN}". Uploading a corpus to an external '
            "AI service publishes it; only explicitly open corpora may be imported."
        )
    return data


def _labels(row: dict[str, Any], question: str, reference: str, span_text: str) -> ItemLabels:
    """Question-type + difficulty for a row: honor valid inline labels, else classify locally."""
    qtype = str(row.get("question_type") or "").strip()
    if qtype not in QUESTION_TYPES:
        qtype = classify_question_type(question, reference) or DEFAULT_QUESTION_TYPE
    difficulty = str(row.get("difficulty") or "").strip()
    if difficulty not in DIFFICULTIES:
        difficulty = classify_difficulty(len(span_text), rare=False)
    return ItemLabels(question_type=qtype, difficulty=difficulty)


def _row_to_item(
    row: dict[str, Any], corpus_texts: dict[str, str], index: int, report: ImportReport
) -> tuple[GoldItem, ItemLabels] | None:
    """Re-ground one Artifact B row into a GoldItem + labels, or None (dropped + reported)."""
    item_id = str(row.get("id") or f"ext-{index:04d}")
    question = str(row.get("question") or "").strip()
    quote = str(row.get("quote") or row.get("reference_answer") or "").strip()
    doc_id = str(row.get("source_doc_id") or "").strip()
    if not question or not quote or not doc_id:
        report.drop(item_id, "missing question, quote, or source_doc_id")
        return None
    text = corpus_texts.get(doc_id)
    if text is None:
        report.drop(item_id, f"source_doc_id {doc_id} is not in the corpus")
        return None
    grounded = ground_span(text, quote)  # exact, then normalized-but-exact fallback
    if grounded is None:
        report.drop(item_id, f"quote is not a verbatim substring of {doc_id}")
        return None
    start, exact = grounded
    if exact != quote:
        report.repair(item_id, "quote re-grounded to exact corpus text")
    reference = str(row.get("reference_answer") or "").strip() or exact
    labels = _labels(row, question, reference, exact)
    item = GoldItem(
        id=item_id,
        lang=str(row.get("lang") or "uk"),
        question=question,
        reference_answer=reference,
        source_doc_id=doc_id,
        source_spans=[
            SourceSpan(doc_id=doc_id, char_start=start, char_end=start + len(exact), text=exact)
        ],
        provenance=PROVENANCE_EXTERNAL,
        verified=False,
        split="final",
    )
    return item, labels


def _label_distribution(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


@dataclass
class ImportResult:
    out_dir: Path
    items: list[GoldItem]
    item_labels: dict[str, ItemLabels]
    report: ImportReport
    validation: ValidationReport


def _write_bundle(
    out_dir: Path,
    items: list[GoldItem],
    item_labels: dict[str, ItemLabels],
    corpus_texts: dict[str, str],
    sidecar: dict[str, Any],
    report: ImportReport,
    *,
    retrieval_ranks: dict[str, int | None] | None = None,
    retrieval_k: int | None = None,
    needle_report: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_goldset(items, out_dir / GOLDSET_FILENAME)

    # Verbatim corpus copy: write only the referenced docs, byte-identical to the text the spans
    # were grounded against, so validate-goldset offset round-trips exactly.
    corpus_dir = out_dir / CORPUS_DIRNAME
    for doc_id in sorted({it.source_doc_id for it in items}):
        dest = corpus_dir / doc_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(corpus_texts[doc_id], encoding="utf-8")

    # Item provenance: question_type / difficulty per item (NOT part of the GoldItem schema);
    # `retrieval_rank` is additive and present only when an index was given (needle parity with
    # the local ontology lane -- the verify worksheet reads it from this file).
    with (out_dir / ITEM_PROVENANCE_FILENAME).open("w", encoding="utf-8") as fh:
        for item in items:
            label = item_labels[item.id]
            row: dict[str, Any] = {
                "id": item.id,
                "question_type": label.question_type,
                "difficulty": label.difficulty,
            }
            if retrieval_ranks is not None:
                row["retrieval_rank"] = retrieval_ranks.get(item.id)
                row["retrieval_k"] = retrieval_k
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    provenance = {
        "kind": "external-draft-import",
        "provenance": PROVENANCE_EXTERNAL,
        "synthetic": False,
        "verified": False,
        "service": sidecar.get("service"),
        "service_model": sidecar.get("service_model"),
        "export_date": sidecar.get("export_date"),
        "data_classification": sidecar.get("data_classification"),
        "operator": sidecar.get("operator"),
        "n_items": len(items),
        "question_type_distribution": _label_distribution(
            [item_labels[it.id].question_type for it in items]
        ),
        "difficulty_distribution": _label_distribution(
            [item_labels[it.id].difficulty for it in items]
        ),
        "import_report": report.to_dict(),
    }
    if needle_report is not None:
        provenance["needle_retrieval"] = needle_report
    (out_dir / PROVENANCE_FILENAME).write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / IMPORT_REPORT_FILENAME).write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_retriever(index_dir: Path | str | None) -> NeedleRetriever | None:
    if index_dir is None:
        return None
    from llb.rag.store import RagStore

    return RagStore.load(index_dir)


def _build_items(
    rows: list[Any], corpus_texts: dict[str, str], report: ImportReport
) -> tuple[list[GoldItem], dict[str, ItemLabels]]:
    """Ground every artifact row; drop (and count) rows that fail verbatim grounding."""
    items: list[GoldItem] = []
    item_labels: dict[str, ItemLabels] = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            report.drop(f"row-{i}", "row is not a JSON object")
            continue
        report.loaded += 1
        built = _row_to_item(row, corpus_texts, i, report)
        if built is None:
            continue
        item, label = built
        items.append(item)
        item_labels[item.id] = label
    return items, item_labels


def _annotate_needles(
    items: list[GoldItem],
    item_labels: dict[str, ItemLabels],
    retriever: NeedleRetriever | None,
    retrieval_k: int,
    drop_nonretrievable: bool,
    report: ImportReport,
) -> tuple[
    list[GoldItem], dict[str, ItemLabels], dict[str, int | None] | None, dict[str, Any] | None
]:
    """Needle-parity lane: annotate gold-span retrieval ranks, optionally dropping rank-less items."""
    if retriever is None or not items:
        return items, item_labels, None, None
    needle_rows, needle_report = annotate_needle_retrieval(
        items, retriever, k=retrieval_k, drop_nonretrievable=drop_nonretrievable
    )
    retrieval_ranks = {
        str(row["id"]): cast("int | None", row["retrieval_rank"]) for row in needle_rows
    }
    if drop_nonretrievable:
        for item in items:
            if item.id not in retrieval_ranks:
                report.drop(item.id, f"gold span not retrieved within top-{retrieval_k}")
        items = [item for item in items if item.id in retrieval_ranks]
        item_labels = {item.id: item_labels[item.id] for item in items}
    return items, item_labels, retrieval_ranks, needle_report


def import_external_draft(
    artifact: Path,
    corpus_root: Path,
    sidecar: Path,
    out_dir: Path,
    *,
    seed: int = 13,
    retrieval_index_dir: Path | str | None = None,
    retrieval_k: int = 10,
    drop_nonretrievable_needles: bool = False,
    retriever: NeedleRetriever | None = None,
) -> ImportResult:
    """Import a grounded-JSONL Artifact B export into a canonical draft bundle.

    Enforces the open-data sidecar first (aborts before writing anything on a missing/non-open
    sidecar), re-grounds each quote (dropping + counting non-verbatim rows), assigns splits, and
    writes goldset.jsonl + verbatim corpus/ + provenance.json + item_provenance.jsonl + report.

    `retrieval_index_dir` (external-import-needle-parity) annotates each imported item with its
    gold-span retrieval rank against the given full-corpus index -- the same needle signal the
    local ontology lane records -- into `item_provenance.jsonl`, where the verify worksheet
    already reads it. `drop_nonretrievable_needles` additionally drops rank-less items (explicit
    opt-in only). Without an index the lane is an exact no-op. `retriever` is injectable for
    tests.
    """
    sidecar_data = load_sidecar(sidecar)  # egress gate BEFORE any bundle write
    if drop_nonretrievable_needles and retrieval_index_dir is None and retriever is None:
        raise SystemExit(
            "[import-external-draft] --drop-nonretrievable-needles requires --retrieval-index-dir"
        )
    corpus_texts = load_corpus_texts(Path(corpus_root))
    rows = load_jsonl_rows(load_json_documents(Path(artifact)))

    report = ImportReport()
    items, item_labels = _build_items(rows, corpus_texts, report)
    resolved_retriever = (
        retriever if retriever is not None else _load_retriever(retrieval_index_dir)
    )
    items, item_labels, retrieval_ranks, needle_report = _annotate_needles(
        items, item_labels, resolved_retriever, retrieval_k, drop_nonretrievable_needles, report
    )

    if not items:
        raise SystemExit(
            "[import-external-draft] no verbatim-grounded items to import "
            f"({len(report.dropped)} rows dropped); no bundle written."
        )

    split_map = assign_splits([it.id for it in items], seed=seed)
    for item in items:
        item.split = cast(Split, split_map[item.id])
    report.kept = len(items)

    out_dir = Path(out_dir)
    _write_bundle(
        out_dir,
        items,
        item_labels,
        corpus_texts,
        sidecar_data,
        report,
        retrieval_ranks=retrieval_ranks,
        retrieval_k=retrieval_k if retrieval_ranks is not None else None,
        needle_report=needle_report,
    )
    validation = validate_items(items, out_dir / CORPUS_DIRNAME)
    _LOG.info(
        "[import-external-draft] imported %d items (verified=false, %d dropped) -> %s",
        len(items),
        len(report.dropped),
        out_dir,
    )
    return ImportResult(out_dir, items, item_labels, report, validation)
