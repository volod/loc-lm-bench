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
from pathlib import Path
from typing import Any, cast

from llb.goldset.schema import GoldItem, Split, dump_goldset
from llb.goldset.splits import assign_splits
from llb.goldset.validate import validate_items
from llb.prep.curation.input import load_corpus_texts, load_json_documents, load_jsonl_rows
from llb.prep.ontology.models import ItemLabels
from llb.prep.ontology.needles import NeedleRetriever, annotate_needle_retrieval
from llb.prep.external_draft_schema import (
    CORPUS_DIRNAME,
    GOLDSET_FILENAME,
    IMPORT_REPORT_FILENAME,
    ITEM_PROVENANCE_FILENAME,
    ImportReport,
    ImportResult,
    PROVENANCE_EXTERNAL,
    PROVENANCE_FILENAME,
    _label_distribution,
    _row_to_item,
    load_sidecar,
)

_LOG = logging.getLogger(__name__)


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
