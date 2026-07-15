"""Planted-label and corpus-document loading and alignment."""

import json
from pathlib import Path

from llb.bench.text_analysis.constants import TEXT_ANALYSIS_LABELS
from llb.core.contracts.benchmarks import PlantedLabelRecord
from llb.rag.chunking.corpus import iter_docs
from llb.scoring import text_analysis_labels as labels


def load_planted_by_doc(bundle: Path | str) -> dict[str, list[labels.PlantedLabel]]:
    path = Path(bundle) / TEXT_ANALYSIS_LABELS
    by_doc: dict[str, list[PlantedLabelRecord]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record: PlantedLabelRecord = json.loads(line)
        by_doc.setdefault(str(record.get("doc_id", "")), []).append(record)
    return {doc_id: labels.load_planted_labels(records) for doc_id, records in by_doc.items()}


def load_corpus_docs(bundle: Path | str) -> dict[str, str]:
    docs: dict[str, str] = {}
    for relative_path, text in iter_docs(Path(bundle) / "corpus"):
        docs[relative_path] = text
        docs.setdefault(Path(relative_path).stem, text)
    return docs


def matching_doc_ids(
    bundle: Path | str,
    labels_by_doc: dict[str, list[labels.PlantedLabel]],
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
