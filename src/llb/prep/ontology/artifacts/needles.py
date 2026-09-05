"""Citation-valid needle selection, retrieval annotation, and label summaries."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from llb.core.contracts.data_prep.goldset import NeedleItemRecord
from llb.goldset.schema import GoldItem
from llb.prep.ontology.artifacts.citations import ratio, span_page_refs
from llb.prep.ontology.constants import DEFAULT_QUESTION_TYPE
from llb.prep.ontology.models import ItemLabels
from llb.prep.ontology.drafting.needles import NeedleRetriever, annotate_needle_retrieval

NEEDLE_ITEM_SCHEMA_ID = "llb.needle-item"
NEEDLE_ITEM_SCHEMA_VERSION = "1.0.0"
_GOLD_IDENTITY = ("schema_id", "schema_version")


def citation_valid_items(items: list[GoldItem], index: dict[str, dict[str, Any]]) -> list[GoldItem]:
    return [
        item for item in items if all(span_page_refs(span, index) for span in item.source_spans)
    ]


def needle_rows_and_report(
    needles: list[GoldItem],
    *,
    retrieval_store: NeedleRetriever | None,
    retrieval_k: int,
    drop_nonretrievable_needles: bool,
    item_labels: dict[str, ItemLabels] | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if retrieval_store is None:
        rows = [cast(dict[str, object], item.model_dump()) for item in needles]
        report: dict[str, object] = {"enabled": False}
    else:
        rows, report = annotate_needle_retrieval(
            needles,
            retrieval_store,
            k=retrieval_k,
            drop_nonretrievable=drop_nonretrievable_needles,
        )
    if item_labels:
        for row in rows:
            label = item_labels.get(str(row.get("id")))
            if label is not None:
                row["question_type"] = label.question_type
                row["difficulty"] = label.difficulty
    return [_needle_row(row) for row in rows], report


def _needle_row(row: dict[str, object]) -> dict[str, object]:
    """One needle row at its registered contract, replacing the gold row's own identity."""
    fields = {key: value for key, value in row.items() if key not in _GOLD_IDENTITY}
    record = NeedleItemRecord.model_validate(
        {
            "schema_id": NEEDLE_ITEM_SCHEMA_ID,
            "schema_version": NEEDLE_ITEM_SCHEMA_VERSION,
            **fields,
        }
    )
    return cast(dict[str, object], record.model_dump())


def load_needle_items(path: Path) -> list[NeedleItemRecord]:
    """Read a needle sidecar at the current contract, whatever version it was written at."""
    from llb.artifacts.default_registry import DEFAULT_REGISTRY

    rows: list[NeedleItemRecord] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = DEFAULT_REGISTRY.read_as(
            NEEDLE_ITEM_SCHEMA_ID, json.loads(line), source=f"{path}:{line_no}"
        )
        rows.append(cast(NeedleItemRecord, record))
    return rows


def label_distributions(
    items: list[GoldItem], item_labels: dict[str, ItemLabels] | None
) -> tuple[dict[str, int], dict[str, int]]:
    by_type: dict[str, int] = defaultdict(int)
    by_difficulty: dict[str, int] = defaultdict(int)
    labels = item_labels or {}
    for item in items:
        label = labels.get(item.id)
        by_type[label.question_type if label else DEFAULT_QUESTION_TYPE] += 1
        by_difficulty[label.difficulty if label else "medium"] += 1
    return dict(sorted(by_type.items())), dict(sorted(by_difficulty.items()))


def retrieval_fraction_by_type(
    needle_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"needles": 0, "retrievable": 0})
    for row in needle_rows:
        question_type = str(row.get("question_type") or DEFAULT_QUESTION_TYPE)
        groups[question_type]["needles"] += 1
        if row.get("retrieval_rank") is not None:
            groups[question_type]["retrievable"] += 1
    return {
        question_type: {
            "needles": group["needles"],
            "retrievable": group["retrievable"],
            "retrievable_fraction": ratio(group["retrievable"], group["needles"]),
        }
        for question_type, group in sorted(groups.items())
    }


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
