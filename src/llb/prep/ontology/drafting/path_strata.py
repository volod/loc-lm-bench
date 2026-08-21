"""Deterministic relation/document-stratified selection of multi-hop draft paths."""

from collections import Counter
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from llb.prep.ontology.models import MultiHopSeed

DOCUMENT_MODE_SAME = "same-document"
DOCUMENT_MODE_CROSS = "cross-document"
DOCUMENT_MODES = (DOCUMENT_MODE_SAME, DOCUMENT_MODE_CROSS)


@dataclass(frozen=True)
class PathStratumTargets:
    """Minimum coverage requested on each path-stratification axis."""

    relation_pair: int
    document_mode: int
    source_document: int


@dataclass(frozen=True)
class _PathDescriptor:
    index: int
    seed: MultiHopSeed
    relation_pair: str
    document_mode: str
    source_documents: tuple[str, ...]


def _describe(index: int, seed: MultiHopSeed) -> _PathDescriptor:
    relations = tuple(step.relation for step in seed.steps)
    documents = tuple(sorted({step.evidence.doc_id for step in seed.steps}))
    mode = DOCUMENT_MODE_SAME if len(documents) == 1 else DOCUMENT_MODE_CROSS
    return _PathDescriptor(
        index=index,
        seed=seed,
        relation_pair=" -> ".join(relations),
        document_mode=mode,
        source_documents=documents,
    )


def _deficit_score(
    path: _PathDescriptor,
    counts: dict[str, Counter[str]],
    targets: PathStratumTargets,
) -> float:
    relation = float(counts["relation_pairs"][path.relation_pair] < targets.relation_pair)
    mode = float(counts["document_modes"][path.document_mode] < targets.document_mode)
    documents = [
        float(counts["source_documents"][doc_id] < targets.source_document)
        for doc_id in path.source_documents
    ]
    document = sum(documents)
    return relation + mode + document


def _balance_score(
    path: _PathDescriptor, counts: dict[str, Counter[str]]
) -> tuple[float, int, int, int]:
    document_counts = [counts["source_documents"][doc_id] for doc_id in path.source_documents]
    document_mean = sum(document_counts) / len(document_counts) if document_counts else 0.0
    return (
        counts["relation_pairs"][path.relation_pair]
        + counts["document_modes"][path.document_mode]
        + document_mean,
        counts["relation_pairs"][path.relation_pair],
        counts["document_modes"][path.document_mode],
        path.index,
    )


def _choose_next(
    remaining: list[_PathDescriptor],
    counts: dict[str, Counter[str]],
    targets: PathStratumTargets,
) -> _PathDescriptor:
    return min(
        remaining,
        key=lambda path: (
            -_deficit_score(path, counts, targets),
            *_balance_score(path, counts),
        ),
    )


def _record_selection(path: _PathDescriptor, counts: dict[str, Counter[str]]) -> None:
    counts["relation_pairs"][path.relation_pair] += 1
    counts["document_modes"][path.document_mode] += 1
    counts["source_documents"].update(path.source_documents)


def _stratum_rows(
    categories: Sequence[str],
    available: Counter[str],
    selected: Counter[str],
    target: int,
    *,
    budget_exhausted: bool,
) -> dict[str, dict[str, int | str]]:
    rows: dict[str, dict[str, int | str]] = {}
    for category in categories:
        selected_count = selected[category]
        available_count = available[category]
        if selected_count >= target:
            status = "covered"
        elif available_count <= selected_count:
            status = "exhausted"
        elif budget_exhausted:
            status = "budget-exhausted"
        else:
            status = "exhausted"
        rows[category] = {
            "target": target,
            "available": available_count,
            "selected": selected_count,
            "status": status,
        }
    return rows


def _all_covered_or_exhausted(report_axes: Sequence[Mapping[str, object]]) -> bool:
    return all(
        row.get("status") in {"covered", "exhausted"}
        for axis in report_axes
        for row in axis.values()
        if isinstance(row, dict)
    )


def select_stratified_paths(
    seeds: Sequence[MultiHopSeed],
    *,
    max_paths: int,
    targets: PathStratumTargets,
    source_documents: Sequence[str],
) -> tuple[list[MultiHopSeed], dict[str, object]]:
    """Allocate the path budget across relation, document-mode, and source-document strata."""
    descriptors = [_describe(index, seed) for index, seed in enumerate(seeds)]
    remaining = list(descriptors)
    selected: list[_PathDescriptor] = []
    counts = {
        "relation_pairs": Counter[str](),
        "document_modes": Counter[str](),
        "source_documents": Counter[str](),
    }
    while remaining and len(selected) < max_paths:
        chosen = _choose_next(remaining, counts, targets)
        selected.append(chosen)
        _record_selection(chosen, counts)
        remaining.remove(chosen)

    available = {
        "relation_pairs": Counter(path.relation_pair for path in descriptors),
        "document_modes": Counter(path.document_mode for path in descriptors),
        "source_documents": Counter(
            doc_id for path in descriptors for doc_id in path.source_documents
        ),
    }
    budget_exhausted = bool(remaining) and len(selected) >= max_paths
    relation_rows = _stratum_rows(
        sorted(available["relation_pairs"]),
        available["relation_pairs"],
        counts["relation_pairs"],
        targets.relation_pair,
        budget_exhausted=budget_exhausted,
    )
    mode_rows = _stratum_rows(
        DOCUMENT_MODES,
        available["document_modes"],
        counts["document_modes"],
        targets.document_mode,
        budget_exhausted=budget_exhausted,
    )
    document_rows = _stratum_rows(
        sorted(set(source_documents)),
        available["source_documents"],
        counts["source_documents"],
        targets.source_document,
        budget_exhausted=budget_exhausted,
    )
    report: dict[str, object] = {
        "kind": "multihop-path-strata",
        "strategy": "relation-document-stratified",
        "path_budget": max_paths,
        "candidate_paths": len(descriptors),
        "selected_paths": len(selected),
        "unfilled_path_budget": max(0, max_paths - len(selected)),
        "budget_exhausted": budget_exhausted,
        "targets": {
            "relation_pair": targets.relation_pair,
            "document_mode": targets.document_mode,
            "source_document": targets.source_document,
        },
        "relation_pairs": relation_rows,
        "document_modes": mode_rows,
        "source_documents": document_rows,
    }
    report["all_requested_covered_or_exhausted"] = _all_covered_or_exhausted(
        (relation_rows, mode_rows, document_rows)
    )
    return [path.seed for path in selected], report
