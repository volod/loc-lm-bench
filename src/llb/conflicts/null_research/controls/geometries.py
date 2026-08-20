"""Higher-capacity similarity geometries scored on exactly the same controls.

Mean centering removes ONE anisotropy direction. If the residual corpus shift that keeps control
pairs separable is carried by a few more directions, or by the unequal variance of the encoder's
axes, then a richer geometry -- whitening, or stripping several leading components -- should shrink
that shift without destroying the separation of pairs a non-cosine tier already proved related.

Every variant is evaluated in the same frame as the shipped space: the same comparable chunks, the
same control bank, the same rank baseline, and per-pair recovery of the swept baseline rather than a
count comparison, because a rescaled space's absolute cosines are not comparable to the shipped
threshold.
"""

from dataclasses import dataclass

import numpy as np

from llb.conflicts.null_research.generations.advanced import candidate_gates
from llb.conflicts.null_research.controls.balance import (
    BalancedControls,
    raw_rows,
    score_separability,
)
from llb.conflicts.null_research.evaluation import (
    FIXTURE_POSITIVE_DOC_PAIRS,
    fixture_metrics,
    paired_transfer_payload,
)
from llb.conflicts.null_research.geometry import CorpusGeometry, DocPair
from llb.conflicts.null_research.controls.matching import (
    MAX_ABS_STANDARDIZED_DIFFERENCE,
    MAX_MEMBERSHIP_AUC,
)
from llb.core.contracts.common import JsonObject

GEOMETRY_METHODS = ("whitened_cosine", "anisotropy_stripped_cosine")
STRIPPED_COMPONENTS = 3
WHITENING_RIDGE = 1e-2


@dataclass(frozen=True)
class GeometrySpace:
    """One corpus re-expressed in a variant geometry, with its controls rescored there."""

    scores: np.ndarray
    document_maxima: dict[DocPair, float]
    control_scores: np.ndarray
    related_scores: list[float]


def _unit_rows(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return np.asarray(rows / norms)


def _variant_map(centered: np.ndarray, method: str) -> np.ndarray:
    """The linear map a variant applies, fitted on the target corpus's comparable rows."""
    _, values, right = np.linalg.svd(centered - centered.mean(axis=0), full_matrices=False)
    eigenvalues = (values**2) / max(1, len(centered))
    if method == "whitened_cosine":
        scale = 1.0 / np.sqrt(eigenvalues + WHITENING_RIDGE)
        return np.asarray(right.T @ np.diag(scale) @ right)
    if method == "anisotropy_stripped_cosine":
        components = right[:STRIPPED_COMPONENTS]
        return np.asarray(np.eye(centered.shape[1]) - components.T @ components)
    raise ValueError(f"unknown geometry variant {method!r}")


def cross_document_scores(rows: np.ndarray, groups: list[int]) -> np.ndarray:
    """Cross-document upper-triangle similarities in a fixed order, so spaces stay pairable."""
    labels = np.asarray(groups)
    similarity = rows @ rows.T
    upper = np.triu(np.ones_like(similarity, dtype=bool), k=1)
    return np.asarray(similarity[upper & (labels[:, None] != labels[None, :])])


def baseline_pair_scores(corpus: CorpusGeometry) -> np.ndarray:
    """The shipped centered space's scores in the same pair order as any variant geometry."""
    doc_ids = [corpus.chunks[index]["doc_id"] for index in corpus.allowed]
    codes = {doc_id: code for code, doc_id in enumerate(sorted(set(doc_ids)))}
    rows = np.asarray([corpus.vectors.row(index) for index in corpus.allowed], dtype="float64")
    return cross_document_scores(rows, [codes[doc_id] for doc_id in doc_ids])


def _document_maxima(rows: np.ndarray, doc_ids: list[str]) -> dict[DocPair, float]:
    similarity = rows @ rows.T
    unique = sorted(set(doc_ids))
    labels = np.asarray(doc_ids)
    maxima: dict[DocPair, float] = {}
    for position, left in enumerate(unique):
        for right in unique[position + 1 :]:
            block = similarity[np.ix_(labels == left, labels == right)]
            if block.size:
                maxima[(left, right)] = float(block.max())
    return maxima


def build_geometry_space(
    corpus: CorpusGeometry,
    controls: BalancedControls,
    related_pairs: list[tuple[int, int]],
    method: str,
) -> GeometrySpace:
    """Re-express one corpus and its control bank in a variant geometry."""
    mean = np.asarray(corpus.raw_vectors.mean_vector(), dtype="float64")
    centered = raw_rows(corpus) - mean
    mapping = _variant_map(centered, method)
    rows = _unit_rows(centered @ mapping)
    references = _unit_rows((controls.reference_raw - mean) @ mapping)
    doc_ids = [corpus.chunks[index]["doc_id"] for index in corpus.allowed]
    codes = {doc_id: code for code, doc_id in enumerate(sorted(set(doc_ids)))}
    positions = {ordinal: position for position, ordinal in enumerate(corpus.allowed)}
    return GeometrySpace(
        scores=cross_document_scores(rows, [codes[doc_id] for doc_id in doc_ids]),
        document_maxima=_document_maxima(rows, doc_ids),
        control_scores=rows @ references.T,
        related_scores=[
            float(rows[positions[left]] @ rows[positions[right]]) for left, right in related_pairs
        ],
    )


def _shift_diagnostics(space: GeometrySpace, weights: np.ndarray) -> JsonObject:
    """Mean shift is the easy half; the score-level AUC is what a threshold actually rides on."""
    observed = np.asarray(space.scores, dtype="float64")
    control_mean = float(
        (space.control_scores @ weights).sum() / (space.control_scores.shape[0] * weights.sum())
    )
    pooled = float(np.concatenate((observed, space.control_scores.reshape(-1))).std())
    pooled = pooled if pooled > 1e-12 else 1.0
    shift = abs(float(observed.mean()) - control_mean) / pooled
    related = np.asarray(space.related_scores, dtype="float64")
    separation = (float(related.mean()) - float(observed.mean())) / pooled if related.size else None
    separability = score_separability(observed, space.control_scores, weights)
    return {
        "control_shift_z": round(shift, 6),
        "score_separability_auc": round(separability, 6),
        "related_separation_z": round(separation, 6) if separation is not None else None,
        "related_anchor_pairs": int(related.size),
        "control_shift_resolved": shift <= MAX_ABS_STANDARDIZED_DIFFERENCE,
        "exchangeable": shift <= MAX_ABS_STANDARDIZED_DIFFERENCE
        and separability <= MAX_MEMBERSHIP_AUC,
    }


def geometry_candidate(
    method: str,
    spaces: dict[str, GeometrySpace],
    baselines: dict[str, np.ndarray],
    tails: dict[str, JsonObject],
    thresholds: dict[str, float],
    rank: JsonObject,
    weights: dict[str, np.ndarray],
    *,
    transfer_threshold: float,
    max_goods_candidates: int,
) -> JsonObject:
    """Score a variant geometry against the shipped baseline on fixture, HR, and goods."""
    diagnostics = {
        dataset: _shift_diagnostics(space, weights[dataset]) for dataset, space in spaces.items()
    }
    fixture = fixture_metrics(
        spaces["fixture"].document_maxima, thresholds["fixture"], FIXTURE_POSITIVE_DOC_PAIRS
    )
    transfers = {
        dataset: paired_transfer_payload(
            baselines[dataset],
            spaces[dataset].scores,
            thresholds[dataset],
            transfer_threshold,
        )
        for dataset in ("hr", "goods")
    }
    gates = candidate_gates(
        fixture,
        rank,
        transfers["hr"],
        transfers["goods"],
        tails,
        diagnostics,
        max_goods_candidates=max_goods_candidates,
        eligible=True,
        control_key="exchangeable",
        extra={
            "control_shift_resolved": all(
                bool(payload["control_shift_resolved"]) for payload in diagnostics.values()
            )
        },
    )
    return {
        "method": method,
        "thresholds": {key: round(value, 6) for key, value in thresholds.items()},
        "null_tails": tails,
        "diagnostics": diagnostics,
        "fixture": fixture,
        "hr": transfers["hr"],
        "goods": transfers["goods"],
        "gates": gates,
    }
