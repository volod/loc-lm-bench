"""Feature construction, balancing diagnostics, and row selection for matched controls."""

from dataclasses import dataclass
from hashlib import sha256
import math
import re

import numpy as np

from llb.conflicts.null_research_geometry import CorpusGeometry
from llb.conflicts.vector_math import Vector, dot
from llb.conflicts.vectorops import VectorSet
from llb.core.contracts.common import JsonObject
from llb.core.contracts.rag import ChunkRecord

MAX_MEMBERSHIP_AUC = 0.60
MAX_ABS_STANDARDIZED_DIFFERENCE = 0.25
MIN_MEMBERSHIP_ROWS = 20
SURFACE_FEATURE_NAMES = (
    "log_tokens",
    "numeric_density",
    "heading_depth",
    "lexical_entropy",
    "punctuation_density",
    "encoder_mean_affinity",
)

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class MatchedRows:
    source_scores: dict[int, list[float]]
    target_features: np.ndarray
    matched_features: np.ndarray
    selected_references: set[tuple[str, int]]
    selected_reference_texts: set[str]
    match_distances: list[float]


@dataclass(frozen=True)
class _MatchingSpace:
    target_features: np.ndarray
    reference_features: dict[str, np.ndarray]
    reference_vectors: dict[str, VectorSet]
    normalized_target: np.ndarray
    normalized_references: dict[str, np.ndarray]


def surface_features(chunk: ChunkRecord, vector: Vector, mean_vector: Vector) -> list[float]:
    """Structural covariates a control must match before its cosine can stand in for a null."""
    tokens = _TOKEN.findall(chunk["text"])
    total = max(1, len(tokens))
    counts: dict[str, int] = {}
    for token in tokens:
        folded = token.casefold()
        counts[folded] = counts.get(folded, 0) + 1
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    headings = [
        len(line) - len(line.lstrip("#"))
        for line in chunk["text"].splitlines()
        if line.startswith("#")
    ]
    punctuation = sum(not char.isalnum() and not char.isspace() for char in chunk["text"])
    return [
        math.log1p(len(tokens)),
        sum(any(char.isdigit() for char in token) for token in tokens) / total,
        min(headings) / 6.0 if headings else 0.0,
        entropy / math.log(total) if total > 1 else 0.0,
        punctuation / max(1, len(chunk["text"])),
        dot(vector, mean_vector),
    ]


def _rank_auc(positive: list[float], negative: list[float]) -> float:
    labelled = sorted([(value, 1) for value in positive] + [(value, 0) for value in negative])
    positive_ranks = 0.0
    position = 0
    while position < len(labelled):
        stop = position + 1
        while stop < len(labelled) and labelled[stop][0] == labelled[position][0]:
            stop += 1
        average_rank = (position + 1 + stop) / 2.0
        positive_ranks += average_rank * sum(label for _, label in labelled[position:stop])
        position = stop
    positives = len(positive)
    negatives = len(negative)
    return (positive_ranks - positives * (positives + 1) / 2.0) / (positives * negatives)


def membership_diagnostics(target: np.ndarray, reference: np.ndarray) -> JsonObject:
    """Cross-fit a diagonal discriminant and report balance on the matched feature rows."""
    rows = len(target)
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    if rows >= 4:
        for fold in (0, 1):
            train = np.asarray([index % 2 != fold for index in range(rows)])
            test = ~train
            pooled = np.concatenate((target[train], reference[train]))
            variance = pooled.var(axis=0)
            variance[variance < 1e-12] = 1.0
            weights = (target[train].mean(axis=0) - reference[train].mean(axis=0)) / variance
            positive_scores.extend((target[test] @ weights).tolist())
            negative_scores.extend((reference[test] @ weights).tolist())
        auc = _rank_auc(positive_scores, negative_scores)
    else:
        auc = 1.0
    pooled_scale = np.concatenate((target, reference)).std(axis=0)
    pooled_scale[pooled_scale < 1e-12] = 1.0
    differences = np.abs((target.mean(axis=0) - reference.mean(axis=0)) / pooled_scale)
    oriented_auc = max(auc, 1.0 - auc)
    max_difference = float(differences.max())
    resolved = rows >= MIN_MEMBERSHIP_ROWS
    return {
        "classifier": "two-fold nearest-centroid linear discriminant",
        "classifier_rows_per_class": rows,
        "classifier_resolved": resolved,
        "membership_auc": round(oriented_auc, 6),
        "max_abs_standardized_difference": round(max_difference, 6),
        "feature_names": list(SURFACE_FEATURE_NAMES),
        "exchangeable": resolved
        and oriented_auc <= MAX_MEMBERSHIP_AUC
        and max_difference <= MAX_ABS_STANDARDIZED_DIFFERENCE,
    }


def _feature_matrix(geometry: CorpusGeometry, mean_vector: Vector) -> np.ndarray:
    return np.asarray(
        [
            surface_features(geometry.chunks[index], geometry.raw_vectors.row(index), mean_vector)
            for index in geometry.allowed
        ],
        dtype="float64",
    )


def _matching_space(
    target: CorpusGeometry, references: dict[str, CorpusGeometry]
) -> _MatchingSpace:
    target_docs = {target.chunks[index]["doc_id"] for index in target.allowed}
    feature_mean = target.raw_vectors.mean_vector()
    target_features = _feature_matrix(target, feature_mean)
    reference_features: dict[str, np.ndarray] = {}
    reference_vectors: dict[str, VectorSet] = {}
    for name, reference in references.items():
        if not reference.allowed:
            raise ValueError(f"matched reference {name!r} has no comparable chunks")
        if target.raw_vectors.dim != reference.raw_vectors.dim:
            raise ValueError("target and reference stores use different vector dimensions")
        reference_docs = {reference.chunks[index]["doc_id"] for index in reference.allowed}
        overlap = target_docs & reference_docs
        if overlap:
            preview = ", ".join(sorted(overlap)[:3])
            raise ValueError("matched controls require disjoint document ids; overlap: " + preview)
        reference_features[name] = _feature_matrix(reference, feature_mean)
        reference_vectors[name] = (
            reference.raw_vectors.centered(target.center_mean)
            if target.center_mean is not None
            else reference.raw_vectors
        )
    combined = np.concatenate([target_features, *reference_features.values()])
    center = combined.mean(axis=0)
    scale = combined.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return _MatchingSpace(
        target_features=target_features,
        reference_features=reference_features,
        reference_vectors=reference_vectors,
        normalized_target=(target_features - center) / scale,
        normalized_references={
            name: (features - center) / scale for name, features in reference_features.items()
        },
    )


def _select_rows(
    target: CorpusGeometry,
    references: dict[str, CorpusGeometry],
    space: _MatchingSpace,
    matches_per_reference: int,
) -> MatchedRows:
    source_scores: dict[int, list[float]] = {}
    matched_feature_rows: list[np.ndarray] = []
    selected_references: set[tuple[str, int]] = set()
    selected_reference_texts: set[str] = set()
    match_distances: list[float] = []
    for position, source in enumerate(target.allowed):
        values: list[float] = []
        matched_features: list[tuple[float, np.ndarray]] = []
        for name, reference in references.items():
            distances = (
                (space.normalized_references[name] - space.normalized_target[position]) ** 2
            ).sum(axis=1)
            take = min(matches_per_reference, len(reference.allowed))
            for reference_position in np.argsort(distances, kind="stable")[:take].tolist():
                reference_ordinal = reference.allowed[reference_position]
                values.append(
                    dot(
                        target.vectors.row(source),
                        space.reference_vectors[name].row(reference_ordinal),
                    )
                )
                distance = float(math.sqrt(distances[reference_position]))
                matched_features.append(
                    (distance, space.reference_features[name][reference_position])
                )
                selected_references.add((name, reference_ordinal))
                selected_reference_texts.add(
                    sha256(reference.chunks[reference_ordinal]["text"].encode("utf-8")).hexdigest()
                )
                match_distances.append(distance)
        source_scores[source] = values
        matched_feature_rows.append(min(matched_features, key=lambda item: item[0])[1])
    return MatchedRows(
        source_scores=source_scores,
        target_features=space.target_features,
        matched_features=np.asarray(matched_feature_rows),
        selected_references=selected_references,
        selected_reference_texts=selected_reference_texts,
        match_distances=match_distances,
    )


def match_reference_rows(
    target: CorpusGeometry,
    references: dict[str, CorpusGeometry],
    matches_per_reference: int,
) -> MatchedRows:
    """Match every target to the nearest covariate rows in each reference domain."""
    return _select_rows(
        target,
        references,
        _matching_space(target, references),
        matches_per_reference,
    )
