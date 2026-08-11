"""Surface-matched multi-reference controls for conflict-null research."""

from dataclasses import dataclass
from statistics import mean, median

from llb.conflicts.null_research_geometry import CorpusGeometry
from llb.conflicts.null_research_matching import (
    match_reference_rows,
    membership_diagnostics,
)
from llb.core.contracts.common import JsonObject


@dataclass(frozen=True)
class MatchedControls:
    """Matched control scores clustered by target chunk."""

    scores: list[float]
    source_scores: dict[int, list[float]]
    source_centers: dict[int, float]
    effective_units: int
    diagnostics: JsonObject

    def residual_scores(self) -> list[float]:
        return sorted(
            score - self.source_centers[source]
            for source, values in self.source_scores.items()
            for score in values
        )

    def cluster_maxima(self, *, residual: bool = False) -> list[float]:
        return sorted(
            max(values) - (self.source_centers[source] if residual else 0.0)
            for source, values in self.source_scores.items()
        )


def build_matched_controls(
    target: CorpusGeometry,
    references: dict[str, CorpusGeometry],
    *,
    matches_per_reference: int,
) -> MatchedControls:
    """Select surface/encoder-neighborhood peers from every independent reference corpus."""
    if matches_per_reference < 1:
        raise ValueError("matches per reference must be positive")
    if not references:
        raise ValueError("matched controls need at least one reference corpus")
    if not target.allowed:
        raise ValueError("matched controls need target chunks")
    rows = match_reference_rows(target, references, matches_per_reference)
    source_scores = rows.source_scores
    unique_source_texts = len({target.chunks[source]["text"] for source in source_scores})
    diagnostics = {
        **membership_diagnostics(rows.target_features, rows.matched_features),
        "reference_domains": sorted(references),
        "matches_per_reference": matches_per_reference,
        "control_rows": sum(len(values) for values in source_scores.values()),
        "source_clusters": len(source_scores),
        "unique_source_texts": unique_source_texts,
        "unique_reference_chunks": len(rows.selected_references),
        "unique_reference_texts": len(rows.selected_reference_texts),
        "mean_standardized_match_distance": round(mean(rows.match_distances), 6),
    }
    return MatchedControls(
        scores=sorted(score for values in source_scores.values() for score in values),
        source_scores=source_scores,
        source_centers={source: median(values) for source, values in source_scores.items()},
        effective_units=min(unique_source_texts, len(rows.selected_reference_texts)),
        diagnostics=diagnostics,
    )
