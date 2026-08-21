"""Execute and decide the independent-null research matrix.

This module is the DISPATCHER: it validates the option combination a generation needs, prepares
the corpus geometries and the rank baseline every generation shares, and hands off to the matching
builder in `null_research_summaries`.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from llb.conflicts.constants import (
    DEFAULT_CROSS_ENCODER_ROWS,
    DEFAULT_SYNTHESIS_PER_DOCUMENT,
    RESEARCH_GENERATION_FOURTH,
    RESEARCH_GENERATION_INITIAL,
    RESEARCH_GENERATION_NEXT,
    RESEARCH_GENERATION_THIRD,
    RESEARCH_GENERATIONS,
)
from llb.conflicts.null_research.evaluation import (
    DEFAULT_MAX_GOODS_CANDIDATES,
    DEFAULT_RESEARCH_FPR,
    DEFAULT_RESEARCH_RANK_BUDGET,
    DEFAULT_TRANSFER_THRESHOLD,
    FIXTURE_POSITIVE_DOC_PAIRS,
    rank_baseline,
)
from llb.conflicts.null_research.geometry import CorpusGeometry, prepare_geometry
from llb.conflicts.null_research.generations.initial import RESEARCH_METHODS
from llb.conflicts.null_research.summaries import (
    EmbedTexts,
    ResearchBudget,
    fourth_generation_summary,
    initial_summary,
    next_generation_summary,
    third_generation_summary,
)
from llb.conflicts.store_access import StoreView
from llb.core.contracts.common import JsonObject

if TYPE_CHECKING:
    from llb.prep.frontier.telemetry import LLMComplete
    from llb.rag.rerank import RerankScorer

__all__ = ["RESEARCH_METHODS", "run_null_research"]

Corpus = tuple[Path, StoreView]


def _reference_geometry(
    generation: str,
    reference: Corpus | None,
    domain_reference: Corpus | None,
) -> CorpusGeometry:
    """The collected reference bank every generation before the fourth thresholds against.

    The fourth generation builds its control bank from the target corpus itself, so it never asks
    for one -- and this function is never reached on that path.
    """
    if reference is None:
        raise ValueError(f"{generation}-generation research requires a reference corpus")
    if generation in (RESEARCH_GENERATION_NEXT, RESEARCH_GENERATION_THIRD) and (
        domain_reference is None
    ):
        raise ValueError(f"{generation}-generation research requires a domain reference corpus")
    return prepare_geometry("reference", *reference)


def _scored_geometries(fixture: Corpus, hr: Corpus, goods: Corpus) -> dict[str, CorpusGeometry]:
    """The three corpora every generation scores, in the order the artifact records them."""
    return {
        name: prepare_geometry(name, corpus, store)
        for name, (corpus, store) in {"fixture": fixture, "hr": hr, "goods": goods}.items()
    }


def run_null_research(
    *,
    fixture: Corpus,
    hr: Corpus,
    goods: Corpus,
    reference: Corpus | None = None,
    embed: EmbedTexts | None = None,
    domain_reference: Corpus | None = None,
    generation: str = RESEARCH_GENERATION_INITIAL,
    complete: "LLMComplete | None" = None,
    scorer: "RerankScorer | None" = None,
    fpr: float = DEFAULT_RESEARCH_FPR,
    rank_budget: int = DEFAULT_RESEARCH_RANK_BUDGET,
    transfer_threshold: float = DEFAULT_TRANSFER_THRESHOLD,
    max_goods_candidates: int = DEFAULT_MAX_GOODS_CANDIDATES,
    permutations: int = 3,
    matches_per_reference: int = 2,
    adjudication_budget: int = 50,
    role_samples_per_type: int = 12,
    synthesis_per_document: int = DEFAULT_SYNTHESIS_PER_DOCUMENT,
    cross_encoder_rows: int = DEFAULT_CROSS_ENCODER_ROWS,
    adjudicator_model: str = "",
    cross_encoder_model: str = "",
    seed: int = 0,
) -> JsonObject:
    """Evaluate all proposed nulls and return a complete adopt-or-reject record."""
    if generation not in RESEARCH_GENERATIONS:
        raise ValueError(
            f"unknown research generation {generation!r}; expected one of {RESEARCH_GENERATIONS}"
        )
    corpora = _scored_geometries(fixture, hr, goods)
    rank = rank_baseline(
        corpora["fixture"].observed_similarities,
        corpora["fixture"].document_maxima,
        FIXTURE_POSITIVE_DOC_PAIRS,
        rank_budget,
    )
    budget = ResearchBudget(
        fpr=fpr,
        rank_budget=rank_budget,
        transfer_threshold=transfer_threshold,
        max_goods_candidates=max_goods_candidates,
        seed=seed,
    )
    if generation != RESEARCH_GENERATION_THIRD and embed is None:
        raise ValueError(f"{generation}-generation research requires an embedder")
    if generation == RESEARCH_GENERATION_FOURTH:
        if complete is None or scorer is None:
            raise ValueError(
                "fourth-generation research requires a claim adjudication model and a cross-encoder"
            )
        assert embed is not None
        return fourth_generation_summary(
            corpora,
            rank,
            complete,
            embed,
            scorer,
            budget,
            adjudicator_model=adjudicator_model,
            cross_encoder_model=cross_encoder_model,
            synthesis_per_document=synthesis_per_document,
            cross_encoder_rows=cross_encoder_rows,
        )
    reference_geometry = _reference_geometry(generation, reference, domain_reference)
    if generation == RESEARCH_GENERATION_THIRD:
        if complete is None:
            raise ValueError("third-generation research requires a claim adjudication model")
        assert domain_reference is not None
        return third_generation_summary(
            corpora,
            reference_geometry,
            domain_reference,
            rank,
            complete,
            budget,
            adjudicator_model=adjudicator_model,
            adjudication_budget=adjudication_budget,
            role_samples_per_type=role_samples_per_type,
        )
    if generation == RESEARCH_GENERATION_NEXT:
        assert domain_reference is not None and embed is not None
        return next_generation_summary(
            corpora,
            reference_geometry,
            domain_reference,
            rank,
            embed,
            budget,
            matches_per_reference=matches_per_reference,
        )
    assert embed is not None
    return initial_summary(
        corpora, reference_geometry, rank, embed, budget, permutations=permutations
    )
