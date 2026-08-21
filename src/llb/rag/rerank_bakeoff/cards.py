"""Per-reranker card references: the scores a candidate must reproduce before it can be ranked.

Same gate as the encoder side (`llb.rag.encoders.cards`), one stage later and with one extra
wrinkle: cross-encoder cards print their reference in whichever space their own snippet uses. The
gte card calls the model through transformers and prints RAW LOGITS; the jina card calls its own
`compute_score` helper and prints SIGMOID probabilities. sentence-transformers' `CrossEncoder`
returns the sigmoid, so a logit-space card declares `TRANSFORM_SIGMOID` and the published number is
squashed before comparison rather than the observed one being un-squashed.

An id with no entry is scored ungated and says so on its row: two of the five roster cards publish
no reference numbers at all, and "nobody checked" must not read as "it reproduces".

Pure apart from the injected scorer: the tables and the arithmetic have no torch, no download.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from llb.rag.encoders.card_parity import (
    TRANSFORM_SIGMOID,
    CardExpectation,
    CardParityResult,
    compare_to_card,
    probe_error_result,
    unpublished_result,
)
from llb.rag.rerank import RerankScorer

_JINA_QUERY = "Organic skincare products for sensitive skin"

# One card example, as the `RerankScorer` seam takes it: a query and the passages scored against
# it. A card whose pairs do not share a query becomes several groups, and the observed scores are
# concatenated in group order -- the order the card prints them in.
QueryGroup = tuple[str, tuple[str, ...]]


@dataclass(frozen=True)
class RerankCardReference:
    """One reranker card's documented example and the scores it publishes, in card order."""

    model: str
    source: str
    groups: tuple[QueryGroup, ...]
    expectation: CardExpectation = field(default_factory=CardExpectation)


RERANK_CARD_REFERENCES: dict[str, RerankCardReference] = {
    "jinaai/jina-reranker-v2-base-multilingual": RerankCardReference(
        model="jinaai/jina-reranker-v2-base-multilingual",
        source="https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual",
        groups=(
            (
                _JINA_QUERY,
                (
                    "Organic skincare for sensitive skin with aloe vera and chamomile.",
                    "New makeup trends focus on bold colors and innovative techniques",
                    "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille",
                    "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken",
                    "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla",
                    "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas "
                    "innovadoras",
                    "针对敏感肌专门设计的天然有机护肤产品",
                    "新的化妆趋势注重鲜艳的颜色和创新的技巧",
                    "敏感肌のために特別に設計された天然有機スキンケア製品",
                    "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています",
                ),
            ),
        ),
        # `compute_score` already applies the sigmoid, so these are in the space `CrossEncoder`
        # returns. bfloat16 weights move the third decimal, which the default tolerance covers.
        expectation=CardExpectation(
            values=(
                0.8311431,
                0.0940102,
                0.6334103,
                0.0826973,
                0.7620701,
                0.0994702,
                0.9263037,
                0.0583458,
                0.8418256,
                0.1112412,
            )
        ),
    ),
    "Alibaba-NLP/gte-multilingual-reranker-base": RerankCardReference(
        model="Alibaba-NLP/gte-multilingual-reranker-base",
        source="https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base",
        # The card's three pairs carry three different queries, so each is its own group.
        groups=(
            ("中国的首都在哪儿", ("北京",)),
            ("what is the capital of China?", ("北京",)),
            ("how to implement quick sort in python?", ("Introduction of quick sort",)),
        ),
        # Card prints `logits.view(-1).float()` -> raw logits; CrossEncoder returns the sigmoid.
        expectation=CardExpectation(values=(1.2315, 0.5923, 0.3041), transform=TRANSFORM_SIGMOID),
    ),
}


def card_reference(model_name: str) -> RerankCardReference | None:
    """The declared card reference for a reranker id (None when nobody recorded one)."""
    return RERANK_CARD_REFERENCES.get(model_name)


def check_rerank_card(model_name: str, scorer: RerankScorer) -> CardParityResult:
    """Run this reranker's declared card example and say whether it reproduced the card.

    The scorer is the same seam the lane scores through, so a candidate cleared here is cleared on
    the exact call path its row is measured on. A probe that raises is a verdict, not an exception:
    a candidate that cannot run its own card example has told us what we needed to know.
    """
    reference = card_reference(model_name)
    if reference is None:
        return unpublished_result(model_name)
    observed: list[float] = []
    try:
        for query, passages in reference.groups:
            scores: Sequence[float] = scorer(query, list(passages))
            observed.extend(float(score) for score in scores)
    except Exception as exc:  # a broken remote-code load raises here, which IS the verdict
        return probe_error_result(
            model_name, reference.source, f"card probe failed: {type(exc).__name__}: {exc}"
        )
    return compare_to_card(model_name, reference.source, reference.expectation, tuple(observed))
