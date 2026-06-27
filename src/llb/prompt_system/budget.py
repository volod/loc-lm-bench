"""M7.3 context-budget controller -- fit a prompt-system package into a model's context window.

Estimates per-model token costs (tokenizer injectable, defaulting to a char/token ratio so the base
install needs no tokenizer dependency), RESERVES space for the question, retrieved chunks, tool
transcript, and answer, then TRIMS the anthology / metadata / graph sections -- least-salient items
first -- so every prompt candidate fits the selected model. The dropped-context report makes every
omission explicit for the human review loop. Pure + deterministic.
"""

from dataclasses import dataclass, field
from typing import Protocol

from typing_extensions import TypedDict

DEFAULT_CHARS_PER_TOKEN = 4.0
DEFAULT_ANSWER_TOKENS = 512


class Tokenizer(Protocol):
    """Token-count seam: a real model tokenizer or the dependency-free char-ratio estimate."""

    def count(self, text: str) -> int: ...


@dataclass(slots=True)
class CharRatioTokenizer:
    """Estimate token count as ceil(len(text) / chars_per_token). Dependency-free default."""

    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN

    def count(self, text: str) -> int:
        if self.chars_per_token <= 0:
            raise ValueError("chars_per_token must be > 0")
        return int(-(-len(text) // self.chars_per_token))  # ceil division


@dataclass(slots=True)
class ContextBudget:
    """The token budget split for one model: reservations + what is left for the prompt system."""

    context_window: int
    question_tokens: int
    chunk_tokens: int
    transcript_tokens: int
    answer_tokens: int

    @property
    def reserved(self) -> int:
        return (
            self.question_tokens + self.chunk_tokens + self.transcript_tokens + self.answer_tokens
        )

    @property
    def prompt_budget(self) -> int:
        """Tokens left for the anthology + metadata + graph sections (never negative)."""
        return max(0, self.context_window - self.reserved)


def plan_budget(
    context_window: int,
    *,
    question_tokens: int,
    chunk_tokens: int,
    transcript_tokens: int = 0,
    answer_tokens: int = DEFAULT_ANSWER_TOKENS,
) -> ContextBudget:
    """Reserve the question / retrieved-chunk / tool-transcript / answer budgets up front."""
    if context_window <= 0:
        raise ValueError("context_window must be > 0")
    return ContextBudget(
        context_window=context_window,
        question_tokens=question_tokens,
        chunk_tokens=chunk_tokens,
        transcript_tokens=transcript_tokens,
        answer_tokens=answer_tokens,
    )


class SectionItem(TypedDict):
    item_id: str
    text: str


class DroppedSection(TypedDict):
    section: str
    n_kept: int
    n_dropped: int
    dropped_ids: list[str]
    dropped_tokens: int


class DroppedContextReport(TypedDict):
    budget_tokens: int
    used_tokens: int
    sections: list[DroppedSection]


@dataclass(slots=True)
class FitResult:
    """The kept items per section (in priority order) plus the dropped-context report."""

    kept: dict[str, list[SectionItem]]
    report: DroppedContextReport
    used_tokens: int = field(default=0)


def fit_sections(
    sections: list[tuple[str, list[SectionItem]]],
    budget_tokens: int,
    tokenizer: Tokenizer,
) -> FitResult:
    """Greedily keep items section-by-section (in the given priority order) until the budget is
    spent; the rest are dropped and reported. Within a section, items are kept in their given order
    (callers pass them most-salient first), so the most important context survives a tight budget."""
    kept: dict[str, list[SectionItem]] = {}
    dropped: list[DroppedSection] = []
    used = 0
    for name, items in sections:
        section_kept: list[SectionItem] = []
        dropped_ids: list[str] = []
        dropped_tokens = 0
        for item in items:
            cost = tokenizer.count(item["text"])
            if used + cost <= budget_tokens:
                section_kept.append(item)
                used += cost
            else:
                dropped_ids.append(item["item_id"])
                dropped_tokens += cost
        kept[name] = section_kept
        dropped.append(
            {
                "section": name,
                "n_kept": len(section_kept),
                "n_dropped": len(dropped_ids),
                "dropped_ids": dropped_ids,
                "dropped_tokens": dropped_tokens,
            }
        )
    report: DroppedContextReport = {
        "budget_tokens": budget_tokens,
        "used_tokens": used,
        "sections": dropped,
    }
    return FitResult(kept=kept, report=report, used_tokens=used)
