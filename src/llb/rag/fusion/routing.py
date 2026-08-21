"""Deterministic question-type routing for graph-vector fusion.

The router is deliberately cheap and score-split independent.  A sidecar label wins when the
question has one; otherwise a conservative lexical heuristic decides whether the question looks
like it needs evidence linked across spans.  The routed graph share is therefore either the
configured weight or exactly zero -- the latter preserves the vector ranking byte-for-byte.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

ROUTER_FIXED = "fixed"
ROUTER_QUESTION_TYPE = "question_type"

ROUTE_GRAPH = "graph"
ROUTE_VECTOR = "vector"

# These labels describe the evidence shape strongly enough to route without inspecting the
# scored split.  Unknown labels fall through to the heuristic instead of silently becoming a
# graph or vector decision.
MULTI_SPAN_TYPES = frozenset({"comparative", "multi-hop"})
SINGLE_SPAN_TYPES = frozenset({"definition", "factoid", "numeric", "procedural"})

HEURISTIC_LONG_QUESTION_WORDS = 16
HEURISTIC_MIN_LINKED_ENTITIES = 2
_WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_BRIDGE_RE = re.compile(
    r"\b(?:"
    r"між|порівня(?:й|йте|ти|но)|відрізня(?:ється|ються)|пов['\u2019]?язан\w*|"
    r"залеж\w+\s+від|на\s+основі|через\s+що|який\s+з\s+них|"
    r"between|compare|compared|difference|relationship|related|based\s+on|which\s+of"
    r")\b",
    re.IGNORECASE,
)


class RoutingDecision(NamedTuple):
    """One auditable routing result."""

    graph_weight: float
    route: str
    source: str
    question_type: str | None
    signals: tuple[str, ...]


@dataclass(frozen=True)
class HeuristicPolicy:
    """Two deterministic thresholds of the sidecar-free fallback."""

    long_question_words: int = HEURISTIC_LONG_QUESTION_WORDS
    min_linked_entities: int = HEURISTIC_MIN_LINKED_ENTITIES

    def __post_init__(self) -> None:
        if self.long_question_words < 1:
            raise ValueError("long-question threshold must be at least 1")
        if self.min_linked_entities < 0:
            raise ValueError("linked-entity threshold must be non-negative")

    @property
    def label(self) -> str:
        return f"w{self.long_question_words}/e{self.min_linked_entities}"


DEFAULT_HEURISTIC_POLICY = HeuristicPolicy()


def normalize_question_type(question_type: str) -> str:
    """Normalize the sidecar taxonomy's historical underscore/hyphen spelling variants."""
    return question_type.strip().lower().replace("_", "-")


class QuestionTypeRouter:
    """Choose the configured graph share only for likely multi-span questions."""

    def __init__(
        self,
        graph_weight: float,
        question_types: Mapping[str, str] | None = None,
        heuristic_policy: HeuristicPolicy = DEFAULT_HEURISTIC_POLICY,
    ) -> None:
        if not 0.0 <= graph_weight <= 1.0:
            raise ValueError(f"graph weight must be within [0, 1], got {graph_weight}")
        self.configured_graph_weight = graph_weight
        self.question_types = dict(question_types or {})
        self.heuristic_policy = heuristic_policy

    def decide(self, question: str) -> RoutingDecision:
        """Route by a recognized sidecar label, else by deterministic question text signals."""
        raw_type = self.question_types.get(question)
        if raw_type is not None:
            question_type = normalize_question_type(raw_type)
            if question_type in MULTI_SPAN_TYPES:
                return self._decision(True, "sidecar", question_type, (question_type,))
            if question_type in SINGLE_SPAN_TYPES:
                return self._decision(False, "sidecar", question_type, (question_type,))
        else:
            question_type = None
        signals = heuristic_signals(question, self.heuristic_policy)
        # A bridge term is a direct relation/comparison signal.  Without one, require both a long
        # question and multiple named entities so ordinary verbose factoids stay on vector.
        entity_ready = bool(
            {"multiple_linked_entities", "entity_requirement_disabled"}.intersection(signals)
        )
        use_graph = "bridge_term" in signals or ("long_question" in signals and entity_ready)
        return self._decision(use_graph, "heuristic", question_type, tuple(sorted(signals)))

    def graph_weight(self, question: str) -> float:
        return self.decide(question).graph_weight

    def _decision(
        self,
        use_graph: bool,
        source: str,
        question_type: str | None,
        signals: tuple[str, ...],
    ) -> RoutingDecision:
        return RoutingDecision(
            self.configured_graph_weight if use_graph else 0.0,
            ROUTE_GRAPH if use_graph else ROUTE_VECTOR,
            source,
            question_type,
            signals,
        )


def heuristic_signals(
    question: str, policy: HeuristicPolicy = DEFAULT_HEURISTIC_POLICY
) -> set[str]:
    """Return the stable, human-readable signals used by the fallback router."""
    words = _WORD_RE.findall(question)
    signals: set[str] = set()
    if len(words) >= policy.long_question_words:
        signals.add("long_question")
    # Ignore the sentence's first token: capitalization there is grammatical, not an entity cue.
    named = {word.casefold() for word in words[1:] if word[:1].isupper()}
    if policy.min_linked_entities == 0:
        signals.add("entity_requirement_disabled")
    elif len(named) >= policy.min_linked_entities:
        signals.add("multiple_linked_entities")
    if _BRIDGE_RE.search(question):
        signals.add("bridge_term")
    return signals
