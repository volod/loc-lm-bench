"""M7.1 agentic harnesses -- the named-harness registry (LangGraph vs CrewAI vs the pure loop).

Each harness returns the SAME canonical `Episode`, so the objective scorer + isolation contract +
gated judge are unchanged and only the agent framework varies. The two framework harnesses are
OPT-IN, lazy extras (`[eval]` for LangGraph, `[crewai]` for CrewAI), so the base install stays
light; the loop harness is always available and needs no extra.
"""

from llb.bench.agentic import (
    HARNESS_CREWAI,
    HARNESS_LANGGRAPH,
    HARNESS_LOOP,
    HARNESS_NAMES,
    Harness,
)
from llb.bench.harness.base import loop_harness


def get_harness(name: str) -> Harness:
    """Resolve a harness id to its `Harness` callable (heavy frameworks imported lazily)."""
    if name == HARNESS_LOOP:
        return loop_harness
    if name == HARNESS_LANGGRAPH:
        from llb.bench.harness.langgraph import langgraph_harness

        return langgraph_harness
    if name == HARNESS_CREWAI:
        from llb.bench.harness.crewai import make_crewai_harness

        return make_crewai_harness()
    raise SystemExit(f"unknown harness '{name}'; choose one of {', '.join(HARNESS_NAMES)}")


__all__ = [
    "HARNESS_CREWAI",
    "HARNESS_LANGGRAPH",
    "HARNESS_LOOP",
    "HARNESS_NAMES",
    "Harness",
    "get_harness",
    "loop_harness",
]
