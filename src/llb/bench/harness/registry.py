"""Named agent-harness resolution with lazy optional-framework imports."""

from llb.bench.agentic.model import (
    HARNESS_CREWAI,
    HARNESS_LANGGRAPH,
    HARNESS_LOOP,
    HARNESS_NAMES,
    Harness,
)
from llb.bench.harness.base import loop_harness


def get_harness(name: str) -> Harness:
    """Resolve a harness id to its implementation."""
    if name == HARNESS_LOOP:
        return loop_harness
    if name == HARNESS_LANGGRAPH:
        from llb.bench.harness.langgraph import langgraph_harness

        return langgraph_harness
    if name == HARNESS_CREWAI:
        from llb.bench.harness.crewai import make_crewai_harness

        return make_crewai_harness()
    raise SystemExit(f"unknown harness '{name}'; choose one of {', '.join(HARNESS_NAMES)}")
