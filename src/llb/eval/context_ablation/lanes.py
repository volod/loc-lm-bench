"""Resolve context-lane labels into scored `run-eval` configs.

A lane label IS the `RunConfig.context_strategy` it selects, so the comparison never invents a
knob: `make compare-context-strategies` and `make run-eval CONTEXT_STRATEGY=<label>` produce the
same bundle for the same lane, which is what makes the `rag` lane's per-case scores checkable
against a plain scored run.
"""

from llb.core.config import RunConfig
from llb.eval.context_ablation.models import LANE_CLOSED_BOOK, LANE_RAG, LANES


def parse_lanes(spec: str) -> list[str]:
    """Parse a comma-separated lane selection, de-duplicated in the order given.

    `closed_book` is forced to the front when selected: it is the reference every derived number
    is stated against, and a comparison whose baseline moved would not be readable beside another.
    """
    labels = [token.strip() for token in spec.split(",") if token.strip()]
    unknown = [label for label in labels if label not in LANES]
    if unknown:
        raise ValueError(
            f"unknown context lane(s) {', '.join(unknown)}; choose from {', '.join(LANES)}"
        )
    ordered = list(dict.fromkeys(labels))
    if not ordered:
        raise ValueError("no lane parsed from the lane selection")
    if LANE_CLOSED_BOOK in ordered:
        ordered.remove(LANE_CLOSED_BOOK)
        ordered.insert(0, LANE_CLOSED_BOOK)
    return ordered


def lane_config(config: RunConfig, lane: str, *, run_name_prefix: str) -> RunConfig:
    """`config` with this lane's context strategy applied and a lane-identifying run name."""
    if lane not in LANES:
        raise ValueError(f"unknown context lane {lane!r}; choose from {', '.join(LANES)}")
    return config.with_overrides(context_strategy=lane, run_name=f"{run_name_prefix}-{lane}")


def default_lanes() -> list[str]:
    """All three lanes, baseline first."""
    return list(LANES)


__all__ = ["LANE_CLOSED_BOOK", "LANE_RAG", "default_lanes", "lane_config", "parse_lanes"]
