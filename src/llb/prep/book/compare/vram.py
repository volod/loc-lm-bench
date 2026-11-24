"""Host VRAM against the budget a selected configuration names.

A budget of zero is a CPU engine and is not checked. The same engine named on
several stages is refused once. The message names the engine, which is the refusal
a dry run shows the operator.
"""

from collections.abc import Sequence

from llb.core.contracts.data_prep.book import BookStageSpec


def vram_requirement_errors(stages: Sequence[BookStageSpec], host_mb: int | None) -> list[str]:
    """One refusal per engine whose budget is larger than ``host_mb``."""
    errors: list[str] = []
    seen: set[str] = set()
    for spec in stages:
        raw = spec.params.get("vram_mb")
        if not isinstance(raw, dict):
            continue
        for engine, need in raw.items():
            if not isinstance(engine, str) or engine in seen:
                continue
            if isinstance(need, bool) or not isinstance(need, (int, float)) or need <= 0:
                continue
            seen.add(engine)
            errors.append(_refusal(engine, need, host_mb))
    return [error for error in errors if error]


def _refusal(engine: str, need: float, host_mb: int | None) -> str:
    shown = int(need) if float(need).is_integer() else need
    if host_mb is None:
        have = "no CUDA device"
    elif host_mb >= need:
        return ""
    else:
        have = f"{host_mb} MB"
    return (
        f"EngineUnavailable: engine {engine} unavailable: "
        f"needs {shown} MB VRAM and this host has {have}"
    )
