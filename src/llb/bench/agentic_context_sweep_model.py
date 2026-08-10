"""The vocabulary of the context-policy constant sweep: axes, grids, cells, and verdicts.

Three constants decide what `observation_cap` and `keep_last_n` do:
`DEFAULT_OBSERVATION_CAP_CHARS`, `OBSERVATION_HEAD_SHARE`, and `DEFAULT_KEEP_LAST_N`. This lane
holds the model, the task set, and the policy FIXED within each axis and varies only one constant,
pairs every non-shipped setting against the shipped value over SHARED bootstrap index sets, and
states a pin / expose / inapplicable verdict per axis. Defaults are not rewritten here: a separated
favorable delta is an expose recommendation, not an automatic ship.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from llb.bench.agentic.context_policy import (
    DEFAULT_KEEP_LAST_N,
    DEFAULT_OBSERVATION_CAP_CHARS,
    OBSERVATION_HEAD_SHARE,
    POLICY_KEEP_LAST_N,
    POLICY_OBSERVATION_CAP,
)
from llb.bench.agentic_context_report import (
    PolicyReport,
)
from llb.rag.fusion_evidence.paired import PairedComparison

_LOG = logging.getLogger(__name__)

METHOD = "agentic-context-sweep"

# Measured grids. Cap and head-share ride under `observation_cap`; keep-last-n under its own policy.
CAP_GRID: tuple[int, ...] = (400, 800, 1600)
HEAD_SHARE_GRID: tuple[float, ...] = (0.5, 0.6, 0.7)
KEEP_LAST_N_GRID: tuple[int, ...] = (1, 2, 3)
# Long-transcript keep grid: same cells, but the lane runs them alone at a higher max_steps over
# medium-observation pipeline tasks (see `agentic_long_transcript`).
KEEP_LONG_TRANSCRIPT_GRID: tuple[int, ...] = KEEP_LAST_N_GRID

AXIS_CAP = "observation_cap_chars"
AXIS_HEAD = "observation_head_share"
AXIS_KEEP = "keep_last_n"
AXES: tuple[str, ...] = (AXIS_CAP, AXIS_HEAD, AXIS_KEEP)

VERDICT_PIN = "pin"
VERDICT_EXPOSE = "expose"
VERDICT_INAPPLICABLE = "inapplicable"


@dataclass(frozen=True, slots=True)
class SweepSetting:
    """One cell of the constant grid: which axis, which policy, and the full override map."""

    axis: str
    label: str
    policy_name: str
    overrides: dict[str, Any]
    is_shipped: bool


@dataclass(slots=True)
class SettingReport:
    """One setting's scored outcome plus its paired delta against the shipped cell of its axis."""

    setting: SweepSetting
    report: PolicyReport
    paired: dict[str, PairedComparison]


@dataclass(slots=True)
class AxisVerdict:
    """Pin / expose / inapplicable for one constant, with the evidence sentence."""

    axis: str
    shipped_value: Any
    verdict: str
    reason: str


@dataclass(slots=True)
class ConstantSweepRun:
    """Outcome of one three-axis constant sweep for a fixed model and task set."""

    model: str
    backend: str
    settings: list[SettingReport]
    verdicts: list[AxisVerdict]
    table: str
    task_set_digest: str
    max_prompt_chars: int


def shipped_value(axis: str) -> Any:
    """The currently shipped default for one axis."""
    return {
        AXIS_CAP: DEFAULT_OBSERVATION_CAP_CHARS,
        AXIS_HEAD: OBSERVATION_HEAD_SHARE,
        AXIS_KEEP: DEFAULT_KEEP_LAST_N,
    }[axis]


def default_grid() -> list[SweepSetting]:
    """The three one-dimensional grids the CUDA evidence run walks."""
    return grid_for_axes(AXES)


def keep_long_transcript_grid() -> list[SweepSetting]:
    """Keep-only grid for the long-transcript lane (keep=1/2/3 under `keep_last_n`)."""
    return grid_for_axes((AXIS_KEEP,), keep_values=KEEP_LONG_TRANSCRIPT_GRID)


def grid_for_axes(
    axes: Sequence[str],
    *,
    keep_values: Sequence[int] | None = None,
) -> list[SweepSetting]:
    """Build the requested one-dimensional grids; unknown axis names raise."""
    unknown = [a for a in axes if a not in AXES]
    if unknown:
        raise SystemExit(f"unknown sweep axes: {unknown}; choose from {AXES}")
    keep_values = tuple(keep_values) if keep_values is not None else KEEP_LAST_N_GRID
    settings: list[SweepSetting] = []
    if AXIS_CAP in axes:
        for cap in CAP_GRID:
            settings.append(
                SweepSetting(
                    axis=AXIS_CAP,
                    label=f"cap={cap}",
                    policy_name=POLICY_OBSERVATION_CAP,
                    overrides={
                        "observation_cap_chars": cap,
                        "observation_head_share": OBSERVATION_HEAD_SHARE,
                        "keep_last_n": DEFAULT_KEEP_LAST_N,
                    },
                    is_shipped=cap == DEFAULT_OBSERVATION_CAP_CHARS,
                )
            )
    if AXIS_HEAD in axes:
        for share in HEAD_SHARE_GRID:
            settings.append(
                SweepSetting(
                    axis=AXIS_HEAD,
                    label=f"head={share}",
                    policy_name=POLICY_OBSERVATION_CAP,
                    overrides={
                        "observation_cap_chars": DEFAULT_OBSERVATION_CAP_CHARS,
                        "observation_head_share": share,
                        "keep_last_n": DEFAULT_KEEP_LAST_N,
                    },
                    is_shipped=share == OBSERVATION_HEAD_SHARE,
                )
            )
    if AXIS_KEEP in axes:
        for keep in keep_values:
            settings.append(
                SweepSetting(
                    axis=AXIS_KEEP,
                    label=f"keep={keep}",
                    policy_name=POLICY_KEEP_LAST_N,
                    overrides={
                        "observation_cap_chars": DEFAULT_OBSERVATION_CAP_CHARS,
                        "observation_head_share": OBSERVATION_HEAD_SHARE,
                        "keep_last_n": keep,
                    },
                    is_shipped=keep == DEFAULT_KEEP_LAST_N,
                )
            )
    return settings


def parse_axes(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse a comma-separated axes string (or pass-through a sequence) into known axis names."""
    if raw is None:
        return AXES
    if isinstance(raw, str):
        parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    else:
        parts = tuple(raw)
    return parts if parts else AXES
