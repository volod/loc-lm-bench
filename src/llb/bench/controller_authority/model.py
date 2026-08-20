"""Data contracts for controller-channel authority runs."""

from dataclasses import dataclass

from llb.core.contracts.benchmarks import AgenticCaseRow
from llb.core.contracts.common import ChatMessage


@dataclass(frozen=True, slots=True)
class ChannelCell:
    placement: str
    rows: list[AgenticCaseRow]
    snapshots: dict[str, list[ChatMessage]]
    manifest: str | None = None
    tokens_per_s: float = 0.0


@dataclass(frozen=True, slots=True)
class ChannelSeedRun:
    seed: int
    model: str
    backend: str
    cells: dict[str, ChannelCell]
