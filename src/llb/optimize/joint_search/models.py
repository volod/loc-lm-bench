"""Result dataclasses for joint-search."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llb.core.contracts.runs import EvalResult
from llb.optimize.joint_search.halving import HalvingLedger


@dataclass
class FinalistTuneResult:
    """Per-finalist multi-objective tune outcome (injectable for CI)."""

    name: str
    backend: str
    source: str
    study_name: str
    overrides_by_pick: dict[str, dict[str, Any]]
    finals: dict[str, EvalResult]
    report_dir: Path | None = None


@dataclass
class JointSearchResult:
    """Full joint-search run: ledger, scoreboard paths, and recommended pick."""

    run_id: str
    run_dir: Path
    ledger: HalvingLedger
    finalists: list[FinalistTuneResult]
    scoreboard_paths: dict[str, Path]
    recommended: dict[str, Any] | None
    skipped: list[dict[str, str]] = field(default_factory=list)
