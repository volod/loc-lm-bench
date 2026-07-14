"""Text-analysis run, scoring, judge, and persistence data contracts."""

from dataclasses import dataclass
from pathlib import Path

from llb.bench.common import JudgeScorer, Mirror
from llb.core.contracts import BoardRow, JudgeInputRecord, RunPaths, TextAnalysisCaseRow
from llb.scoring.leaderboard import ModelResult
from llb.scoring.judge.model import JudgeOutcome


@dataclass(slots=True)
class TextAnalysisRun:
    result: ModelResult
    rows: list[TextAnalysisCaseRow]
    board: list[BoardRow]
    table: str
    paths: RunPaths | None
    judged_quality: float | None = None
    judged_quality_ci: tuple[float, float] | None = None
    judge_trusted: bool = False
    judge_reason: str = "no judge configured"


@dataclass(slots=True)
class ScoredTextAnalysisDocs:
    doc_ids: list[str]
    rows: list[TextAnalysisCaseRow]
    case_objectives: list[float]
    judge_records: list[JudgeInputRecord]
    judge_row_index: list[int]
    n_ok: int


@dataclass(slots=True)
class JudgeQualityResult:
    outcome: JudgeOutcome
    value: float | None
    ci: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    model: str | None
    rho: float | None
    threshold: float
    scorer: JudgeScorer | None
    base_url: str | None


@dataclass(frozen=True, slots=True)
class TextAnalysisPersistInput:
    data_dir: Path | str | None
    run_name: str
    model: str
    backend: str
    bundle: Path | str
    synthetic: bool
    n_docs: int
    result: ModelResult
    reliability: float
    rows: list[TextAnalysisCaseRow]
    judge_result: JudgeQualityResult
    judge_config: JudgeConfig
    verification_cfg: dict[str, object]
    tokens_per_s: float
    mirror: Mirror | None
