"""Configuration and result contracts for the auto-RAG pipeline."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

GatePolicy = Literal["auto", "human", "local", "frontier"]

STAGES = (
    "ingest",
    "draft",
    "verification",
    "retrieval",
    "joint_search",
    "prompt_system",
    "final_eval",
    "recommendation",
)


@dataclass(frozen=True, slots=True)
class AutoRagSettings:
    """All score- or artifact-affecting inputs pinned by the run manifest."""

    corpus: Path
    data_dir: Path
    run_id: str
    draft_model: str
    candidates: Path
    candidate_models: tuple[str, ...] = ()
    gate_policy: GatePolicy = "auto"
    judge_model: str | None = None
    judge_base_url: str | None = None
    egress_consent: bool = False
    max_usd: float | None = None
    max_calls: int | None = None
    max_items: int = 60
    doc_limit: int | None = None
    seed: int = 13
    draft_max_tokens: int = 4096
    draft_num_ctx: int | None = 8192
    draft_concurrency: int = 1
    verify_threshold: float = 0.5
    min_accept_rate: float = 0.5
    retrieval_k: int = 10
    retrieval_recall_gate: float = 0.8
    trials: int = 20
    screen_limit: int = 8
    min_finalists: int = 2
    objectives: str = "quality,latency"
    eval_limit: int | None = None
    max_model_len: int = 8192
    parity_check: bool = False

    @property
    def run_dir(self) -> Path:
        return self.data_dir / "auto-rag" / self.run_id

    def manifest_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["corpus"] = str(self.corpus)
        payload["data_dir"] = str(self.data_dir)
        payload["candidates"] = str(self.candidates)
        payload["candidate_models"] = list(self.candidate_models)
        return payload


@dataclass(frozen=True, slots=True)
class AutoRagStatus:
    """Outcome returned to the CLI and callers."""

    run_dir: Path
    completed: tuple[str, ...]
    recommendation: Path | None
    report: Path | None
    resumed: bool
