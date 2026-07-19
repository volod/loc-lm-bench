"""Hard per-run frontier cost ledger with JSONL persistence and resume."""

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llb.core.contracts.judging import JudgeScore
from llb.scoring.policy.consent import scorer_dir
from llb.scoring.policy.errors import BudgetExceeded

LEDGER_FILENAME = "ledger.jsonl"
STATE_FILENAME = "ledger_state.json"


@dataclass(frozen=True)
class LedgerEntry:
    """One scored frontier call (or a failed attempt) recorded for audit and resume."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    case_index: int | None = None
    error: str | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }
        if self.case_index is not None:
            payload["case_index"] = self.case_index
        if self.error:
            payload["error"] = self.error
        if self.faithfulness is not None:
            payload["faithfulness"] = float(self.faithfulness)
        if self.answer_relevancy is not None:
            payload["answer_relevancy"] = float(self.answer_relevancy)
        return payload

    def as_score(self) -> JudgeScore:
        """Judge scores for resume; failures and missing fields become zeros."""
        return {
            "faithfulness": float(self.faithfulness or 0.0),
            "answer_relevancy": float(self.answer_relevancy or 0.0),
        }


@dataclass
class CostLedger:
    """Append-only spend tracker that aborts cleanly at the configured cap.

    Successful (and failed-but-attempted) case scores are keyed by ``case_index`` so a
    budget-abort resume can skip already-scored cases without re-spending.
    """

    max_usd: float | None = None
    max_calls: int | None = None
    calls: int = 0
    cost_usd: float = 0.0
    path: Path | None = None
    state_path: Path | None = None
    case_scores: dict[int, JudgeScore] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @classmethod
    def open(
        cls,
        run_dir: Path,
        *,
        max_usd: float | None,
        max_calls: int | None,
    ) -> "CostLedger":
        """Open or resume the ledger under ``run_dir/scorer/``."""
        root = scorer_dir(run_dir)
        root.mkdir(parents=True, exist_ok=True)
        ledger = cls(
            max_usd=max_usd,
            max_calls=max_calls,
            path=root / LEDGER_FILENAME,
            state_path=root / STATE_FILENAME,
        )
        ledger._load_resume_state()
        return ledger

    def scored_case(self, case_index: int) -> JudgeScore | None:
        """Return a previously checkpointed score for ``case_index``, if any."""
        with self._lock:
            return self.case_scores.get(case_index)

    def remaining_calls(self) -> int | None:
        if self.max_calls is None:
            return None
        return max(0, self.max_calls - self.calls)

    def remaining_usd(self) -> float | None:
        if self.max_usd is None:
            return None
        return max(0.0, self.max_usd - self.cost_usd)

    def reserve_call(self) -> None:
        with self._lock:
            self._raise_if_exhausted()
            self.calls += 1
            self._persist_state()

    def record(self, entry: LedgerEntry) -> None:
        with self._lock:
            self.cost_usd += entry.cost_usd
            if entry.case_index is not None:
                self.case_scores[entry.case_index] = entry.as_score()
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry.to_dict(), ensure_ascii=True) + "\n")
            self._persist_state()
            if self.max_usd is not None and self.cost_usd > self.max_usd:
                raise BudgetExceeded(
                    f"frontier spend budget exceeded: ${self.cost_usd:.6f} > ${self.max_usd:.6f}",
                    calls=self.calls,
                    cost_usd=self.cost_usd,
                )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            remaining_usd = self.remaining_usd()
            return {
                "calls": self.calls,
                "cost_usd": round(self.cost_usd, 6),
                "max_calls": self.max_calls,
                "max_usd": self.max_usd,
                "remaining_calls": self.remaining_calls(),
                "remaining_usd": None if remaining_usd is None else round(remaining_usd, 6),
                "scored_cases": len(self.case_scores),
                "resumable": True,
            }

    def abort_payload(self, reason: str) -> dict[str, Any]:
        payload = self.summary()
        payload["status"] = "aborted"
        payload["reason"] = reason
        return payload

    def _raise_if_exhausted(self) -> None:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExceeded(
                f"frontier call budget exhausted: {self.calls} >= {self.max_calls}",
                calls=self.calls,
                cost_usd=self.cost_usd,
            )
        if self.max_usd is not None and self.cost_usd >= self.max_usd:
            raise BudgetExceeded(
                f"frontier spend budget exhausted: ${self.cost_usd:.6f} >= ${self.max_usd:.6f}",
                calls=self.calls,
                cost_usd=self.cost_usd,
            )

    def _persist_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.write_text(
            json.dumps(
                {
                    "calls": self.calls,
                    "cost_usd": round(self.cost_usd, 6),
                    "max_calls": self.max_calls,
                    "max_usd": self.max_usd,
                    "scored_cases": len(self.case_scores),
                    "resumable": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _load_resume_state(self) -> None:
        if self.state_path is not None and self.state_path.is_file():
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.calls = int(raw.get("calls", 0))
            self.cost_usd = float(raw.get("cost_usd", 0.0))
        self._load_case_scores()

    def _load_case_scores(self) -> None:
        """Rebuild the case-index checkpoint map from ``ledger.jsonl``."""
        if self.path is None or not self.path.is_file():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or "case_index" not in payload:
                continue
            try:
                case_index = int(payload["case_index"])
            except (TypeError, ValueError):
                continue
            self.case_scores[case_index] = {
                "faithfulness": float(payload.get("faithfulness") or 0.0),
                "answer_relevancy": float(payload.get("answer_relevancy") or 0.0),
            }
