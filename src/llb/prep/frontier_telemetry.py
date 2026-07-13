"""Thread-safe frontier call telemetry and draft budget enforcement."""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

LLMComplete = Callable[[str], str]


@dataclass
class CallRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float = 0.0
    error: str | None = None


@dataclass
class ProvenanceLog:
    """Accumulate one record per attempted endpoint call."""

    calls: list[CallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        *,
        latency_s: float = 0.0,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self.calls.append(
                CallRecord(model, prompt_tokens, completion_tokens, cost_usd, latency_s, error)
            )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            calls = list(self.calls)
        models = sorted({call.model for call in calls})
        total_latency = sum(call.latency_s for call in calls)
        return {
            "calls": len(calls),
            "models": models,
            "total_prompt_tokens": sum(call.prompt_tokens for call in calls),
            "total_completion_tokens": sum(call.completion_tokens for call in calls),
            "total_cost_usd": round(sum(call.cost_usd for call in calls), 6),
            "total_latency_s": round(total_latency, 3),
            "average_latency_s": round(total_latency / len(calls), 3) if calls else 0.0,
            "call_records": [
                {
                    "model": call.model,
                    "prompt_tokens": call.prompt_tokens,
                    "completion_tokens": call.completion_tokens,
                    "cost_usd": round(call.cost_usd, 6),
                    "latency_s": round(call.latency_s, 3),
                    **({"error": call.error} if call.error else {}),
                }
                for call in calls
            ],
        }


class DraftBudgetExceeded(RuntimeError):
    """A frontier draft stopped at a configured call or measured-spend limit."""

    def __init__(self, reason: str, *, calls: int, cost_usd: float):
        super().__init__(reason)
        self.reason = reason
        self.calls = calls
        self.cost_usd = cost_usd


@dataclass
class DraftBudget:
    """Thread-safe call/spend guard shared by all frontier phases in one run."""

    max_calls: int | None = None
    max_usd: float | None = None
    calls: int = 0
    cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def reserve_call(self) -> None:
        with self._lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise DraftBudgetExceeded(
                    f"frontier call budget exhausted: {self.calls} >= {self.max_calls}",
                    calls=self.calls,
                    cost_usd=self.cost_usd,
                )
            if self.max_usd is not None and self.cost_usd >= self.max_usd:
                raise DraftBudgetExceeded(
                    f"frontier spend budget exhausted: ${self.cost_usd:.6f} >= ${self.max_usd:.6f}",
                    calls=self.calls,
                    cost_usd=self.cost_usd,
                )
            self.calls += 1

    def record_cost(self, cost_usd: float) -> None:
        with self._lock:
            self.cost_usd += cost_usd
            if self.max_usd is not None and self.cost_usd > self.max_usd:
                raise DraftBudgetExceeded(
                    f"frontier spend budget exceeded: ${self.cost_usd:.6f} > ${self.max_usd:.6f}",
                    calls=self.calls,
                    cost_usd=self.cost_usd,
                )


def budgeted_complete(
    complete: LLMComplete, log: ProvenanceLog, budget: DraftBudget
) -> LLMComplete:
    """Guard a logging completion callable with shared call and measured-spend limits."""

    def guarded(prompt: str) -> str:
        budget.reserve_call()
        before = int(log.summary()["calls"])
        try:
            text = complete(prompt)
        except DraftBudgetExceeded:
            raise
        except Exception:
            if int(log.summary()["calls"]) == before:
                log.record("injected-completer", 0, 0, 0.0, error="completion failed")
            raise
        records = log.summary()["call_records"]
        call_cost = float(records[-1]["cost_usd"]) if len(records) > before else 0.0
        budget.record_cost(call_cost)
        return text

    return guarded
