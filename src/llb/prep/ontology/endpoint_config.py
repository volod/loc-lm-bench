"""Phase-aware endpoint configuration and provenance for ontology drafting."""

from dataclasses import dataclass, field

from llb.core.config import DEFAULT_OLLAMA_HOST
from llb.prep.frontier_telemetry import LLMComplete, ProvenanceLog

ENDPOINT_LOCAL = "local"
ENDPOINT_FRONTIER = "frontier"
ENDPOINT_KINDS = (ENDPOINT_LOCAL, ENDPOINT_FRONTIER)
LOCAL_BACKEND_OLLAMA = "ollama"
LOCAL_BACKEND_VLLM = "vllm"
LOCAL_BACKEND_OPENAI = "openai"
LOCAL_BACKENDS = (LOCAL_BACKEND_OLLAMA, LOCAL_BACKEND_VLLM, LOCAL_BACKEND_OPENAI)
PHASE_EXTRACTION = "extraction"
PHASE_DRAFTING = "drafting"
DEFAULT_LOCAL_BASE_URL = f"{DEFAULT_OLLAMA_HOST}/v1"


@dataclass(frozen=True)
class EndpointConfig:
    """One local or explicitly consented frontier endpoint."""

    kind: str = ENDPOINT_LOCAL
    model: str = ""
    backend: str = LOCAL_BACKEND_OLLAMA
    base_url: str = DEFAULT_LOCAL_BASE_URL
    api_key: str = "not-needed"
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: float = 120.0
    think: bool | None = None
    num_ctx: int | None = None
    egress_consent: bool = False
    max_usd: float | None = None
    max_calls: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ENDPOINT_KINDS:
            raise ValueError(f"endpoint kind must be one of {ENDPOINT_KINDS}, got {self.kind!r}")
        if not self.model:
            raise ValueError("endpoint model must be set")
        if self.backend not in LOCAL_BACKENDS:
            raise ValueError(f"local backend must be one of {LOCAL_BACKENDS}, got {self.backend!r}")
        if self.kind != ENDPOINT_LOCAL and self.backend != LOCAL_BACKEND_OLLAMA:
            raise ValueError("local backend can only be set when endpoint kind is local")
        if self.kind == ENDPOINT_FRONTIER and not self.egress_consent:
            raise ValueError("frontier endpoint requires explicit egress consent")
        if self.kind == ENDPOINT_FRONTIER and self.max_usd is None and self.max_calls is None:
            raise ValueError("frontier endpoint requires --max-usd or --max-calls")
        if self.max_usd is not None and self.max_usd <= 0:
            raise ValueError("max_usd must be > 0 when set")
        if self.max_calls is not None and self.max_calls < 1:
            raise ValueError("max_calls must be >= 1 when set")
        if self.kind == ENDPOINT_LOCAL and (self.max_usd is not None or self.max_calls is not None):
            raise ValueError("frontier budgets can only be set when endpoint kind is frontier")
        if self.kind == ENDPOINT_LOCAL and self.egress_consent:
            raise ValueError("egress consent can only be set when endpoint kind is frontier")

    @property
    def egress(self) -> bool:
        return self.kind == ENDPOINT_FRONTIER

    def provenance(self) -> dict[str, object]:
        record: dict[str, object] = {
            "kind": self.kind,
            "model": self.model,
            "egress": self.egress,
        }
        if self.kind == ENDPOINT_LOCAL:
            record.update({"backend": self.backend, "base_url": self.base_url})
        if self.think is not None:
            record["think"] = self.think
        if self.num_ctx is not None:
            record["num_ctx"] = self.num_ctx
        if self.kind == ENDPOINT_FRONTIER:
            record.update(
                {
                    "egress_consent": self.egress_consent,
                    "max_usd": self.max_usd,
                    "max_calls": self.max_calls,
                }
            )
        return record


@dataclass(frozen=True)
class EndpointPlan:
    extraction: EndpointConfig
    drafting: EndpointConfig

    @classmethod
    def single(cls, config: EndpointConfig) -> "EndpointPlan":
        return cls(extraction=config, drafting=config)

    @property
    def egress(self) -> bool:
        return self.extraction.egress or self.drafting.egress

    def config_provenance(self) -> dict[str, object]:
        return {
            "egress": self.egress,
            "stages": {
                PHASE_EXTRACTION: self.extraction.provenance(),
                PHASE_DRAFTING: self.drafting.provenance(),
            },
        }


@dataclass
class EndpointLogs:
    extraction: ProvenanceLog = field(default_factory=ProvenanceLog)
    drafting: ProvenanceLog = field(default_factory=ProvenanceLog)

    def summary(self) -> dict[str, object]:
        phases = {
            PHASE_EXTRACTION: self.extraction.summary(),
            PHASE_DRAFTING: self.drafting.summary(),
        }
        records = [
            {"phase": phase, **record}
            for phase, summary in phases.items()
            for record in summary["call_records"]
        ]
        calls = sum(int(summary["calls"]) for summary in phases.values())
        latency = sum(float(summary["total_latency_s"]) for summary in phases.values())
        return {
            "calls": calls,
            "cost_usd": round(sum(float(s["total_cost_usd"]) for s in phases.values()), 6),
            "latency_s": round(latency, 3),
            "average_latency_s": round(latency / calls, 3) if calls else 0.0,
            "call_telemetry": records,
            "stages": phases,
        }


@dataclass(frozen=True)
class EndpointCompleters:
    extraction: LLMComplete
    drafting: LLMComplete

    @classmethod
    def single(cls, complete: LLMComplete) -> "EndpointCompleters":
        return cls(extraction=complete, drafting=complete)


def endpoint_provenance(plan: EndpointPlan, logs: EndpointLogs) -> dict[str, object]:
    configured = plan.config_provenance()["stages"]
    measured = logs.summary()
    metrics = measured["stages"]
    assert isinstance(configured, dict) and isinstance(metrics, dict)
    return {
        "egress": plan.egress,
        "calls": measured["calls"],
        "cost_usd": measured["cost_usd"],
        "latency_s": measured["latency_s"],
        "average_latency_s": measured["average_latency_s"],
        "call_telemetry": measured["call_telemetry"],
        "stages": {
            phase: {**configured[phase], **metrics[phase]}
            for phase in (PHASE_EXTRACTION, PHASE_DRAFTING)
        },
    }
