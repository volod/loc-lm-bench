"""Validated construction of ontology endpoint configurations."""

from dataclasses import dataclass

from llb.prep.ontology.endpoint_config import (
    DEFAULT_LOCAL_BASE_URL,
    ENDPOINT_FRONTIER,
    ENDPOINT_KINDS,
    ENDPOINT_LOCAL,
    LOCAL_BACKEND_OLLAMA,
    LOCAL_BACKENDS,
    EndpointConfig,
)


@dataclass(slots=True)
class EndpointConfigBuilder:
    """Collect endpoint values, validate them by concern, and build an immutable config."""

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

    def override_base_url(self, base_url: str | None) -> "EndpointConfigBuilder":
        if base_url is not None:
            self.base_url = base_url
        return self

    def override_max_tokens(self, max_tokens: int | None) -> "EndpointConfigBuilder":
        if max_tokens is not None:
            self.max_tokens = max_tokens
        return self

    def build(self) -> EndpointConfig:
        self._validate_identity()
        self._validate_routing()
        self._validate_budget_values()
        self._validate_frontier()
        self._validate_local()
        return EndpointConfig(
            kind=self.kind,
            model=self.model,
            backend=self.backend,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            think=self.think,
            num_ctx=self.num_ctx,
            egress_consent=self.egress_consent,
            max_usd=self.max_usd,
            max_calls=self.max_calls,
        )

    def _validate_identity(self) -> None:
        if self.kind not in ENDPOINT_KINDS:
            raise ValueError(f"endpoint kind must be one of {ENDPOINT_KINDS}, got {self.kind!r}")
        if not self.model:
            raise ValueError("endpoint model must be set")

    def _validate_routing(self) -> None:
        if self.backend not in LOCAL_BACKENDS:
            raise ValueError(f"local backend must be one of {LOCAL_BACKENDS}, got {self.backend!r}")
        if self.kind != ENDPOINT_LOCAL and self.backend != LOCAL_BACKEND_OLLAMA:
            raise ValueError("local backend can only be set when endpoint kind is local")

    def _validate_budget_values(self) -> None:
        if self.max_usd is not None and self.max_usd <= 0:
            raise ValueError("max_usd must be > 0 when set")
        if self.max_calls is not None and self.max_calls < 1:
            raise ValueError("max_calls must be >= 1 when set")

    def _validate_frontier(self) -> None:
        if self.kind != ENDPOINT_FRONTIER:
            return
        if not self.egress_consent:
            raise ValueError("frontier endpoint requires explicit egress consent")
        if self.max_usd is None and self.max_calls is None:
            raise ValueError("frontier endpoint requires --max-usd or --max-calls")

    def _validate_local(self) -> None:
        if self.kind != ENDPOINT_LOCAL:
            return
        if self.max_usd is not None or self.max_calls is not None:
            raise ValueError("frontier budgets can only be set when endpoint kind is frontier")
        if self.egress_consent:
            raise ValueError("egress consent can only be set when endpoint kind is frontier")
