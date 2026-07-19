"""Exceptions raised by the scorer-policy seam."""


class ScorerPolicyError(ValueError):
    """Invalid scorer-policy configuration or consent state."""


class BudgetExceeded(RuntimeError):
    """Frontier scoring stopped at a configured call or spend cap."""

    def __init__(self, reason: str, *, calls: int, cost_usd: float):
        super().__init__(reason)
        self.reason = reason
        self.calls = calls
        self.cost_usd = cost_usd
