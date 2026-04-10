"""
SLO / Error Budget tracker — each service has a target SLO and a
finite error budget that burns as errors accumulate.

Core SRE concept: the budget represents how much unreliability is
"allowed" before the SLO is breached. Faster resolution preserves
more budget; delayed fixes exhaust it.
"""
from __future__ import annotations


class ServiceSLO:
    """Tracks one service's SLO target and remaining error budget."""

    __slots__ = ("service", "target", "budget_remaining", "burn_rate",
                 "breached", "_baseline_error_rate")

    def __init__(self, service: str, target: float, baseline_error_rate: float = 0.01):
        self.service = service
        self.target = target                # e.g. 0.999 = 99.9%
        self.budget_remaining = 1.0         # 1.0 = 100% of budget
        self.burn_rate = 1.0                # 1.0 = normal
        self.breached = False
        self._baseline_error_rate = baseline_error_rate

    def advance(self, current_error_rate: float, step: int, max_steps: int) -> None:
        """Burn budget based on current error rate vs target."""
        allowed_error = 1.0 - self.target  # e.g. 0.001 for 99.9%
        if allowed_error <= 0:
            return

        excess = max(0.0, current_error_rate - allowed_error)
        if excess > 0:
            self.burn_rate = round(min(excess / allowed_error, 50.0), 1)
            burn_amount = (excess / allowed_error) * (1.0 / max(max_steps, 1))
            self.budget_remaining = round(max(0.0, self.budget_remaining - burn_amount), 4)
        else:
            self.burn_rate = 1.0

        if self.budget_remaining <= 0:
            self.breached = True

    def status_line(self) -> str:
        """Human-readable one-liner."""
        pct = self.budget_remaining * 100
        if self.breached:
            return (f"{self.service}: SLO {self.target*100:.2f}% BREACHED "
                    f"— error budget exhausted (burn rate: {self.burn_rate}x)")
        if pct < 30:
            return (f"{self.service}: SLO {self.target*100:.2f}% — "
                    f"error budget {pct:.0f}% ⚠️ CRITICAL (burn rate: {self.burn_rate}x)")
        if pct < 60:
            return (f"{self.service}: SLO {self.target*100:.2f}% — "
                    f"error budget {pct:.0f}% (burn rate: {self.burn_rate}x)")
        return (f"{self.service}: SLO {self.target*100:.2f}% — "
                f"error budget {pct:.0f}% OK")


# Default SLO targets per service
DEFAULT_SLOS: dict[str, float] = {
    "api-gateway": 0.999,
    "auth-service": 0.999,
    "order-service": 0.9995,
    "inventory-service": 0.999,
    "payment-service": 0.9999,
    "user-db": 0.9999,
    "inventory-db": 0.9999,
    "payment-db": 0.9999,
}


def create_slo_tracker() -> dict[str, ServiceSLO]:
    """Instantiate fresh SLO objects for all services."""
    return {svc: ServiceSLO(svc, target) for svc, target in DEFAULT_SLOS.items()}
