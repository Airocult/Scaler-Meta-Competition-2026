"""
Reward shaper — converts raw scenario signals into step-level rewards.

Design principles:
  1. **Urgency scaling**: step penalty grows as the episode progresses,
     creating real time-pressure that an agent must learn to manage.
  2. **Information decay**: repeated info-gathering on the *same* service
     yields diminishing returns, encouraging breadth-first investigation.
  3. **Phase-aware bonuses**: higher rewards for actions that advance the
     incident phase (investigating → mitigating → verifying → resolved).
  4. **Partial credit everywhere**: no step produces zero signal — even
     neutral actions return a small negative to keep the gradient alive.
"""


class RewardShaper:
    # Step penalty increases linearly with a grace period for initial orientation.
    # Grace period: first 3 steps have zero step penalty (let agents orient).
    # After grace: penalty grows from -0.002 to -0.012, capped at -0.08 cumulative.
    GRACE_STEPS = 3
    BASE_STEP_PENALTY_MIN = -0.002
    BASE_STEP_PENALTY_MAX = -0.012
    MAX_CUMULATIVE_STEP_PENALTY = -0.08

    REWARDS: dict[str, float] = {
        # Positive — milestone events
        "root_cause_progress": +0.15,
        "root_cause_identified": +0.30,
        "fix_applied_correctly": +0.30,
        "resolution_verified": +0.15,
        "postmortem_written": +0.10,
        "postmortem_quality": +0.05,
        # Negative — penalties
        "wrong_fix_applied": -0.08,
        "destructive_action": -0.20,
        "escalate_used": -0.05,
        "repeated_same_action": -0.06,
        # Neutral/minor
        "info_gathered": +0.02,
        "info_gathered_redundant": +0.005,
        "no_effect": -0.01,
    }

    def __init__(self):
        self._services_queried: dict[str, int] = {}
        self._cumulative_step_penalty: float = 0.0

    def compute(
        self,
        event: str,
        step_count: int,
        max_steps: int,
        previous_reward: float,
        service: str = "",
    ) -> float:
        # Grace period: first N steps have zero step penalty
        if step_count <= self.GRACE_STEPS:
            base = 0.0
        else:
            # Urgency-scaled step penalty after grace period
            effective_step = step_count - self.GRACE_STEPS
            effective_max = max(max_steps - self.GRACE_STEPS, 1)
            progress = effective_step / effective_max
            base = self.BASE_STEP_PENALTY_MIN + (
                (self.BASE_STEP_PENALTY_MAX - self.BASE_STEP_PENALTY_MIN) * progress
            )
            # Cap cumulative step penalty
            if self._cumulative_step_penalty + base < self.MAX_CUMULATIVE_STEP_PENALTY:
                base = max(self.MAX_CUMULATIVE_STEP_PENALTY - self._cumulative_step_penalty, base)
            self._cumulative_step_penalty += base

        # Information decay: same service queried multiple times
        # Allow 3 full-reward queries per service (was 2), then diminish
        if event == "info_gathered" and service:
            self._services_queried[service] = self._services_queried.get(service, 0) + 1
            query_count = self._services_queried[service]
            if query_count > 3:
                event = "info_gathered_redundant"

        event_reward = self.REWARDS.get(event, 0.0)
        return round(base + event_reward, 4)
