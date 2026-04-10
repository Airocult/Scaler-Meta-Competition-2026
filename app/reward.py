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
    # Step penalty increases linearly: -0.003 at step 1 → -0.015 at max_steps
    BASE_STEP_PENALTY_MIN = -0.003
    BASE_STEP_PENALTY_MAX = -0.015

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

    def compute(
        self,
        event: str,
        step_count: int,
        max_steps: int,
        previous_reward: float,
        service: str = "",
    ) -> float:
        # Urgency-scaled step penalty: grows linearly with progress
        progress = step_count / max(max_steps, 1)
        base = self.BASE_STEP_PENALTY_MIN + (
            (self.BASE_STEP_PENALTY_MAX - self.BASE_STEP_PENALTY_MIN) * progress
        )

        # Information decay: same service queried multiple times
        if event == "info_gathered" and service:
            self._services_queried[service] = self._services_queried.get(service, 0) + 1
            query_count = self._services_queried[service]
            if query_count > 2:
                event = "info_gathered_redundant"

        event_reward = self.REWARDS.get(event, 0.0)
        return round(base + event_reward, 4)
