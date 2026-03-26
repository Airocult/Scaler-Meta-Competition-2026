"""
Reward shaper — converts raw scenario signals into step-level rewards.
Ensures the reward function has meaningful gradient everywhere —
not just binary at episode end.
"""


class RewardShaper:
    BASE_STEP_PENALTY = -0.01  # small cost per step (time pressure)

    REWARDS: dict[str, float] = {
        # Positive
        "root_cause_progress": +0.15,
        "root_cause_identified": +0.30,
        "fix_applied_correctly": +0.30,
        "resolution_verified": +0.15,
        "postmortem_written": +0.10,
        "postmortem_quality": +0.05,
        # Negative
        "wrong_fix_applied": -0.10,
        "destructive_action": -0.20,
        "escalate_used": -0.05,
        "repeated_same_action": -0.05,
        # Neutral/minor
        "info_gathered": +0.02,
        "no_effect": -0.02,
    }

    def compute(
        self,
        event: str,
        step_count: int,
        max_steps: int,
        previous_reward: float,
    ) -> float:
        base = self.BASE_STEP_PENALTY
        event_reward = self.REWARDS.get(event, 0.0)
        return round(base + event_reward, 4)
