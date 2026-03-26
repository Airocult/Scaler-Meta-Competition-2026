"""
Abstract base class for all scenarios.
"""
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.data.service_graph import ServiceGraph
from app.reward import RewardShaper


class BaseScenario(ABC):
    task_id: str
    max_steps: int

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.step_count = 0
        self.incident_phase = IncidentPhase.INVESTIGATING
        self.cumulative_reward = 0.0
        self.hints_used = 0
        self.done = False
        self.graph = ServiceGraph()
        self.reward_shaper = RewardShaper()
        self._last_action: str | None = None
        self._last_params: dict | None = None
        self._action_history: list[dict] = []

        # Flags common to all scenarios
        self._root_cause_identified = False
        self._fix_applied = False
        self._resolution_verified = False
        self._postmortem_written = False

    def _base_timestamp(self) -> str:
        """Incident start time."""
        return "2026-03-26T03:00:00Z"

    def _current_timestamp(self) -> str:
        base = datetime(2026, 3, 26, 3, 0, 0)
        current = base + timedelta(minutes=self.step_count * 2)
        return current.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _available_actions(self) -> list[str]:
        return [
            "read_logs", "check_metrics", "list_services", "check_alerts",
            "check_deployments", "check_dependencies", "run_diagnostic",
            "apply_fix", "verify_health", "write_postmortem", "escalate",
        ]

    def _build_observation(self, last_action_result: str) -> Observation:
        return Observation(
            timestamp=self._current_timestamp(),
            alert_summary=self._get_alert_summary(),
            service_statuses=self._get_service_statuses(),
            last_action_result=last_action_result,
            incident_phase=self.incident_phase,
            available_actions=self._available_actions(),
            step_count=self.step_count,
            time_elapsed_minutes=self.step_count * 2,
            hints_used=self.hints_used,
        )

    def _compute_reward(self, event: str) -> float:
        reward = self.reward_shaper.compute(
            event=event,
            step_count=self.step_count,
            max_steps=self.max_steps,
            previous_reward=self.cumulative_reward,
        )
        self.cumulative_reward += reward
        return reward

    def _is_repeated_action(self, action: Action) -> bool:
        if (self._last_action == action.action_type
                and self._last_params == action.parameters):
            return True
        return False

    def apply_action(self, action: Action) -> tuple[Observation, float, bool]:
        """Process an action. Returns (observation, step_reward, done)."""
        if self.done:
            obs = self._build_observation("Episode already completed.")
            return obs, 0.0, True

        self.step_count += 1

        # Check for repeated action
        is_repeat = self._is_repeated_action(action)
        self._last_action = action.action_type
        self._last_params = action.parameters.copy()
        self._action_history.append({
            "action_type": action.action_type,
            "parameters": action.parameters.copy(),
            "step": self.step_count,
        })

        if is_repeat:
            repeat_penalty = self._compute_reward("repeated_same_action")
            obs = self._build_observation("Same action repeated — no new information.")
            # Check max steps
            if self.step_count >= self.max_steps:
                self.done = True
            return obs, repeat_penalty, self.done

        # Delegate to scenario-specific handler
        obs, reward, done = self._handle_action(action)
        self.done = done

        # Check max steps
        if self.step_count >= self.max_steps and not self.done:
            self.done = True

        return obs, reward, self.done

    def get_current_state(self) -> dict:
        return {
            "task_id": self.task_id,
            "step_count": self.step_count,
            "incident_phase": self.incident_phase.value,
            "done": self.done,
            "cumulative_reward": self.cumulative_reward,
            "hints_used": self.hints_used,
            "root_cause_identified": self._root_cause_identified,
            "fix_applied": self._fix_applied,
            "resolution_verified": self._resolution_verified,
            "postmortem_written": self._postmortem_written,
        }

    @abstractmethod
    def get_initial_observation(self) -> Observation:
        ...

    @abstractmethod
    def _handle_action(self, action: Action) -> tuple[Observation, float, bool]:
        """Returns: (observation, step_reward, done)"""
        ...

    @abstractmethod
    def _get_alert_summary(self) -> str:
        ...

    @abstractmethod
    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        ...

    @abstractmethod
    def get_grader_score(self) -> float:
        ...
