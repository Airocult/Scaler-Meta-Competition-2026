"""
Core environment — extends OpenEnv Environment base class.
Stateful, thread-safe via asyncio.Lock. One active episode at a time.
"""
import asyncio
from typing import Optional, Any

from openenv.core.env_server.interfaces import Environment

from app.models import SREAction, SREObservation, SREState
from app.scenarios.base import BaseScenario
from app.scenarios.task1_memory_leak import MemoryLeakScenario
from app.scenarios.task2_db_cascade import DBCascadeScenario
from app.scenarios.task3_race_condition import RaceConditionScenario


SCENARIO_MAP: dict[str, type[BaseScenario]] = {
    "task1_memory_leak": MemoryLeakScenario,
    "task2_db_cascade": DBCascadeScenario,
    "task3_race_condition": RaceConditionScenario,
}


class SREBenchEnvironment(Environment[SREAction, SREObservation, SREState]):
    """
    OpenEnv-compliant SRE incident response environment.
    Stateful singleton — one active episode at a time, thread-safe via async lock.
    """

    def __init__(self):
        super().__init__()
        self._scenario: Optional[BaseScenario] = None
        self._lock = asyncio.Lock()

    # ── Sync interface (required by Environment ABC) ──────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SREObservation:
        if task_id is None or task_id not in SCENARIO_MAP:
            raise ValueError(
                f"Unknown task_id: {task_id}. Available: {list(SCENARIO_MAP.keys())}"
            )
        self._scenario = SCENARIO_MAP[task_id](seed=seed if seed is not None else 42)
        obs = self._scenario.get_initial_observation()
        obs.done = False
        obs.reward = 0.0
        return obs

    def step(
        self,
        action: SREAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SREObservation:
        if self._scenario is None:
            raise RuntimeError("Call reset() before step()")
        obs, step_reward, done = self._scenario.apply_action(action)
        obs.done = done
        obs.reward = step_reward
        obs.step_reward = step_reward
        obs.cumulative_reward = self._scenario.cumulative_reward
        obs.episode_score = self._scenario.get_grader_score() if done else 0.0
        return obs

    @property
    def state(self) -> SREState:
        if self._scenario is None:
            return SREState()
        s = self._scenario.get_current_state()
        return SREState(
            task_id=s["task_id"],
            step_count=s["step_count"],
            incident_phase=s["incident_phase"],
            done=s["done"],
            cumulative_reward=s["cumulative_reward"],
            hints_used=s["hints_used"],
            root_cause_identified=s["root_cause_identified"],
            fix_applied=s["fix_applied"],
            resolution_verified=s["resolution_verified"],
            postmortem_written=s["postmortem_written"],
        )

    # ── Async interface (overrides default sync-wrapping) ─────

    async def reset_async(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SREObservation:
        async with self._lock:
            return self.reset(seed=seed, episode_id=episode_id, task_id=task_id, **kwargs)

    async def step_async(
        self,
        action: SREAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SREObservation:
        async with self._lock:
            return self.step(action, timeout_s=timeout_s, **kwargs)

    # ── Custom SREBench methods ───────────────────────────────

    async def get_grader_score(self) -> float:
        if self._scenario is None:
            raise RuntimeError("No active episode")
        return self._scenario.get_grader_score()

    def get_metadata(self):
        from openenv.core.env_server.interfaces import EnvironmentMetadata
        return EnvironmentMetadata(
            name="SREBench",
            description="On-Call Incident Response RL Environment for SRE/DevOps agents",
            version="1.0.0",
        )
