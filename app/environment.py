"""
Core environment — stateful, thread-safe via asyncio.Lock.
One active episode at a time.
"""
import asyncio
from typing import Optional

from app.models import Action, Observation, Reward
from app.scenarios.base import BaseScenario
from app.scenarios.task1_memory_leak import MemoryLeakScenario
from app.scenarios.task2_db_cascade import DBCascadeScenario
from app.scenarios.task3_race_condition import RaceConditionScenario


SCENARIO_MAP: dict[str, type[BaseScenario]] = {
    "task1_memory_leak": MemoryLeakScenario,
    "task2_db_cascade": DBCascadeScenario,
    "task3_race_condition": RaceConditionScenario,
}


class SREBenchEnvironment:
    """
    Stateful environment. One instance per active episode.
    Thread-safe via a lock — supports concurrent API calls.
    """

    def __init__(self):
        self._scenario: Optional[BaseScenario] = None
        self._lock = asyncio.Lock()

    async def reset(self, task_id: str, seed: int = 42) -> Observation:
        async with self._lock:
            if task_id not in SCENARIO_MAP:
                raise ValueError(f"Unknown task_id: {task_id}. Available: {list(SCENARIO_MAP.keys())}")
            self._scenario = SCENARIO_MAP[task_id](seed=seed)
            return self._scenario.get_initial_observation()

    async def step(self, action: Action) -> tuple[Observation, Reward]:
        async with self._lock:
            if self._scenario is None:
                raise RuntimeError("Call reset() before step()")
            obs, step_reward, done = self._scenario.apply_action(action)
            reward = Reward(
                step_reward=step_reward,
                cumulative_reward=self._scenario.cumulative_reward,
                episode_score=self._scenario.get_grader_score() if done else 0.0,
                done=done,
                info={
                    "step_count": obs.step_count,
                    "incident_phase": obs.incident_phase.value,
                    "task_id": self._scenario.task_id,
                },
            )
            return obs, reward

    async def state(self) -> dict:
        if self._scenario is None:
            return {"status": "no_episode_active"}
        return self._scenario.get_current_state()

    async def get_grader_score(self) -> float:
        if self._scenario is None:
            raise RuntimeError("No active episode")
        return self._scenario.get_grader_score()
