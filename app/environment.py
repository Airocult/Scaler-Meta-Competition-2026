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
from app.scenarios.task4_dns_failure import DNSFailureScenario
from app.scenarios.task5_cert_expiry import CertExpiryScenario
from app.scenarios.task6_network_partition import NetworkPartitionScenario
from app.scenarios.task7_kafka_lag import KafkaLagScenario
from app.scenarios.task8_redis_failover import RedisFailoverScenario
from app.scenarios.task9_disk_full import DiskFullScenario
from app.scenarios.task10_rate_limit import RateLimitScenario
from app.scenarios.task11_db_migration_lock import DbMigrationLockScenario
from app.scenarios.task12_health_flap import HealthFlapScenario
from app.scenarios.task13_pod_eviction import PodEvictionScenario
from app.scenarios.task14_cascading_timeout import CascadingTimeoutScenario
from app.scenarios.task15_secret_rotation import SecretRotationScenario
from app.scenarios.task16_log_storm import LogStormScenario


SCENARIO_MAP: dict[str, type[BaseScenario]] = {
    "task1_memory_leak": MemoryLeakScenario,
    "task2_db_cascade": DBCascadeScenario,
    "task3_race_condition": RaceConditionScenario,
    "task4_dns_failure": DNSFailureScenario,
    "task5_cert_expiry": CertExpiryScenario,
    "task6_network_partition": NetworkPartitionScenario,
    "task7_kafka_lag": KafkaLagScenario,
    "task8_redis_failover": RedisFailoverScenario,
    "task9_disk_full": DiskFullScenario,
    "task10_rate_limit": RateLimitScenario,
    "task11_db_migration_lock": DbMigrationLockScenario,
    "task12_health_flap": HealthFlapScenario,
    "task13_pod_eviction": PodEvictionScenario,
    "task14_cascading_timeout": CascadingTimeoutScenario,
    "task15_secret_rotation": SecretRotationScenario,
    "task16_log_storm": LogStormScenario,
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
        if task_id is None:
            task_id = "task1_memory_leak"  # default task for validation
        if task_id not in SCENARIO_MAP:
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
        obs.episode_score = self._scenario.get_grader_score()
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
            severity_classified=s["severity_classified"],
            status_page_updated=s["status_page_updated"],
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
