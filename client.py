"""
SREBench OpenEnv client — persistent WebSocket session for RL training loops.

Usage:
    async with SREBenchEnv(base_url="ws://localhost:7860") as env:
        result = await env.reset(task_id="task1_memory_leak", seed=42)
        while not result.done:
            action = SREAction(action_type="list_services", reasoning="investigating")
            result = await env.step(action)

Sync wrapper:
    env = SREBenchEnv(base_url="ws://localhost:7860").sync()
    with env:
        result = env.reset(task_id="task1_memory_leak", seed=42)
        result = env.step(action)
"""
from openenv.core.env_client import EnvClient

from app.models import SREAction, SREObservation, SREState


class SREBenchEnv(EnvClient[SREAction, SREObservation, SREState]):
    """
    OpenEnv WebSocket client for SREBench.

    Maintains a persistent session with the environment server.
    Supports both async (default) and sync interfaces.
    """
    pass
