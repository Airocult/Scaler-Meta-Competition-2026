#!/usr/bin/env python3
"""
Example: Run SREBench episodes using the OpenEnv WebSocket client.
This script demonstrates the interaction loop that torchforge's GRPO
trainer uses internally.

Usage:
    # Start server first:  uvicorn app.main:app --host 0.0.0.0 --port 7860
    python examples/torchforge_grpo/run_episode.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.models import SREAction


# Demonstration policy — a hardcoded optimal sequence for task1
TASK1_OPTIMAL_ACTIONS = [
    SREAction(action_type="list_services", reasoning="Survey all service statuses"),
    SREAction(
        action_type="read_logs",
        parameters={"service": "order-service"},
        reasoning="order-service is down, check logs for root cause",
    ),
    SREAction(
        action_type="check_metrics",
        parameters={"service": "order-service", "metric": "memory"},
        reasoning="Logs show OOM kills, verify with memory metrics",
    ),
    SREAction(
        action_type="apply_fix",
        parameters={"service": "order-service", "fix_type": "restart"},
        reasoning="Confirmed memory leak causing OOM, restart to resolve",
    ),
    SREAction(
        action_type="verify_health",
        parameters={"service": "order-service"},
        reasoning="Verify the service is healthy after restart",
    ),
]


async def run_episode_http():
    """Run an episode using HTTP endpoints (works without WebSocket)."""
    import httpx

    base_url = os.getenv("SREBENCH_URL", "http://localhost:7860")

    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        # Reset
        resp = await client.post("/reset", json={"task_id": "task1_memory_leak", "seed": 42})
        data = resp.json()
        obs = data["observation"]
        print(f"Episode started: {obs['alert_summary']}")
        print(f"Services: {list(obs['service_statuses'].keys())}")

        # Step through optimal actions
        for i, action in enumerate(TASK1_OPTIMAL_ACTIONS):
            resp = await client.post("/step", json={"action": action.model_dump()})
            data = resp.json()
            obs = data["observation"]
            reward = data["reward"]
            done = data["done"]

            print(f"\nStep {i+1}: {action.action_type}")
            print(f"  Result: {obs['last_action_result'][:80]}...")
            print(f"  Reward: {reward}, Done: {done}")

            if done:
                print(f"\n  Episode complete! Final score: {obs.get('episode_score', 'N/A')}")
                break

        # Get grader score
        resp = await client.get("/grader")
        score = resp.json()["episode_score"]
        print(f"\nGrader score: {score:.4f}")


if __name__ == "__main__":
    asyncio.run(run_episode_http())
