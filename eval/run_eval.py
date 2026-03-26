#!/usr/bin/env python3
"""
Comprehensive SREBench Evaluation Script
Tests all 3 tasks across multiple evaluation paths:
  - Optimal path (maximum score)
  - Suboptimal paths (partial investigation)
  - Anti-patterns / edge cases (wrong service, repeated actions, escalation)
  - Boundary conditions (max steps, empty postmortem, no reset)
  - OpenEnv endpoint compliance
  - Reproducibility across seeds
"""
import asyncio
import json
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from httpx import AsyncClient, ASGITransport
from app.main import app
from app.environment import SREBenchEnvironment
from app.models import SREAction

transport = ASGITransport(app=app)

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

results = []


def record(category: str, name: str, passed: bool, details: str = ""):
    status = PASS if passed else FAIL
    results.append({"category": category, "name": name, "passed": passed, "details": details})
    print(f"  {status} {name}" + (f"  ({details})" if details else ""))


async def reset_task(client, task_id, seed=42):
    resp = await client.post("/reset", json={"task_id": task_id, "seed": seed})
    return resp.status_code, resp.json()


async def do_step(client, action_type, parameters=None, reasoning="eval"):
    resp = await client.post("/step", json={
        "action": {
            "action_type": action_type,
            "parameters": parameters or {},
            "reasoning": reasoning,
        }
    })
    return resp.status_code, resp.json()


async def get_grader(client):
    resp = await client.get("/grader")
    return resp.json()["episode_score"]


# ═══════════════════════════════════════════════════════════
# TASK 1: Memory Leak OOM Kill (Easy)
# ═══════════════════════════════════════════════════════════

async def eval_task1_optimal(client):
    """Optimal path: identify → fix → verify → postmortem."""
    await reset_task(client, "task1_memory_leak")
    await do_step(client, "list_services")
    await do_step(client, "read_logs", {"service": "order-service"})
    await do_step(client, "check_metrics", {"service": "order-service", "metric": "memory"})
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "restart"})
    code, data = await do_step(client, "verify_health", {"service": "order-service"})
    score = await get_grader(client)
    record("Task1", "Optimal path", data["done"] and score >= 0.85, f"score={score:.4f}, done={data['done']}")
    return score


async def eval_task1_minimal(client):
    """Minimal path: skip investigation, go straight to fix + verify."""
    await reset_task(client, "task1_memory_leak")
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "restart"})
    code, data = await do_step(client, "verify_health", {"service": "order-service"})
    score = await get_grader(client)
    record("Task1", "Minimal path (skip investigation)", data["done"] and score >= 0.70, f"score={score:.4f}")
    return score


async def eval_task1_with_postmortem(client):
    """Optimal + postmortem for maximum score."""
    await reset_task(client, "task1_memory_leak")
    await do_step(client, "check_metrics", {"service": "order-service", "metric": "memory"})
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "restart"})
    await do_step(client, "verify_health", {"service": "order-service"})
    code, data = await do_step(client, "write_postmortem",
                               {"content": "Root cause: order-service OOM Kill due to memory leak. Fixed by restarting the service. Memory usage was at 98% before restart."})
    score = await get_grader(client)
    record("Task1", "With postmortem (max score)", score >= 0.85, f"score={score:.4f}")
    return score


async def eval_task1_wrong_service(client):
    """Fix wrong service — should get penalty."""
    await reset_task(client, "task1_memory_leak")
    code, data = await do_step(client, "apply_fix", {"service": "auth-service", "fix_type": "restart"})
    reward = data["reward"]
    score = await get_grader(client)
    record("Task1", "Wrong service penalty", reward <= -0.05, f"reward={reward:.4f}, grader={score:.4f}")
    return score


async def eval_task1_repeated_action(client):
    """Repeated same action — should get penalty."""
    await reset_task(client, "task1_memory_leak")
    code1, data1 = await do_step(client, "list_services")
    code2, data2 = await do_step(client, "list_services")
    record("Task1", "Repeated action penalty", data2["reward"] < data1["reward"],
           f"first={data1['reward']:.4f}, repeat={data2['reward']:.4f}")


async def eval_task1_escalation(client):
    """Escalation hint — costs points."""
    await reset_task(client, "task1_memory_leak")
    code, data = await do_step(client, "escalate")
    record("Task1", "Escalation penalty", data["reward"] < 0, f"reward={data['reward']:.4f}")
    # Now fix correctly
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "restart"})
    await do_step(client, "verify_health", {"service": "order-service"})
    score = await get_grader(client)
    record("Task1", "Score after escalation + fix", 0.5 < score < 0.90, f"score={score:.4f}")


async def eval_task1_max_steps(client):
    """Exhaust max steps without resolving."""
    await reset_task(client, "task1_memory_leak")
    done = False
    steps = 0
    for i in range(25):
        code, data = await do_step(client, "check_alerts")
        steps += 1
        if data["done"]:
            done = True
            break
    score = await get_grader(client)
    record("Task1", "Max steps exhaustion", done and steps <= 20, f"steps={steps}, done={done}, score={score:.4f}")


async def eval_task1_verify_before_fix(client):
    """Verify health before applying fix — no effect."""
    await reset_task(client, "task1_memory_leak")
    code, data = await do_step(client, "verify_health", {"service": "order-service"})
    record("Task1", "Verify before fix (no effect)", data["reward"] < 0 and not data["done"],
           f"reward={data['reward']:.4f}, done={data['done']}")


# ═══════════════════════════════════════════════════════════
# TASK 2: Cascading DB Pool Exhaustion (Medium)
# ═══════════════════════════════════════════════════════════

async def eval_task2_optimal(client):
    """Trace full dependency chain → fix payment-db pool → verify."""
    await reset_task(client, "task2_db_cascade")
    await do_step(client, "list_services")
    await do_step(client, "check_dependencies", {"service": "api-gateway"})
    await do_step(client, "read_logs", {"service": "payment-service"})
    await do_step(client, "check_metrics", {"service": "payment-db"})
    await do_step(client, "run_diagnostic", {"service": "payment-db", "type": "connection_pool"})
    await do_step(client, "apply_fix", {"service": "payment-db", "fix_type": "increase_pool_size"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task2", "Optimal path (full trace)", data["done"] and score >= 0.80, f"score={score:.4f}")
    return score


async def eval_task2_with_postmortem(client):
    """Optimal + postmortem."""
    await reset_task(client, "task2_db_cascade")
    await do_step(client, "read_logs", {"service": "payment-service"})
    await do_step(client, "check_metrics", {"service": "payment-db"})
    await do_step(client, "apply_fix", {"service": "payment-db", "fix_type": "increase_pool_size"})
    await do_step(client, "verify_health")
    code, data = await do_step(client, "write_postmortem",
                               {"content": "Root cause: payment-db connection pool exhausted at 198/200 connections. Cascading failures through payment-service, order-service, api-gateway. Fixed by increasing pool size."})
    score = await get_grader(client)
    record("Task2", "With postmortem (max score)", score >= 0.85, f"score={score:.4f}")
    return score


async def eval_task2_surface_fix_trap(client):
    """Fix api-gateway (surface symptom) — grader caps at 0.35."""
    await reset_task(client, "task2_db_cascade")
    await do_step(client, "apply_fix", {"service": "api-gateway", "fix_type": "restart"})
    score = await get_grader(client)
    record("Task2", "Surface fix trap (cap 0.35)", score <= 0.35, f"score={score:.4f}")
    return score


async def eval_task2_partial_trace(client):
    """Trace to payment-service but not payment-db."""
    await reset_task(client, "task2_db_cascade")
    await do_step(client, "list_services")
    await do_step(client, "read_logs", {"service": "payment-service"})
    await do_step(client, "check_metrics", {"service": "payment-service"})
    await do_step(client, "apply_fix", {"service": "payment-service", "fix_type": "restart"})
    score = await get_grader(client)
    record("Task2", "Partial trace (payment-service only)", 0.10 <= score <= 0.50, f"score={score:.4f}")


async def eval_task2_drain_pool(client):
    """Alternative fix: drain_pool instead of increase_pool_size."""
    await reset_task(client, "task2_db_cascade")
    await do_step(client, "check_metrics", {"service": "payment-db"})
    await do_step(client, "apply_fix", {"service": "payment-db", "fix_type": "drain_pool"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task2", "Alt fix: drain_pool", data["done"] and score >= 0.60, f"score={score:.4f}")


async def eval_task2_wrong_then_correct(client):
    """Fix wrong surface first, then correct fix — cap still applies."""
    await reset_task(client, "task2_db_cascade")
    await do_step(client, "apply_fix", {"service": "api-gateway", "fix_type": "restart"})
    await do_step(client, "check_metrics", {"service": "payment-db"})
    await do_step(client, "apply_fix", {"service": "payment-db", "fix_type": "increase_pool_size"})
    await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task2", "Wrong surface then correct (redeemed)", score >= 0.60, f"score={score:.4f}")


# ═══════════════════════════════════════════════════════════
# TASK 3: Race Condition via Config Change (Hard)
# ═══════════════════════════════════════════════════════════

async def eval_task3_optimal(client):
    """Full optimal: notice spike → deploy → config diff → rollback → verify → postmortem."""
    await reset_task(client, "task3_race_condition")
    await do_step(client, "check_metrics", {"service": "inventory-service", "metric": "error_rate"})
    await do_step(client, "check_deployments", {"service": "inventory-service"})
    await do_step(client, "run_diagnostic", {"service": "inventory-service", "type": "config_diff", "deploy_id": "deploy-a1b2c3"})
    await do_step(client, "apply_fix", {"service": "inventory-service", "fix_type": "rollback", "deploy_id": "deploy-a1b2c3"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task3", "Optimal path", data["done"] and score >= 0.80, f"score={score:.4f}")
    return score


async def eval_task3_with_postmortem(client):
    """Optimal + quality postmortem mentioning root cause."""
    await reset_task(client, "task3_race_condition")
    await do_step(client, "check_metrics", {"service": "inventory-service", "metric": "error_rate"})
    await do_step(client, "check_deployments", {"service": "inventory-service"})
    await do_step(client, "run_diagnostic", {"service": "inventory-service", "type": "config_diff", "deploy_id": "deploy-a1b2c3"})
    await do_step(client, "apply_fix", {"service": "inventory-service", "fix_type": "rollback", "deploy_id": "deploy-a1b2c3"})
    await do_step(client, "verify_health")
    code, data = await do_step(client, "write_postmortem",
                               {"content": "Root cause: inventory-service deploy changed redis lock_timeout_ms from 5000 to 500, causing race condition under load. Fixed by rolling back the config change."})
    score = await get_grader(client)
    record("Task3", "With quality postmortem (max score)", score >= 0.80, f"score={score:.4f}")
    return score


async def eval_task3_restart_trap(client):
    """Restart >2 times without checking deploys — dead-end penalty."""
    await reset_task(client, "task3_race_condition")
    await do_step(client, "list_services")
    await do_step(client, "apply_fix", {"service": "inventory-service", "fix_type": "restart"})
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "restart"})
    await do_step(client, "apply_fix", {"service": "api-gateway", "fix_type": "restart"})
    score = await get_grader(client)
    record("Task3", "Restart trap (>2 restarts, -0.10)", score <= 0.15, f"score={score:.4f}")


async def eval_task3_no_deploy_check(client):
    """Skip deployment check entirely."""
    await reset_task(client, "task3_race_condition")
    await do_step(client, "list_services")
    await do_step(client, "check_metrics", {"service": "inventory-service", "metric": "error_rate"})
    await do_step(client, "apply_fix", {"service": "inventory-service", "fix_type": "restart"})
    score = await get_grader(client)
    record("Task3", "No deploy check", score <= 0.30, f"score={score:.4f}")


async def eval_task3_partial_investigation(client):
    """Notice spike + deploy but skip config diff and rollback wrong."""
    await reset_task(client, "task3_race_condition")
    await do_step(client, "check_metrics", {"service": "inventory-service", "metric": "error_rate"})
    await do_step(client, "check_deployments", {"service": "inventory-service"})
    await do_step(client, "apply_fix", {"service": "inventory-service", "fix_type": "restart"})
    score = await get_grader(client)
    record("Task3", "Partial investigation (no config diff)", 0.15 <= score <= 0.50, f"score={score:.4f}")


async def eval_task3_rollback_config(client):
    """Use rollback_config instead of rollback."""
    await reset_task(client, "task3_race_condition")
    await do_step(client, "check_deployments", {"service": "inventory-service"})
    await do_step(client, "apply_fix", {"service": "inventory-service", "fix_type": "rollback_config", "deploy_id": "deploy-a1b2c3"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task3", "Alt fix: rollback_config", data["done"] and score >= 0.50, f"score={score:.4f}")


# ═══════════════════════════════════════════════════════════
# OPENENV ENDPOINT COMPLIANCE
# ═══════════════════════════════════════════════════════════

async def eval_openenv_endpoints(client):
    """Verify all OpenEnv standard endpoints work correctly."""
    # /health
    resp = await client.get("/health")
    record("OpenEnv", "/health returns 200", resp.status_code == 200, f"status={resp.json().get('status')}")

    # /schema
    resp = await client.get("/schema")
    data = resp.json()
    record("OpenEnv", "/schema has action+observation", "action" in data and "observation" in data)

    # /metadata
    resp = await client.get("/metadata")
    data = resp.json()
    record("OpenEnv", "/metadata name=SREBench", data.get("name") == "SREBench")

    # /docs
    resp = await client.get("/docs")
    record("OpenEnv", "/docs (Swagger UI)", resp.status_code == 200)

    # /openapi.json
    resp = await client.get("/openapi.json")
    data = resp.json()
    record("OpenEnv", "/openapi.json valid", "paths" in data and "/reset" in data["paths"])

    # /reset response format
    resp = await client.post("/reset", json={"task_id": "task1_memory_leak", "seed": 42})
    data = resp.json()
    has_keys = all(k in data for k in ("observation", "reward", "done"))
    record("OpenEnv", "/reset format {observation, reward, done}", has_keys)

    # /step response format
    resp = await client.post("/step", json={"action": {"action_type": "list_services", "reasoning": "test"}})
    data = resp.json()
    has_keys = all(k in data for k in ("observation", "reward", "done"))
    record("OpenEnv", "/step format {observation, reward, done}", has_keys)

    # /state
    resp = await client.get("/state")
    record("OpenEnv", "/state returns dict", resp.status_code == 200 and isinstance(resp.json(), dict))

    # Custom endpoints
    resp = await client.get("/tasks")
    data = resp.json()
    record("OpenEnv", "/tasks returns 3 tasks", len(data.get("tasks", [])) == 3)

    resp = await client.get("/grader")
    score = resp.json().get("episode_score")
    record("OpenEnv", "/grader returns score", isinstance(score, (int, float)))


# ═══════════════════════════════════════════════════════════
# REPRODUCIBILITY TESTS
# ═══════════════════════════════════════════════════════════

async def eval_reproducibility(client):
    """Same seed produces identical results."""
    async def run_seq(seed):
        await reset_task(client, "task1_memory_leak", seed=seed)
        r1 = await do_step(client, "list_services")
        r2 = await do_step(client, "read_logs", {"service": "order-service"})
        return [r1[1], r2[1]]

    run_a = await run_seq(42)
    run_b = await run_seq(42)
    match = json.dumps(run_a, sort_keys=True) == json.dumps(run_b, sort_keys=True)
    record("Reproducibility", "seed=42 produces identical results", match)

    # Different seeds should vary (at minimum different timestamps aren't expected to differ,
    # but the structure should be consistent)
    await reset_task(client, "task1_memory_leak", seed=99)
    r99 = await do_step(client, "list_services")
    record("Reproducibility", "seed=99 runs without error", r99[0] == 200)


# ═══════════════════════════════════════════════════════════
# CROSS-TASK STATE ISOLATION
# ═══════════════════════════════════════════════════════════

async def eval_state_isolation(client):
    """Reset between tasks should fully isolate state."""
    # Start task1
    await reset_task(client, "task1_memory_leak")
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "restart"})

    # Switch to task2
    code, data = await reset_task(client, "task2_db_cascade")
    obs = data["observation"]
    record("Isolation", "Reset to task2 clears task1 state",
           obs["service_statuses"]["api-gateway"]["status"] == "degraded" and obs["step_count"] == 0)

    # Switch to task3
    code, data = await reset_task(client, "task3_race_condition")
    obs = data["observation"]
    record("Isolation", "Reset to task3 clears task2 state",
           obs["step_count"] == 0 and obs["incident_phase"] == "investigating")


# ═══════════════════════════════════════════════════════════
# ERROR HANDLING
# ═══════════════════════════════════════════════════════════

async def eval_error_handling(client):
    """Invalid inputs should return proper errors."""
    # Invalid task_id
    try:
        resp = await client.post("/reset", json={"task_id": "nonexistent", "seed": 42})
        record("Errors", "Invalid task_id returns error", resp.status_code >= 400,
               f"status={resp.status_code}")
    except Exception as e:
        record("Errors", "Invalid task_id raises exception", True,
               f"exception={type(e).__name__}")

    # Invalid action_type — environment handles it with a penalty, not 422
    try:
        await client.post("/reset", json={"task_id": "task1_memory_leak", "seed": 42})
        resp = await client.post("/step", json={"action": {"action_type": "fly_to_moon", "reasoning": "test"}})
        # Could return 200 with negative reward, or 422 from validation
        ok = resp.status_code in (200, 422)
        record("Errors", "Invalid action_type handled", ok,
               f"status={resp.status_code}")
    except Exception as e:
        record("Errors", "Invalid action_type raises exception", True,
               f"exception={type(e).__name__}")

    # Missing action field
    try:
        resp = await client.post("/step", json={"wrong_field": "test"})
        record("Errors", "Missing action field returns error", resp.status_code == 422,
               f"status={resp.status_code}")
    except Exception as e:
        record("Errors", "Missing action field raises exception", True,
               f"exception={type(e).__name__}")


# ═══════════════════════════════════════════════════════════
# REWARD SIGNAL ANALYSIS
# ═══════════════════════════════════════════════════════════

async def eval_reward_signals(client):
    """Verify reward signal characteristics across a full episode."""
    await reset_task(client, "task1_memory_leak")
    rewards = []
    actions = [
        ("list_services", {}),
        ("read_logs", {"service": "order-service"}),
        ("check_metrics", {"service": "order-service", "metric": "memory"}),
        ("check_alerts", {}),
        ("run_diagnostic", {"service": "order-service", "type": "general"}),
        ("apply_fix", {"service": "order-service", "fix_type": "restart"}),
        ("verify_health", {"service": "order-service"}),
    ]
    for act, params in actions:
        code, data = await do_step(client, act, params, f"Testing {act}")
        rewards.append((act, data["reward"], data["done"]))
        if data["done"]:
            break

    # Verify reward properties
    all_bounded = all(-0.25 <= r <= 0.35 for _, r, _ in rewards)
    record("Rewards", "All rewards in [-0.25, 0.35]", all_bounded,
           f"rewards={[(a, round(r, 4)) for a, r, _ in rewards]}")

    # Fix step should have positive reward
    fix_reward = [r for a, r, _ in rewards if a == "apply_fix"][0]
    record("Rewards", "apply_fix has positive reward", fix_reward > 0, f"fix_reward={fix_reward:.4f}")

    # Verify step should have positive reward
    verify_rewards = [r for a, r, _ in rewards if a == "verify_health"]
    if verify_rewards:
        record("Rewards", "verify_health has positive reward", verify_rewards[0] > 0,
               f"verify_reward={verify_rewards[0]:.4f}")

    # Episode should terminate
    record("Rewards", "Episode terminates", rewards[-1][2], f"last_done={rewards[-1][2]}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

async def main():
    print("=" * 72)
    print("  SREBench Complete Evaluation")
    print("=" * 72)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # ── Task 1 ─────────────────────────────────
        print("\n── Task 1: Memory Leak OOM Kill (Easy) ──")
        t1_optimal = await eval_task1_optimal(client)
        t1_minimal = await eval_task1_minimal(client)
        t1_max = await eval_task1_with_postmortem(client)
        await eval_task1_wrong_service(client)
        await eval_task1_repeated_action(client)
        await eval_task1_escalation(client)
        await eval_task1_max_steps(client)
        await eval_task1_verify_before_fix(client)

        # ── Task 2 ─────────────────────────────────
        print("\n── Task 2: Cascading DB Pool Exhaustion (Medium) ──")
        t2_optimal = await eval_task2_optimal(client)
        t2_max = await eval_task2_with_postmortem(client)
        await eval_task2_surface_fix_trap(client)
        await eval_task2_partial_trace(client)
        await eval_task2_drain_pool(client)
        await eval_task2_wrong_then_correct(client)

        # ── Task 3 ─────────────────────────────────
        print("\n── Task 3: Race Condition via Config Change (Hard) ──")
        t3_optimal = await eval_task3_optimal(client)
        t3_max = await eval_task3_with_postmortem(client)
        await eval_task3_restart_trap(client)
        await eval_task3_no_deploy_check(client)
        await eval_task3_partial_investigation(client)
        await eval_task3_rollback_config(client)

        # ── OpenEnv Compliance ─────────────────────
        print("\n── OpenEnv Endpoint Compliance ──")
        await eval_openenv_endpoints(client)

        # ── Reproducibility ────────────────────────
        print("\n── Reproducibility ──")
        await eval_reproducibility(client)

        # ── State Isolation ────────────────────────
        print("\n── State Isolation ──")
        await eval_state_isolation(client)

        # ── Error Handling ─────────────────────────
        print("\n── Error Handling ──")
        await eval_error_handling(client)

        # ── Reward Signals ─────────────────────────
        print("\n── Reward Signal Analysis ──")
        await eval_reward_signals(client)

    # ── Summary ────────────────────────────────────
    print("\n" + "=" * 72)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print(f"\n  RESULTS: {passed}/{total} passed, {failed} failed\n")

    if failed:
        print("  Failed tests:")
        for r in results:
            if not r["passed"]:
                print(f"    {FAIL} [{r['category']}] {r['name']}: {r['details']}")
        print()

    # Score summary
    print("  ── Score Summary ──")
    print(f"  Task 1 Optimal:        {t1_optimal:.4f}")
    print(f"  Task 1 Minimal:        {t1_minimal:.4f}")
    print(f"  Task 1 Max (w/postm):  {t1_max:.4f}")
    print(f"  Task 2 Optimal:        {t2_optimal:.4f}")
    print(f"  Task 2 Max (w/postm):  {t2_max:.4f}")
    print(f"  Task 3 Optimal:        {t3_optimal:.4f}")
    print(f"  Task 3 Max (w/postm):  {t3_max:.4f}")
    print()
    print(f"  Average optimal:       {(t1_optimal + t2_optimal + t3_optimal) / 3:.4f}")
    print(f"  Average max:           {(t1_max + t2_max + t3_max) / 3:.4f}")

    print("\n" + "=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
