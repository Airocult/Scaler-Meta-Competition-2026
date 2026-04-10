#!/usr/bin/env python3
"""
Comprehensive SREBench Evaluation Script
Tests all 16 tasks across multiple evaluation paths:
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
    """Optimal + postmortem for maximum score (postmortem BEFORE verify — verify ends episode)."""
    await reset_task(client, "task1_memory_leak")
    await do_step(client, "check_metrics", {"service": "order-service", "metric": "memory"})
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "restart"})
    await do_step(client, "write_postmortem",
                  {"content": "Root cause: order-service OOM Kill due to memory leak in the heap. Fixed by restarting the service. Memory usage was at 98% before restart."})
    code, data = await do_step(client, "verify_health", {"service": "order-service"})
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
    """Optimal + postmortem (postmortem BEFORE verify — verify ends episode)."""
    await reset_task(client, "task2_db_cascade")
    await do_step(client, "read_logs", {"service": "payment-service"})
    await do_step(client, "check_metrics", {"service": "payment-db"})
    await do_step(client, "apply_fix", {"service": "payment-db", "fix_type": "increase_pool_size"})
    await do_step(client, "write_postmortem",
                  {"content": "Root cause: payment-db connection pool exhausted causing cascade failures through payment-service via HikariPool connection timeout. Fixed by increasing pool size."})
    code, data = await do_step(client, "verify_health")
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
    record("Task3", "Optimal path", data["done"] and score >= 0.75, f"score={score:.4f}")
    return score


async def eval_task3_with_postmortem(client):
    """Optimal + quality postmortem mentioning root cause (postmortem BEFORE verify)."""
    await reset_task(client, "task3_race_condition")
    await do_step(client, "check_metrics", {"service": "inventory-service", "metric": "error_rate"})
    await do_step(client, "check_deployments", {"service": "inventory-service"})
    await do_step(client, "run_diagnostic", {"service": "inventory-service", "type": "config_diff", "deploy_id": "deploy-a1b2c3"})
    await do_step(client, "apply_fix", {"service": "inventory-service", "fix_type": "rollback", "deploy_id": "deploy-a1b2c3"})
    await do_step(client, "write_postmortem",
                  {"content": "Root cause: inventory-service deploy-a1b2c3 changed redis lock_timeout_ms from 5000 to 500ms, causing race condition under load. Fixed by rolling back the config change."})
    code, data = await do_step(client, "verify_health")
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
    record("Task3", "Alt fix: rollback_config", data["done"] and score >= 0.40, f"score={score:.4f}")


# ═══════════════════════════════════════════════════════════
# TASK 4: DNS Resolution Failure (Easy-Medium)
# ═══════════════════════════════════════════════════════════

async def eval_task4_optimal(client):
    """Optimal path: investigate auth → DNS diagnostic → flush DNS → verify."""
    await reset_task(client, "task4_dns_failure")
    await do_step(client, "list_services")
    await do_step(client, "read_logs", {"service": "auth-service"})
    await do_step(client, "run_diagnostic", {"service": "auth-service", "type": "dns"})
    await do_step(client, "apply_fix", {"service": "auth-service", "fix_type": "flush_dns"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task4", "Optimal path", data["done"] and score >= 0.80, f"score={score:.4f}")
    return score


async def eval_task4_with_postmortem(client):
    """Optimal + postmortem for maximum score."""
    await reset_task(client, "task4_dns_failure")
    await do_step(client, "read_logs", {"service": "auth-service"})
    await do_step(client, "run_diagnostic", {"service": "auth-service", "type": "dns"})
    await do_step(client, "apply_fix", {"service": "auth-service", "fix_type": "flush_dns"})
    await do_step(client, "verify_health")
    code, data = await do_step(client, "write_postmortem",
                               {"content": "Root cause: auth-service DNS cache had stale entry for user-db. The cached IP was 10.0.0.99 but user-db moved to 10.0.1.15. Fixed by flushing DNS cache."})
    score = await get_grader(client)
    record("Task4", "With postmortem (max score)", score >= 0.85, f"score={score:.4f}")
    return score


async def eval_task4_wrong_service(client):
    """Fix wrong service — no effect."""
    await reset_task(client, "task4_dns_failure")
    await do_step(client, "apply_fix", {"service": "api-gateway", "fix_type": "restart"})
    score = await get_grader(client)
    record("Task4", "Wrong service fix", score <= 0.10, f"score={score:.4f}")


async def eval_task4_escalation(client):
    """Use escalation hint then fix correctly."""
    await reset_task(client, "task4_dns_failure")
    await do_step(client, "escalate")
    await do_step(client, "apply_fix", {"service": "auth-service", "fix_type": "flush_dns"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task4", "Escalation + fix (hint penalty)", data["done"] and 0.35 <= score <= 0.80,
           f"score={score:.4f}")


# ═══════════════════════════════════════════════════════════
# TASK 5: Certificate Expiry Chain (Medium-Hard)
# ═══════════════════════════════════════════════════════════

async def eval_task5_optimal(client):
    """Optimal: trace TLS errors → diagnose cert → renew cert → verify."""
    await reset_task(client, "task5_cert_expiry")
    await do_step(client, "list_services")
    await do_step(client, "read_logs", {"service": "payment-service"})
    await do_step(client, "run_diagnostic", {"service": "payment-service", "type": "tls"})
    await do_step(client, "apply_fix", {"service": "payment-service", "fix_type": "renew_cert"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task5", "Optimal path", data["done"] and score >= 0.80, f"score={score:.4f}")
    return score


async def eval_task5_with_postmortem(client):
    """Optimal + postmortem (postmortem BEFORE verify — verify ends episode)."""
    await reset_task(client, "task5_cert_expiry")
    await do_step(client, "read_logs", {"service": "payment-service"})
    await do_step(client, "run_diagnostic", {"service": "payment-service", "type": "tls"})
    await do_step(client, "apply_fix", {"service": "payment-service", "fix_type": "renew_cert"})
    await do_step(client, "write_postmortem",
                  {"content": "Root cause: TLS cert expired on payment-service. SSL auto-renewal had failed. Fixed by manual cert renew. mTLS to payment-db also restored."})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task5", "With postmortem (max score)", score >= 0.85, f"score={score:.4f}")
    return score


async def eval_task5_restart_trap(client):
    """Restart payment-service without renewing cert — doesn't fix."""
    await reset_task(client, "task5_cert_expiry")
    await do_step(client, "apply_fix", {"service": "payment-service", "fix_type": "restart"})
    score = await get_grader(client)
    record("Task5", "Restart trap (cert still expired)", score <= 0.15, f"score={score:.4f}")


async def eval_task5_wrong_service(client):
    """Fix order-service (victim) — wrong."""
    await reset_task(client, "task5_cert_expiry")
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "restart"})
    score = await get_grader(client)
    record("Task5", "Wrong service (order-service)", score <= 0.10, f"score={score:.4f}")


# ═══════════════════════════════════════════════════════════
# TASK 6: Split-Brain Network Partition (Hard)
# ═══════════════════════════════════════════════════════════

async def eval_task6_optimal(client):
    """Full optimal: investigate → find deploy → rollback iptables → reconcile → verify."""
    await reset_task(client, "task6_network_partition")
    await do_step(client, "read_logs", {"service": "inventory-service"})
    await do_step(client, "check_deployments")
    await do_step(client, "run_diagnostic", {"service": "inventory-service", "type": "network"})
    await do_step(client, "apply_fix", {"fix_type": "rollback_deploy"})
    await do_step(client, "apply_fix", {"service": "inventory-service", "fix_type": "reconcile_data"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task6", "Optimal path", data["done"] and score >= 0.70, f"score={score:.4f}")
    return score


async def eval_task6_with_postmortem(client):
    """Optimal + postmortem."""
    await reset_task(client, "task6_network_partition")
    await do_step(client, "read_logs", {"service": "inventory-service"})
    await do_step(client, "check_deployments")
    await do_step(client, "run_diagnostic", {"service": "inventory-service", "type": "network"})
    await do_step(client, "apply_fix", {"fix_type": "rollback_deploy"})
    await do_step(client, "apply_fix", {"service": "inventory-service", "fix_type": "reconcile_data"})
    await do_step(client, "verify_health")
    code, data = await do_step(client, "write_postmortem",
                               {"content": "Root cause: deploy-net-001 added iptables rule blocking inventory-service → inventory-db traffic. Caused split-brain: stale cache served to order-service while DB had updated stock. 15 oversold orders. Fixed by rolling back iptables rule and reconciling cache."})
    score = await get_grader(client)
    record("Task6", "With postmortem (max score)", score >= 0.80, f"score={score:.4f}")
    return score


async def eval_task6_partial_fix(client):
    """Fix network but skip data reconciliation."""
    await reset_task(client, "task6_network_partition")
    await do_step(client, "apply_fix", {"fix_type": "rollback_deploy"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task6", "Partial fix (no reconciliation)", not data["done"] and score <= 0.40,
           f"score={score:.4f}, done={data['done']}")


async def eval_task6_restart_trap(client):
    """Restart inventory-service without fixing network — still fails."""
    await reset_task(client, "task6_network_partition")
    await do_step(client, "apply_fix", {"service": "inventory-service", "fix_type": "restart"})
    score = await get_grader(client)
    record("Task6", "Restart trap (network still partitioned)", score <= 0.10, f"score={score:.4f}")


async def eval_task6_reconcile_before_network(client):
    """Try to reconcile data before fixing network — fails."""
    await reset_task(client, "task6_network_partition")
    await do_step(client, "apply_fix", {"service": "inventory-service", "fix_type": "reconcile_data"})
    score = await get_grader(client)
    record("Task6", "Reconcile before network fix (no effect)", score <= 0.10, f"score={score:.4f}")


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
    record("OpenEnv", "/tasks returns 16 tasks", len(data.get("tasks", [])) == 16)

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
# DISTRIBUTED TRACING TESTS
# ═══════════════════════════════════════════════════════════

async def eval_trace_request_task1(client):
    """trace_request returns valid span waterfall for task1."""
    await reset_task(client, "task1_memory_leak")
    code, data = await do_step(client, "trace_request", {"service": "api-gateway"})
    result = data["observation"]["last_action_result"]
    has_trace = "Trace ID:" in result and "span" in result.lower() or "Service" in result
    has_anomaly = "OutOfMemoryError" in result or "order-service" in result
    record("Tracing", "Task1 trace has span waterfall", has_trace, f"len={len(result)}")
    record("Tracing", "Task1 trace reveals OOM anomaly", has_anomaly, result[:200])


async def eval_trace_request_task2(client):
    """trace_request reveals payment-db pool exhaustion in task2."""
    await reset_task(client, "task2_db_cascade")
    code, data = await do_step(client, "trace_request", {"service": "api-gateway"})
    result = data["observation"]["last_action_result"]
    has_pool = "pool" in result.lower() or "payment-db" in result
    record("Tracing", "Task2 trace reveals DB pool issue", has_pool, result[:200])


async def eval_trace_request_task6(client):
    """trace_request reveals network partition in task6."""
    await reset_task(client, "task6_network_partition")
    code, data = await do_step(client, "trace_request", {"service": "api-gateway"})
    result = data["observation"]["last_action_result"]
    has_partition = "iptables" in result.lower() or "ConnectionTimedOut" in result or "stale_cache" in result
    record("Tracing", "Task6 trace reveals partition", has_partition, result[:200])


async def eval_trace_evidence_tracking(client):
    """trace_request counts toward evidence breadth."""
    await reset_task(client, "task1_memory_leak")
    await do_step(client, "trace_request", {"service": "order-service"})
    await do_step(client, "list_services")
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "restart"})
    await do_step(client, "verify_health", {"service": "order-service"})
    score = await get_grader(client)
    record("Tracing", "trace_request contributes to evidence breadth", score >= 0.75,
           f"score={score:.4f}")


# ═══════════════════════════════════════════════════════════
# SLO / ERROR BUDGET TESTS
# ═══════════════════════════════════════════════════════════

async def eval_check_slo(client):
    """check_slo returns SLO dashboard with burn rates."""
    await reset_task(client, "task2_db_cascade")
    # Do a few steps so SLO burns
    await do_step(client, "list_services")
    await do_step(client, "check_alerts")
    code, data = await do_step(client, "check_slo")
    result = data["observation"]["last_action_result"]
    has_slo = "SLO" in result and "budget" in result.lower()
    record("SLO", "check_slo returns SLO dashboard", has_slo, result[:200])


async def eval_slo_burn_visible(client):
    """SLO warnings appear in observations as budget burns."""
    await reset_task(client, "task2_db_cascade")
    # Run several steps to burn error budget
    for _ in range(8):
        code, data = await do_step(client, "check_alerts")
    result = data["observation"]["last_action_result"]
    has_slo_warning = "SLO" in result
    record("SLO", "SLO burn warnings visible in observations", has_slo_warning, result[-200:])


async def eval_slo_early_fix_bonus(client):
    """Fixing before SLO breach gives bonus score."""
    await reset_task(client, "task1_memory_leak")
    # Quick fix — should be before breach
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "restart"})
    await do_step(client, "verify_health", {"service": "order-service"})
    score_fast = await get_grader(client)

    await reset_task(client, "task1_memory_leak")
    # Slow fix — burn many steps first
    for _ in range(12):
        await do_step(client, "check_alerts")
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "restart"})
    await do_step(client, "verify_health", {"service": "order-service"})
    score_slow = await get_grader(client)

    record("SLO", "Fast fix scores higher (SLO + time bonus)", score_fast > score_slow,
           f"fast={score_fast:.4f}, slow={score_slow:.4f}")


# ═══════════════════════════════════════════════════════════
# INCIDENT COMMUNICATION TESTS
# ═══════════════════════════════════════════════════════════

async def eval_classify_severity_correct(client):
    """Correct severity classification gives bonus."""
    await reset_task(client, "task1_memory_leak")
    code, data = await do_step(client, "classify_severity", {"severity": "SEV2"})
    result = data["observation"]["last_action_result"]
    has_classification = "SEV2" in result and "classified" in result.lower()
    record("Communication", "Correct severity SEV2 accepted", has_classification, result[:150])

    # Now complete and check score includes bonus
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "restart"})
    await do_step(client, "verify_health", {"service": "order-service"})
    score = await get_grader(client)
    record("Communication", "Correct severity adds to score", score >= 0.80,
           f"score={score:.4f}")


async def eval_classify_severity_wrong(client):
    """Wrong severity classification does not add bonus."""
    await reset_task(client, "task1_memory_leak")
    await do_step(client, "classify_severity", {"severity": "SEV4"})  # wrong for task1
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "restart"})
    await do_step(client, "verify_health", {"service": "order-service"})
    score_wrong = await get_grader(client)

    await reset_task(client, "task1_memory_leak")
    await do_step(client, "classify_severity", {"severity": "SEV2"})  # correct
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "restart"})
    await do_step(client, "verify_health", {"service": "order-service"})
    score_correct = await get_grader(client)

    record("Communication", "Wrong severity scores lower than correct",
           score_correct > score_wrong,
           f"correct={score_correct:.4f}, wrong={score_wrong:.4f}")


async def eval_update_status_page(client):
    """Status page update before fix gives bonus."""
    await reset_task(client, "task2_db_cascade")
    await do_step(client, "update_status_page", {
        "status": "investigating",
        "message": "We are investigating elevated error rates affecting payment processing."
    })
    await do_step(client, "read_logs", {"service": "payment-service"})
    await do_step(client, "check_metrics", {"service": "payment-db"})
    await do_step(client, "apply_fix", {"service": "payment-db", "fix_type": "increase_pool_size"})
    await do_step(client, "write_postmortem",
                  {"content": "Root cause: payment-db connection pool exhausted. Fixed by increasing pool size."})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Communication", "Status page before fix contributes to score", score >= 0.85,
           f"score={score:.4f}")


async def eval_status_page_short_message(client):
    """Status page with too-short message is rejected."""
    await reset_task(client, "task1_memory_leak")
    code, data = await do_step(client, "update_status_page", {
        "status": "investigating",
        "message": "Looking into it"
    })
    result = data["observation"]["last_action_result"]
    record("Communication", "Short status message rejected", "rejected" in result.lower() or "20 char" in result.lower(),
           result[:150])


async def eval_classify_severity_invalid(client):
    """Invalid severity value is handled gracefully."""
    await reset_task(client, "task1_memory_leak")
    code, data = await do_step(client, "classify_severity", {"severity": "CRITICAL"})
    result = data["observation"]["last_action_result"]
    record("Communication", "Invalid severity handled", "invalid" in result.lower() or "must be" in result.lower(),
           result[:150])


async def eval_full_communication_flow(client):
    """Full optimal path with all communication actions maximises score."""
    await reset_task(client, "task5_cert_expiry")
    await do_step(client, "list_services")
    await do_step(client, "check_slo")
    await do_step(client, "classify_severity", {"severity": "SEV1"})
    await do_step(client, "update_status_page", {
        "status": "investigating",
        "message": "Payment processing affected by TLS certificate issue. Investigating root cause."
    })
    await do_step(client, "read_logs", {"service": "payment-service"})
    await do_step(client, "trace_request", {"service": "api-gateway"})
    await do_step(client, "run_diagnostic", {"service": "payment-service", "type": "tls"})
    await do_step(client, "apply_fix", {"service": "payment-service", "fix_type": "renew_cert"})
    await do_step(client, "update_status_page", {
        "status": "resolved",
        "message": "TLS certificate renewed. Payment processing restored. All services healthy."
    })
    await do_step(client, "write_postmortem", {
        "content": "Root cause: payment-service TLS certificate expired 2h ago. Auto-renewal via cert-manager failed due to OOM. Renewed certificate manually. Payment-service SSL and mTLS to payment-db restored."
    })
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Communication", "Full communication flow (max score)", score >= 0.90,
           f"score={score:.4f}")
    return score


# ═══════════════════════════════════════════════════════════
# TASK 7: Kafka Consumer Lag (Medium)
# ═══════════════════════════════════════════════════════════

async def eval_task7_optimal(client):
    await reset_task(client, "task7_kafka_lag")
    await do_step(client, "list_services")
    await do_step(client, "check_metrics", {"service": "order-service"})
    await do_step(client, "check_deployments", {})
    await do_step(client, "run_diagnostic", {"service": "order-service", "type": "kafka"})
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "rollback"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task7", "Optimal path", data["done"] and score >= 0.75, f"score={score:.4f}")
    return score


async def eval_task7_max(client):
    await reset_task(client, "task7_kafka_lag")
    await do_step(client, "list_services")
    await do_step(client, "check_slo")
    await do_step(client, "classify_severity", {"severity": "SEV2"})
    await do_step(client, "update_status_page", {"status": "investigating", "message": "Order processing delays. Investigating Kafka consumer lag."})
    await do_step(client, "check_metrics", {"service": "order-service"})
    await do_step(client, "check_deployments", {})
    await do_step(client, "trace_request", {"service": "order-service"})
    await do_step(client, "run_diagnostic", {"service": "order-service", "type": "kafka"})
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "rollback"})
    await do_step(client, "update_status_page", {"status": "resolved", "message": "Kafka consumer config restored. Orders processing normally."})
    await do_step(client, "write_postmortem", {"content": "Root cause: deploy changed kafka session.timeout.ms from 30000 to 3000, causing constant consumer rebalances and lag of 12847 messages. Rolled back deploy."})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task7", "Max path (all features)", data["done"] and score >= 0.85, f"score={score:.4f}")
    return score


# ═══════════════════════════════════════════════════════════
# TASK 8: Redis Sentinel Failover (Medium-Hard)
# ═══════════════════════════════════════════════════════════

async def eval_task8_optimal(client):
    await reset_task(client, "task8_redis_failover")
    await do_step(client, "list_services")
    await do_step(client, "check_metrics", {"service": "inventory-service"})
    await do_step(client, "read_logs", {"service": "inventory-service"})
    await do_step(client, "run_diagnostic", {"service": "inventory-service", "type": "redis"})
    await do_step(client, "apply_fix", {"service": "inventory-service", "fix_type": "reduce_quorum"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task8", "Optimal path", data["done"] and score >= 0.70, f"score={score:.4f}")
    return score


async def eval_task8_max(client):
    await reset_task(client, "task8_redis_failover")
    await do_step(client, "list_services")
    await do_step(client, "classify_severity", {"severity": "SEV2"})
    await do_step(client, "update_status_page", {"status": "investigating", "message": "Inventory service degraded. Redis cache failures."})
    await do_step(client, "check_metrics", {"service": "inventory-service"})
    await do_step(client, "read_logs", {"service": "inventory-service"})
    await do_step(client, "trace_request", {"service": "inventory-service"})
    await do_step(client, "run_diagnostic", {"service": "inventory-service", "type": "sentinel"})
    await do_step(client, "apply_fix", {"service": "inventory-service", "fix_type": "manual_failover"})
    await do_step(client, "update_status_page", {"status": "resolved", "message": "Redis failover completed. Cache recovering."})
    await do_step(client, "write_postmortem", {"content": "Root cause: Redis primary and sentinel co-located. When primary died, sentinel quorum dropped to 2/3. Fixed by reducing quorum. Cache rebuilding."})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task8", "Max path", data["done"] and score >= 0.80, f"score={score:.4f}")
    return score


# ═══════════════════════════════════════════════════════════
# TASK 9: Disk Space Exhaustion (Medium)
# ═══════════════════════════════════════════════════════════

async def eval_task9_optimal(client):
    await reset_task(client, "task9_disk_full")
    await do_step(client, "list_services")
    await do_step(client, "read_logs", {"service": "auth-service"})
    await do_step(client, "read_logs", {"service": "user-db"})
    await do_step(client, "run_diagnostic", {"service": "user-db", "type": "disk"})
    await do_step(client, "apply_fix", {"service": "user-db", "fix_type": "clean_wal_enable_cron"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task9", "Optimal path", data["done"] and score >= 0.75, f"score={score:.4f}")
    return score


async def eval_task9_max(client):
    await reset_task(client, "task9_disk_full")
    await do_step(client, "list_services")
    await do_step(client, "classify_severity", {"severity": "SEV1"})
    await do_step(client, "update_status_page", {"status": "investigating", "message": "Auth service failing. Investigating user-db disk issue."})
    await do_step(client, "read_logs", {"service": "auth-service"})
    await do_step(client, "check_metrics", {"service": "user-db"})
    await do_step(client, "run_diagnostic", {"service": "user-db", "type": "disk"})
    await do_step(client, "apply_fix", {"service": "user-db", "fix_type": "clean_wal"})
    await do_step(client, "write_postmortem", {"content": "Root cause: user-db disk full from WAL log accumulation. pg_archivecleanup cron was disabled 2 days ago. Cleaned WAL and re-enabled rotation cron."})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task9", "Max path", data["done"] and score >= 0.85, f"score={score:.4f}")
    return score


# ═══════════════════════════════════════════════════════════
# TASK 10: Rate Limiter Misconfiguration (Medium)
# ═══════════════════════════════════════════════════════════

async def eval_task10_optimal(client):
    await reset_task(client, "task10_rate_limit")
    await do_step(client, "list_services")
    await do_step(client, "check_metrics", {"service": "api-gateway"})
    await do_step(client, "check_deployments", {})
    await do_step(client, "run_diagnostic", {"service": "api-gateway", "type": "rate_limit"})
    await do_step(client, "apply_fix", {"service": "api-gateway", "fix_type": "rollback"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task10", "Optimal path", data["done"] and score >= 0.75, f"score={score:.4f}")
    return score


async def eval_task10_max(client):
    await reset_task(client, "task10_rate_limit")
    await do_step(client, "list_services")
    await do_step(client, "classify_severity", {"severity": "SEV1"})
    await do_step(client, "update_status_page", {"status": "investigating", "message": "API returning 429 errors. Investigating rate limiter."})
    await do_step(client, "check_metrics", {"service": "api-gateway"})
    await do_step(client, "check_deployments", {})
    await do_step(client, "run_diagnostic", {"service": "api-gateway", "type": "rate_limit"})
    await do_step(client, "apply_fix", {"service": "api-gateway", "fix_type": "restore_rate_limit"})
    await do_step(client, "write_postmortem", {"content": "Root cause: deploy-ratelimit-01 set rate limit to 100 req/s instead of 10000 (missing zero). 90% traffic throttled with 429. Rolled back config."})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task10", "Max path", data["done"] and score >= 0.85, f"score={score:.4f}")
    return score


# ═══════════════════════════════════════════════════════════
# TASK 11: DB Migration Lock (Hard)
# ═══════════════════════════════════════════════════════════

async def eval_task11_optimal(client):
    await reset_task(client, "task11_db_migration_lock")
    await do_step(client, "list_services")
    await do_step(client, "read_logs", {"service": "payment-service"})
    await do_step(client, "read_logs", {"service": "payment-db"})
    await do_step(client, "run_diagnostic", {"service": "payment-db", "type": "locks"})
    await do_step(client, "apply_fix", {"service": "payment-db", "fix_type": "kill_migration"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task11", "Optimal path", data["done"] and score >= 0.70, f"score={score:.4f}")
    return score


async def eval_task11_max(client):
    await reset_task(client, "task11_db_migration_lock")
    await do_step(client, "list_services")
    await do_step(client, "classify_severity", {"severity": "SEV1"})
    await do_step(client, "update_status_page", {"status": "investigating", "message": "Payment processing down. Database lock investigation."})
    await do_step(client, "read_logs", {"service": "payment-service"})
    await do_step(client, "check_metrics", {"service": "payment-db"})
    await do_step(client, "read_logs", {"service": "payment-db"})
    await do_step(client, "run_diagnostic", {"service": "payment-db", "type": "locks"})
    await do_step(client, "apply_fix", {"service": "payment-db", "fix_type": "kill_migration"})
    await do_step(client, "write_postmortem", {"content": "Root cause: DBA ran ALTER TABLE payment_transactions ADD COLUMN during peak hours without lock_timeout. Exclusive lock blocked all writes. Killed migration query."})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task11", "Max path", data["done"] and score >= 0.80, f"score={score:.4f}")
    return score


# ═══════════════════════════════════════════════════════════
# TASK 12: Health Check Flapping (Medium-Hard)
# ═══════════════════════════════════════════════════════════

async def eval_task12_optimal(client):
    await reset_task(client, "task12_health_flap")
    await do_step(client, "list_services")
    await do_step(client, "read_logs", {"service": "order-service"})
    await do_step(client, "check_metrics", {"service": "order-service"})
    await do_step(client, "run_diagnostic", {"service": "order-service", "type": "health_check"})
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "use_shallow_health_check"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task12", "Optimal path", data["done"] and score >= 0.75, f"score={score:.4f}")
    return score


async def eval_task12_max(client):
    await reset_task(client, "task12_health_flap")
    await do_step(client, "list_services")
    await do_step(client, "classify_severity", {"severity": "SEV2"})
    await do_step(client, "update_status_page", {"status": "investigating", "message": "order-service instances flapping. Health check investigation."})
    await do_step(client, "read_logs", {"service": "order-service"})
    await do_step(client, "check_metrics", {"service": "order-service"})
    await do_step(client, "trace_request", {"service": "order-service"})
    await do_step(client, "run_diagnostic", {"service": "order-service", "type": "health_check"})
    await do_step(client, "apply_fix", {"service": "order-service", "fix_type": "simplify_health_check"})
    await do_step(client, "write_postmortem", {"content": "Root cause: order-service /health/deep endpoint called inventory-service which had GC pauses up to 1200ms. Health check timeout 2000ms intermittently exceeded. Switched to shallow health check."})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task12", "Max path", data["done"] and score >= 0.85, f"score={score:.4f}")
    return score


# ═══════════════════════════════════════════════════════════
# TASK 13: Pod Eviction Storm (Hard)
# ═══════════════════════════════════════════════════════════

async def eval_task13_optimal(client):
    await reset_task(client, "task13_pod_eviction")
    await do_step(client, "list_services")
    await do_step(client, "read_logs", {"service": "payment-service"})
    await do_step(client, "check_alerts")
    await do_step(client, "run_diagnostic", {"service": "payment-service", "type": "kubernetes"})
    await do_step(client, "apply_fix", {"fix_type": "delete_batch_job"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task13", "Optimal path", data["done"] and score >= 0.70, f"score={score:.4f}")
    return score


async def eval_task13_max(client):
    await reset_task(client, "task13_pod_eviction")
    await do_step(client, "list_services")
    await do_step(client, "classify_severity", {"severity": "SEV1"})
    await do_step(client, "update_status_page", {"status": "investigating", "message": "Payment service pods evicted. Investigating node pressure."})
    await do_step(client, "read_logs", {"service": "payment-service"})
    await do_step(client, "check_alerts")
    await do_step(client, "check_deployments", {})
    await do_step(client, "run_diagnostic", {"service": "payment-service", "type": "kubernetes"})
    await do_step(client, "apply_fix", {"fix_type": "delete_batch_job"})
    await do_step(client, "write_postmortem", {"content": "Root cause: daily-report-generator DaemonSet deployed without resource limits, consuming 12GB/node. payment-service BestEffort QoS evicted first under memory pressure. Deleted batch job."})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task13", "Max path", data["done"] and score >= 0.80, f"score={score:.4f}")
    return score


# ═══════════════════════════════════════════════════════════
# TASK 14: Cascading Timeout (Medium-Hard)
# ═══════════════════════════════════════════════════════════

async def eval_task14_optimal(client):
    await reset_task(client, "task14_cascading_timeout")
    await do_step(client, "list_services")
    await do_step(client, "read_logs", {"service": "api-gateway"})
    await do_step(client, "check_metrics", {"service": "inventory-service"})
    await do_step(client, "run_diagnostic", {"service": "inventory-service", "type": "query"})
    await do_step(client, "apply_fix", {"fix_type": "recreate_index"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task14", "Optimal path", data["done"] and score >= 0.70, f"score={score:.4f}")
    return score


async def eval_task14_max(client):
    await reset_task(client, "task14_cascading_timeout")
    await do_step(client, "list_services")
    await do_step(client, "classify_severity", {"severity": "SEV2"})
    await do_step(client, "update_status_page", {"status": "investigating", "message": "504 errors on order endpoints. Investigating timeout chain."})
    await do_step(client, "read_logs", {"service": "api-gateway"})
    await do_step(client, "read_logs", {"service": "inventory-service"})
    await do_step(client, "trace_request", {"service": "api-gateway"})
    await do_step(client, "run_diagnostic", {"service": "inventory-service", "type": "query"})
    await do_step(client, "apply_fix", {"fix_type": "recreate_index"})
    await do_step(client, "write_postmortem", {"content": "Root cause: missing index on inventory.sku dropped during nightly maintenance. 25s queries combined with timeout hierarchy mismatch (gateway 10s < order 30s) caused user-facing 504 errors."})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task14", "Max path", data["done"] and score >= 0.80, f"score={score:.4f}")
    return score


# ═══════════════════════════════════════════════════════════
# TASK 15: Secret Rotation Failure (Medium)
# ═══════════════════════════════════════════════════════════

async def eval_task15_optimal(client):
    await reset_task(client, "task15_secret_rotation")
    await do_step(client, "list_services")
    await do_step(client, "read_logs", {"service": "payment-service"})
    await do_step(client, "run_diagnostic", {"service": "payment-service", "type": "secrets"})
    await do_step(client, "apply_fix", {"service": "payment-service", "fix_type": "restart"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task15", "Optimal path", data["done"] and score >= 0.75, f"score={score:.4f}")
    return score


async def eval_task15_max(client):
    await reset_task(client, "task15_secret_rotation")
    await do_step(client, "list_services")
    await do_step(client, "classify_severity", {"severity": "SEV1"})
    await do_step(client, "update_status_page", {"status": "investigating", "message": "All payments failing. Investigating 401 errors."})
    await do_step(client, "read_logs", {"service": "payment-service"})
    await do_step(client, "check_metrics", {"service": "payment-service"})
    await do_step(client, "run_diagnostic", {"service": "payment-service", "type": "secrets"})
    await do_step(client, "apply_fix", {"service": "payment-service", "fix_type": "reload_secrets"})
    await do_step(client, "write_postmortem", {"content": "Root cause: Vault rotated payment-api-key from v2 to v3 but payment-service has no hot-reload. Old key revoked, all API calls return 401. Restarted service to pick up v3."})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task15", "Max path", data["done"] and score >= 0.85, f"score={score:.4f}")
    return score


# ═══════════════════════════════════════════════════════════
# TASK 16: Debug Log Storm (Hard)
# ═══════════════════════════════════════════════════════════

async def eval_task16_optimal(client):
    await reset_task(client, "task16_log_storm")
    await do_step(client, "list_services")
    await do_step(client, "check_metrics", {"service": "auth-service"})
    await do_step(client, "check_deployments", {})
    await do_step(client, "run_diagnostic", {"service": "auth-service", "type": "logging"})
    await do_step(client, "apply_fix", {"service": "auth-service", "fix_type": "rollback"})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task16", "Optimal path", data["done"] and score >= 0.70, f"score={score:.4f}")
    return score


async def eval_task16_max(client):
    await reset_task(client, "task16_log_storm")
    await do_step(client, "list_services")
    await do_step(client, "classify_severity", {"severity": "SEV2"})
    await do_step(client, "update_status_page", {"status": "investigating", "message": "Logging pipeline saturated. auth-service CPU spiked. Investigating."})
    await do_step(client, "check_metrics", {"service": "auth-service"})
    await do_step(client, "check_deployments", {})
    await do_step(client, "read_logs", {"service": "auth-service"})
    await do_step(client, "run_diagnostic", {"service": "auth-service", "type": "logging"})
    await do_step(client, "apply_fix", {"service": "auth-service", "fix_type": "set_log_level_info"})
    await do_step(client, "write_postmortem", {"content": "Root cause: deploy-auth-debug accidentally set LOG_LEVEL=DEBUG in production (meant for staging). 50GB/hr of debug logs saturated shared logging pipeline. auth-service CPU 95%. Rolled back log level."})
    code, data = await do_step(client, "verify_health")
    score = await get_grader(client)
    record("Task16", "Max path", data["done"] and score >= 0.80, f"score={score:.4f}")
    return score


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

        # ── Task 4 ─────────────────────────────────
        print("\n── Task 4: DNS Resolution Failure (Easy-Medium) ──")
        t4_optimal = await eval_task4_optimal(client)
        t4_max = await eval_task4_with_postmortem(client)
        await eval_task4_wrong_service(client)
        await eval_task4_escalation(client)

        # ── Task 5 ─────────────────────────────────
        print("\n── Task 5: Certificate Expiry Chain (Medium-Hard) ──")
        t5_optimal = await eval_task5_optimal(client)
        t5_max = await eval_task5_with_postmortem(client)
        await eval_task5_restart_trap(client)
        await eval_task5_wrong_service(client)

        # ── Task 6 ─────────────────────────────────
        print("\n── Task 6: Network Partition (Hard) ──")
        t6_optimal = await eval_task6_optimal(client)
        t6_max = await eval_task6_with_postmortem(client)
        await eval_task6_partial_fix(client)
        await eval_task6_restart_trap(client)
        await eval_task6_reconcile_before_network(client)

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

        # ── Distributed Tracing ───────────────────
        print("\n── Distributed Tracing ──")
        await eval_trace_request_task1(client)
        await eval_trace_request_task2(client)
        await eval_trace_request_task6(client)
        await eval_trace_evidence_tracking(client)

        # ── SLO / Error Budget ────────────────────
        print("\n── SLO / Error Budget ──")
        await eval_check_slo(client)
        await eval_slo_burn_visible(client)
        await eval_slo_early_fix_bonus(client)

        # ── Incident Communication ────────────────
        print("\n── Incident Communication ──")
        await eval_classify_severity_correct(client)
        await eval_classify_severity_wrong(client)
        await eval_update_status_page(client)
        await eval_status_page_short_message(client)
        await eval_classify_severity_invalid(client)
        t5_full_comm = await eval_full_communication_flow(client)

        # ── Task 7 ─────────────────────────────────
        print("\n── Task 7: Kafka Consumer Lag (Medium) ──")
        t7_optimal = await eval_task7_optimal(client)
        t7_max = await eval_task7_max(client)

        # ── Task 8 ─────────────────────────────────
        print("\n── Task 8: Redis Sentinel Failover (Medium-Hard) ──")
        t8_optimal = await eval_task8_optimal(client)
        t8_max = await eval_task8_max(client)

        # ── Task 9 ─────────────────────────────────
        print("\n── Task 9: Disk Space Exhaustion (Medium) ──")
        t9_optimal = await eval_task9_optimal(client)
        t9_max = await eval_task9_max(client)

        # ── Task 10 ────────────────────────────────
        print("\n── Task 10: Rate Limiter Misconfiguration (Medium) ──")
        t10_optimal = await eval_task10_optimal(client)
        t10_max = await eval_task10_max(client)

        # ── Task 11 ────────────────────────────────
        print("\n── Task 11: DB Migration Lock (Hard) ──")
        t11_optimal = await eval_task11_optimal(client)
        t11_max = await eval_task11_max(client)

        # ── Task 12 ────────────────────────────────
        print("\n── Task 12: Health Check Flapping (Medium-Hard) ──")
        t12_optimal = await eval_task12_optimal(client)
        t12_max = await eval_task12_max(client)

        # ── Task 13 ────────────────────────────────
        print("\n── Task 13: Pod Eviction Storm (Hard) ──")
        t13_optimal = await eval_task13_optimal(client)
        t13_max = await eval_task13_max(client)

        # ── Task 14 ────────────────────────────────
        print("\n── Task 14: Cascading Timeout (Medium-Hard) ──")
        t14_optimal = await eval_task14_optimal(client)
        t14_max = await eval_task14_max(client)

        # ── Task 15 ────────────────────────────────
        print("\n── Task 15: Secret Rotation (Medium) ──")
        t15_optimal = await eval_task15_optimal(client)
        t15_max = await eval_task15_max(client)

        # ── Task 16 ────────────────────────────────
        print("\n── Task 16: Log Storm (Hard) ──")
        t16_optimal = await eval_task16_optimal(client)
        t16_max = await eval_task16_max(client)

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
    print(f"  Task 4 Optimal:        {t4_optimal:.4f}")
    print(f"  Task 4 Max (w/postm):  {t4_max:.4f}")
    print(f"  Task 5 Optimal:        {t5_optimal:.4f}")
    print(f"  Task 5 Max (w/postm):  {t5_max:.4f}")
    print(f"  Task 5 Full Comm:      {t5_full_comm:.4f}")
    print(f"  Task 6 Optimal:        {t6_optimal:.4f}")
    print(f"  Task 6 Max (w/postm):  {t6_max:.4f}")
    print(f"  Task 7 Optimal:        {t7_optimal:.4f}")
    print(f"  Task 7 Max:            {t7_max:.4f}")
    print(f"  Task 8 Optimal:        {t8_optimal:.4f}")
    print(f"  Task 8 Max:            {t8_max:.4f}")
    print(f"  Task 9 Optimal:        {t9_optimal:.4f}")
    print(f"  Task 9 Max:            {t9_max:.4f}")
    print(f"  Task 10 Optimal:       {t10_optimal:.4f}")
    print(f"  Task 10 Max:           {t10_max:.4f}")
    print(f"  Task 11 Optimal:       {t11_optimal:.4f}")
    print(f"  Task 11 Max:           {t11_max:.4f}")
    print(f"  Task 12 Optimal:       {t12_optimal:.4f}")
    print(f"  Task 12 Max:           {t12_max:.4f}")
    print(f"  Task 13 Optimal:       {t13_optimal:.4f}")
    print(f"  Task 13 Max:           {t13_max:.4f}")
    print(f"  Task 14 Optimal:       {t14_optimal:.4f}")
    print(f"  Task 14 Max:           {t14_max:.4f}")
    print(f"  Task 15 Optimal:       {t15_optimal:.4f}")
    print(f"  Task 15 Max:           {t15_max:.4f}")
    print(f"  Task 16 Optimal:       {t16_optimal:.4f}")
    print(f"  Task 16 Max:           {t16_max:.4f}")
    print()
    all_optimal = [t1_optimal, t2_optimal, t3_optimal, t4_optimal, t5_optimal, t6_optimal,
                   t7_optimal, t8_optimal, t9_optimal, t10_optimal, t11_optimal, t12_optimal,
                   t13_optimal, t14_optimal, t15_optimal, t16_optimal]
    all_max = [t1_max, t2_max, t3_max, t4_max, t5_max, t6_max,
               t7_max, t8_max, t9_max, t10_max, t11_max, t12_max,
               t13_max, t14_max, t15_max, t16_max]
    print(f"  Average optimal:       {sum(all_optimal) / len(all_optimal):.4f}")
    print(f"  Average max:           {sum(all_max) / len(all_max):.4f}")

    print("\n" + "=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
