#!/usr/bin/env python3
"""
Deterministic heuristic baseline: Simulates a reasonable-but-suboptimal SRE agent.
No LLM needed — runs scripted action sequences that represent typical agent behaviour:
- Investigates broadly before narrowing down
- Sometimes checks wrong services first
- Occasionally applies wrong fixes before finding the right one
- May skip postmortem on some tasks

Usage:
  python baseline/run_heuristic_baseline.py            # human-readable
  python baseline/run_heuristic_baseline.py --json      # JSON output
"""

import os
import sys
import json
import argparse
import httpx

BASE_URL = os.getenv("SREBENCH_URL", "http://localhost:7860")
SEED = 42


def step(action_type: str, parameters: dict, reasoning: str = "heuristic") -> dict:
    """Send one step to the environment."""
    resp = httpx.post(f"{BASE_URL}/step", json={
        "action": {
            "action_type": action_type,
            "parameters": parameters,
            "reasoning": reasoning,
        }
    }, timeout=30)
    return resp.json()


def reset(task_id: str) -> dict:
    resp = httpx.post(f"{BASE_URL}/reset",
                      json={"task_id": task_id, "seed": SEED}, timeout=30)
    return resp.json()


def get_grader_score() -> float:
    resp = httpx.get(f"{BASE_URL}/grader", timeout=30)
    return resp.json()["episode_score"]


# ─── TASK 1: Memory Leak ──────────────────────────────────────────────
# Weak-medium agent: spends time on api-gateway, eventually finds order-service
# but doesn't verify health. ~14 steps, misses verification + postmortem.
def run_task1():
    reset("task1_memory_leak")
    step("list_services", {})
    step("check_alerts", {})
    step("read_logs", {"service": "api-gateway"})
    step("check_metrics", {"service": "api-gateway"})
    step("apply_fix", {"service": "api-gateway", "fix_type": "restart"})  # wrong service
    step("check_metrics", {"service": "api-gateway"})  # still broken
    step("read_logs", {"service": "auth-service"})  # wrong service
    step("check_metrics", {"service": "auth-service"})
    step("read_logs", {"service": "user-db"})  # wasted step
    step("check_dependencies", {"service": "api-gateway"})
    step("read_logs", {"service": "order-service"})
    step("check_metrics", {"service": "order-service"})
    step("apply_fix", {"service": "order-service", "fix_type": "restart"})
    # Agent doesn't verify health or write postmortem — thinks it's done
    return get_grader_score()


# ─── TASK 2: DB Cascade ───────────────────────────────────────────────
# Agent traces to payment-service but fixes WRONG surface (order-service),
# never finds payment-db root cause. Score capped. ~18 steps.
def run_task2():
    reset("task2_db_cascade")
    step("check_alerts", {})
    step("list_services", {})
    step("read_logs", {"service": "api-gateway"})
    step("check_metrics", {"service": "api-gateway"})
    step("check_dependencies", {"service": "api-gateway"})
    step("read_logs", {"service": "order-service"})
    step("check_metrics", {"service": "order-service"})
    step("apply_fix", {"service": "order-service", "fix_type": "restart"})  # wrong surface
    step("read_logs", {"service": "order-service"})  # re-check
    step("check_dependencies", {"service": "order-service"})
    step("read_logs", {"service": "payment-service"})  # traces here
    step("check_metrics", {"service": "payment-service"})
    step("apply_fix", {"service": "payment-service", "fix_type": "restart"})  # wrong surface again
    step("read_logs", {"service": "payment-service"})
    step("escalate", {})  # needs a hint
    step("check_metrics", {"service": "payment-db"})  # finally finds DB
    step("apply_fix", {"service": "payment-db", "fix_type": "increase_pool_size"})
    step("verify_health", {})
    return get_grader_score()


# ─── TASK 3: Race Condition ───────────────────────────────────────────
# Hardest. Agent restarts 3 times (triggers dead-end trap), finds deploy
# but never checks config_diff. Tries generic rollback without deploy_id.
# Missing several milestones. ~22 steps.
def run_task3():
    reset("task3_race_condition")
    step("check_alerts", {})
    step("list_services", {})
    step("read_logs", {"service": "inventory-service"})
    step("check_metrics", {"service": "inventory-service", "metric": "error_rate"})
    step("apply_fix", {"service": "inventory-service", "fix_type": "restart"})  # dead-end #1
    step("read_logs", {"service": "inventory-service"})
    step("apply_fix", {"service": "inventory-service", "fix_type": "restart"})  # dead-end #2
    step("check_metrics", {"service": "inventory-service", "metric": "error_rate"})
    step("apply_fix", {"service": "inventory-service", "fix_type": "restart"})  # dead-end #3 → penalty!
    step("read_logs", {"service": "inventory-db"})
    step("check_dependencies", {"service": "inventory-service"})
    step("check_metrics", {"service": "inventory-db"})
    step("read_logs", {"service": "order-service"})
    step("check_deployments", {})  # finds deploy
    step("escalate", {})  # needs help
    step("run_diagnostic", {"service": "inventory-service", "type": "general"})  # not config_diff
    step("read_logs", {"service": "inventory-service"})
    step("apply_fix", {"service": "inventory-service", "fix_type": "rollback", "deploy_id": "deploy-a1b2c3"})
    step("verify_health", {})
    # No postmortem — agent ran out of patience
    return get_grader_score()


# ─── TASK 4: DNS Failure ──────────────────────────────────────────────
# Agent investigates api-gateway first, uses escalation, eventually finds
# auth-service DNS. Applied wrong fix first. ~14 steps.
def run_task4():
    reset("task4_dns_failure")
    step("check_alerts", {})
    step("list_services", {})
    step("read_logs", {"service": "api-gateway"})
    step("check_metrics", {"service": "api-gateway"})
    step("apply_fix", {"service": "api-gateway", "fix_type": "restart"})  # wrong service
    step("check_dependencies", {"service": "api-gateway"})
    step("read_logs", {"service": "auth-service"})
    step("check_metrics", {"service": "auth-service"})
    step("apply_fix", {"service": "auth-service", "fix_type": "restart"})  # wrong fix type (restart != flush_dns)
    step("read_logs", {"service": "auth-service"})
    step("escalate", {})  # gets hint
    step("run_diagnostic", {"service": "auth-service", "type": "dns"})
    step("apply_fix", {"service": "auth-service", "fix_type": "flush_dns"})
    step("verify_health", {})
    return get_grader_score()


# ─── TASK 5: Certificate Expiry ───────────────────────────────────────
# Agent struggles — investigates order-service, tries restart on payment-service
# twice, uses escalation. Eventually renews cert. No postmortem. ~18 steps.
def run_task5():
    reset("task5_cert_expiry")
    step("check_alerts", {})
    step("list_services", {})
    step("read_logs", {"service": "api-gateway"})
    step("check_metrics", {"service": "api-gateway"})
    step("read_logs", {"service": "order-service"})
    step("check_metrics", {"service": "order-service"})
    step("apply_fix", {"service": "order-service", "fix_type": "restart"})  # wrong service
    step("check_dependencies", {"service": "order-service"})
    step("read_logs", {"service": "payment-service"})
    step("check_metrics", {"service": "payment-service"})
    step("apply_fix", {"service": "payment-service", "fix_type": "restart"})  # wrong fix
    step("read_logs", {"service": "payment-service"})  # still broken
    step("apply_fix", {"service": "payment-service", "fix_type": "restart"})  # wrong fix again
    step("escalate", {})  # hint
    step("run_diagnostic", {"service": "payment-service", "type": "tls"})
    step("apply_fix", {"service": "payment-service", "fix_type": "renew_cert"})
    step("verify_health", {})
    # No postmortem
    return get_grader_score()


# ─── TASK 6: Network Partition ────────────────────────────────────────
# Hardest. Agent restarts twice, investigates broadly, finds partition
# and fixes it, but doesn't reconcile data → incomplete fix. ~22 steps.
def run_task6():
    reset("task6_network_partition")
    step("check_alerts", {})
    step("list_services", {})
    step("read_logs", {"service": "inventory-service"})
    step("check_metrics", {"service": "inventory-service"})
    step("apply_fix", {"service": "inventory-service", "fix_type": "restart"})  # wrong
    step("read_logs", {"service": "inventory-db"})
    step("check_metrics", {"service": "inventory-db"})
    step("apply_fix", {"service": "inventory-service", "fix_type": "restart"})  # wrong again
    step("check_dependencies", {"service": "inventory-service"})
    step("read_logs", {"service": "order-service"})
    step("check_metrics", {"service": "order-service"})
    step("check_deployments", {})
    step("escalate", {})  # needs help
    step("read_logs", {"service": "inventory-service"})
    step("run_diagnostic", {"service": "inventory-service", "type": "network"})
    step("apply_fix", {"fix_type": "rollback_deploy"})
    step("read_logs", {"service": "inventory-service"})
    # Agent thinks it's fixed but misses data reconciliation phase
    step("verify_health", {})  # won't fully resolve without reconcile
    return get_grader_score()


TASK_RUNNERS = {
    "task1_memory_leak": run_task1,
    "task2_db_cascade": run_task2,
    "task3_race_condition": run_task3,
    "task4_dns_failure": run_task4,
    "task5_cert_expiry": run_task5,
    "task6_network_partition": run_task6,
}


def main():
    parser = argparse.ArgumentParser(description="SREBench heuristic baseline")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON only")
    args = parser.parse_args()

    results = {}
    for task_id, runner in TASK_RUNNERS.items():
        if not args.json:
            print(f"\n▶ Running {task_id}...", file=sys.stderr)
        score = runner()
        results[task_id] = round(score, 4)
        if not args.json:
            print(f"  Score: {score:.4f}", file=sys.stderr)

    if args.json:
        print(json.dumps(results))
    else:
        print("\n=== SREBench Heuristic Baseline Results ===")
        for task_id, score in results.items():
            bar = "█" * int(score * 20)
            print(f"  {task_id:<30} {score:.4f}  {bar}")
        avg = sum(results.values()) / len(results)
        print(f"\n  Average: {avg:.4f}")


if __name__ == "__main__":
    main()
