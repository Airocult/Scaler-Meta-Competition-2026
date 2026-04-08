#!/usr/bin/env python3
"""
Baseline agent: Runs each task with ReAct-style prompting via NVIDIA API (DeepSeek).
Usage:
  python baseline/run_baseline.py            # prints human-readable
  python baseline/run_baseline.py --json     # prints JSON only (for /baseline endpoint)
"""

import os
import sys
import json
import argparse
import httpx
from openai import OpenAI

BASE_URL = os.getenv("SREBENCH_URL", "http://localhost:7860")
TASKS = ["task1_memory_leak", "task2_db_cascade", "task3_race_condition",
         "task4_dns_failure", "task5_cert_expiry", "task6_network_partition"]
SEED = 42
MODEL = os.getenv("BASELINE_MODEL", "gpt-5.1")
API_BASE = os.getenv("BASELINE_API_BASE", "https://api.openai.com/v1")

SYSTEM_PROMPT = """You are an expert SRE on-call engineer debugging a production incident.

## INVESTIGATION METHODOLOGY (follow this order)
1. ORIENT: list_services + check_alerts to understand scope
2. GATHER: read_logs + check_metrics on degraded services — focus on the MOST degraded service first
3. TRACE DEPENDENCIES: check_dependencies on the degraded service, then investigate UPSTREAM (leaf) services
4. CORRELATE: check_deployments to find recent changes. Compare deploy timestamps with error start times.
5. DIAGNOSE: run_diagnostic to confirm root cause. Match diagnostic type to symptoms (see PATTERN MATCHING below).
6. FIX: apply_fix targeting the ROOT CAUSE service, not symptom services. Include deploy_id when rolling back.
7. VERIFY: verify_health AFTER fixing — ALWAYS do this
8. DOCUMENT: write_postmortem mentioning the specific root cause — ALWAYS do this

## CRITICAL RULES
- NEVER apply_fix(fix_type="restart") as first action. Always diagnose first. Restarts don't fix config, certs, DNS, or network issues.
- When multiple services are degraded, the ROOT CAUSE is usually a LEAF service (databases: payment-db, user-db, inventory-db). Errors CASCADE UPWARD through the dependency chain.
- If errors started at time T, check what was deployed at T±5 minutes — timing correlation = likely cause.
- After ANY fix, ALWAYS run verify_health then write_postmortem. These are required for full score.

## PATTERN MATCHING (match symptoms to diagnostic type)
- "OOMKilled", "heap space", "memory exceeded" → memory leak on that service → fix_type="restart" (only for memory leaks)
- "connection pool exhausted", "HikariPool", "too many connections" → DB connection pool issue → trace to the DATABASE service → fix_type="increase_pool_size"
- "SSL handshake", "certificate expired", "TLS failed", "CERT_HAS_EXPIRED" → expired TLS certificate → run_diagnostic type="tls" → fix_type="renew_cert" on the cert-owning service
- "DNS resolution failed", "NXDOMAIN", "stale DNS", "getaddrinfo" → DNS cache issue → run_diagnostic type="dns" → fix_type="flush_dns"
- "connection timed out", "network unreachable", "stale cache", "split-brain" → network partition → run_diagnostic type="iptables" → fix_type="rollback_deploy" then reconcile_data
- Error rate spike correlating with deploy timestamp → config change → run_diagnostic type="config_diff" WITH deploy_id → fix_type="rollback" with deploy_id

## CASCADE TRACING
When alert points to a gateway/frontend service:
1. check_dependencies on the alerted service to see its upstreams
2. Trace the dependency chain toward LEAF services (databases)
3. check_metrics on EACH service in the chain — the one with highest error_rate or critical pool_usage is likely root cause
4. Read logs on the suspected root cause service for confirmation
Example: api-gateway(503s) → order-service → payment-service → payment-db(pool exhausted) = ROOT CAUSE is payment-db

## RESPONSE FORMAT
Output exactly one JSON object per turn:
{
  "action_type": "<action>",
  "parameters": { ... },
  "reasoning": "What I observe → What I suspect → Why this action"
}

## AVAILABLE ACTIONS
- list_services: {} — List all services and statuses
- check_alerts: {} — View active alerts
- read_logs: {"service": "<name>"} — Read application logs
- check_metrics: {"service": "<name>", "metric": "<type>"} — Get service metrics
- check_deployments: {"last_n": 5} or {"service": "<name>"} — Recent deploys with timestamps
- check_dependencies: {"service": "<name>"} — Service dependency graph
- run_diagnostic: {"service": "<name>", "type": "<diag_type>"} — Run diagnostics (types: general, config_diff, network, dns, tls, connection_pool, iptables). Add "deploy_id" for config_diff.
- apply_fix: {"service": "<name>", "fix_type": "<type>"} — Apply a remediation. Add "deploy_id" for rollbacks.
- verify_health: {"service": "<name>"} or {} — Verify resolution
- write_postmortem: {"content": "<detailed text>"} — Document the incident
- escalate: {} — Get a hint (costs points)

Output valid JSON only — no markdown fences, no commentary outside the JSON object.
"""


def call_llm_streaming(client: OpenAI, messages: list) -> str:
    """Call the LLM and return the content (non-streaming for reliability)."""
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        top_p=0.9,
        max_tokens=4096,
    )
    return completion.choices[0].message.content


def extract_json(content: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences and extra text."""
    content = content.strip()
    # Remove <think>...</think> tags if present
    import re
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    # Strip markdown fences
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()
    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object in the text
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No valid JSON found in: {content[:200]}")


def format_observation(obs: dict, step_num: int, reward: float, is_initial: bool = False) -> str:
    """Format observation to highlight key signals for the LLM."""
    parts = []
    if is_initial:
        parts.append(f"🚨 INCIDENT ALERT: {obs['alert_summary']}")
        parts.append(f"\nAvailable actions: {obs['available_actions']}")
    else:
        result = obs.get('last_action_result', '')
        parts.append(f"Result: {result}")
        parts.append(f"Phase: {obs['incident_phase']} | Step reward: {reward}")

    # Compact service statuses highlighting problems
    statuses = obs.get('service_statuses', {})
    degraded = []
    healthy = []
    if isinstance(statuses, dict):
        items = statuses.values()
    else:
        items = statuses
    for s in items:
        if isinstance(s, str):
            healthy.append(s)
            continue
        name = s.get('name', '?')
        status = s.get('status', '')
        err = s.get('error_rate', 0)
        lat = s.get('latency_p99_ms', 0)
        restarts = s.get('restarts_last_hour', 0)
        pool = s.get('connection_pool_usage', 0)
        if status in ('degraded', 'down'):
            extras = []
            if restarts > 0:
                extras.append(f"restarts={restarts}")
            if pool and pool > 0.8:
                extras.append(f"pool={pool} CRITICAL")
            extra_str = f", {', '.join(extras)}" if extras else ""
            degraded.append(f"  ⚠ {name}: {status} (err={err}, lat={lat}ms{extra_str})")
        else:
            healthy.append(name)

    if degraded:
        parts.append("\nDEGRADED/DOWN services (investigate these):")
        parts.extend(degraded)
    if healthy:
        parts.append(f"Healthy: {', '.join(healthy)}")

    # Pattern hints based on observation content
    result_text = obs.get('last_action_result', '')
    if any(kw in result_text.lower() for kw in ['ssl', 'tls', 'certificate', 'cert']):
        parts.append("\n💡 TLS/cert keywords detected → consider run_diagnostic type='tls'")
    if any(kw in result_text.lower() for kw in ['dns', 'nxdomain', 'getaddrinfo', 'stale ip']):
        parts.append("\n💡 DNS keywords detected → consider run_diagnostic type='dns'")
    if any(kw in result_text.lower() for kw in ['connection pool', 'hikaripool', 'pool exhausted']):
        parts.append("\n💡 Connection pool issue detected → trace to the DATABASE service (leaf node)")
    if any(kw in result_text.lower() for kw in ['network unreachable', 'connection timed out', 'iptables', 'split-brain']):
        parts.append("\n💡 Network partition keywords detected → consider run_diagnostic type='iptables'")

    # Periodic reminders
    if step_num > 0 and step_num % 6 == 0:
        parts.append("\n💡 Reminder: Have you checked deployments? Correlated timing? Run diagnostics with deploy_id?")
    if step_num > 0 and step_num % 10 == 0:
        parts.append("💡 Reminder: After fixing root cause, ALWAYS verify_health then write_postmortem.")

    return "\n".join(parts)


def run_episode(client: OpenAI, task_id: str) -> float:
    """Runs one full episode. Returns final grader score."""
    # 1. Reset
    reset_resp = httpx.post(f"{BASE_URL}/reset",
                            json={"task_id": task_id, "seed": SEED},
                            timeout=30)
    obs = reset_resp.json()["observation"]

    initial_msg = format_observation(obs, 0, 0.0, is_initial=True)
    initial_msg += "\n\nFollow the investigation methodology. What do you do first?"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_msg},
    ]

    max_steps = {"task1_memory_leak": 20,
                 "task2_db_cascade": 30,
                 "task3_race_condition": 40,
                 "task4_dns_failure": 25,
                 "task5_cert_expiry": 35,
                 "task6_network_partition": 40}[task_id]

    consecutive_errors = 0
    for step_num in range(max_steps):
        # 2. Get LLM action (with retry)
        raw_action = None
        content = ""
        for attempt in range(3):
            try:
                content = call_llm_streaming(client, messages)
                if not content or not content.strip():
                    print(f"  [step {step_num}] Empty response (attempt {attempt+1})", file=sys.stderr)
                    continue
                raw_action = extract_json(content)
                if "reasoning" not in raw_action:
                    raw_action["reasoning"] = "No explicit reasoning provided."
                break
            except Exception as e:
                print(f"  [step {step_num}] LLM error (attempt {attempt+1}): {e}", file=sys.stderr)
                if attempt == 2:
                    break

        if raw_action is None:
            consecutive_errors += 1
            if consecutive_errors >= 3:
                print(f"  [step {step_num}] 3 consecutive errors, giving up", file=sys.stderr)
                break
            # Try escalation as fallback
            raw_action = {"action_type": "escalate", "parameters": {}, "reasoning": "Fallback after parse error"}
        else:
            consecutive_errors = 0

        # 3. Send to env (OpenEnv format wraps action)
        try:
            step_resp = httpx.post(f"{BASE_URL}/step",
                                   json={"action": raw_action}, timeout=30)
            if step_resp.status_code != 200:
                print(f"  [step {step_num}] Step error: {step_resp.text}", file=sys.stderr)
                break
            step_data = step_resp.json()
        except Exception as e:
            print(f"  [step {step_num}] HTTP error: {e}", file=sys.stderr)
            break

        obs = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]

        # 4. Add to message history
        messages.append({"role": "assistant", "content": content})
        obs_msg = format_observation(obs, step_num + 1, reward)
        messages.append({"role": "user", "content": obs_msg})

        # Context window management: summarize old messages with hypothesis tracking
        if len(messages) > 20:
            # Build hypothesis log from old messages
            old_actions = []
            services_checked = set()
            hypotheses = []
            for m in messages[1:-14]:
                if m["role"] == "assistant":
                    try:
                        act = extract_json(m["content"])
                        at = act.get('action_type', '?')
                        params = act.get('parameters', {})
                        svc = params.get('service', '')
                        reasoning = act.get('reasoning', '')
                        old_actions.append(f"- {at}({json.dumps(params)})")
                        if svc:
                            services_checked.add(svc)
                        if reasoning:
                            hypotheses.append(reasoning[:100])
                    except Exception:
                        old_actions.append(f"- (parse error)")

            summary_parts = [
                "INVESTIGATION LOG:",
                f"Services already investigated: {', '.join(sorted(services_checked)) if services_checked else 'none'}",
                f"Actions taken ({len(old_actions)} steps):",
            ]
            summary_parts.extend(old_actions[-12:])
            if hypotheses:
                summary_parts.append(f"\nLatest reasoning: {hypotheses[-1]}")
            summary_parts.append("\nRemember: After fixing → verify_health → write_postmortem")

            summary = "\n".join(summary_parts)
            messages = [messages[0], {"role": "user", "content": summary}] + messages[-14:]

        if done:
            return obs.get("episode_score", 0.0)

    # Episode ended without done=True — get grader score
    grader_resp = httpx.get(f"{BASE_URL}/grader", timeout=30)
    return grader_resp.json()["episode_score"]


def main():
    parser = argparse.ArgumentParser(description="SREBench baseline agent")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON only (for /baseline endpoint)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY",
              os.environ.get("NVIDIA_API_KEY",
              os.environ.get("OPENROUTER_API_KEY", "")))
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=API_BASE)
    results = {}

    for task_id in TASKS:
        if not args.json:
            print(f"\n▶ Running {task_id}...", file=sys.stderr)
        score = run_episode(client, task_id)
        results[task_id] = round(score, 4)
        if not args.json:
            print(f"  Score: {score:.4f}", file=sys.stderr)

    if args.json:
        print(json.dumps(results))
    else:
        print("\n=== SREBench Baseline Results ===")
        for task_id, score in results.items():
            bar = "█" * int(score * 20)
            print(f"  {task_id:<28} {score:.4f}  {bar}")
        print(f"\n  Average: {sum(results.values()) / len(results):.4f}")


if __name__ == "__main__":
    main()
