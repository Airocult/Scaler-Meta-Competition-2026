"""
SREBench Inference Script
===================================
MANDATORY
- Uses OpenAI Client for all LLM calls
- Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment
- Emits [START], [STEP], [END] structured stdout logs
- Named inference.py in the root directory
"""

import json
import os
import re
import sys
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Set these in your system or .env file ────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")

# Optional (for local Docker deployment)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# SREBench Space URL
SPACE_URL = os.getenv("SREBENCH_URL", "https://neuralninja110-srebench.hf.space")

BENCHMARK = "srebench"
SEED = 42

TASKS = [
    "task1_memory_leak",
    "task2_db_cascade",
    "task3_race_condition",
    "task4_dns_failure",
    "task5_cert_expiry",
    "task6_network_partition",
]

MAX_STEPS_MAP = {
    "task1_memory_leak": 20,
    "task2_db_cascade": 30,
    "task3_race_condition": 40,
    "task4_dns_failure": 25,
    "task5_cert_expiry": 35,
    "task6_network_partition": 40,
}

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
- NEVER apply_fix(fix_type="restart") as first action. Always diagnose first.
- When multiple services are degraded, the ROOT CAUSE is usually a LEAF service (databases). Errors CASCADE UPWARD.
- If errors started at time T, check what was deployed at T±5 minutes — timing correlation = likely cause.
- After ANY fix, ALWAYS run verify_health then write_postmortem. These are required for full score.

## PATTERN MATCHING (match symptoms to diagnostic type)
- "OOMKilled", "heap space", "memory exceeded" → memory leak → fix_type="restart"
- "connection pool exhausted", "HikariPool" → DB pool issue → trace to DATABASE → fix_type="increase_pool_size"
- "SSL handshake", "certificate expired", "TLS failed" → expired TLS cert → run_diagnostic type="tls" → fix_type="renew_cert"
- "DNS resolution failed", "NXDOMAIN", "stale DNS" → DNS cache → run_diagnostic type="dns" → fix_type="flush_dns"
- "connection timed out", "network unreachable", "split-brain" → network partition → run_diagnostic type="iptables" → fix_type="rollback_deploy" then reconcile_data
- Error spike correlating with deploy → config change → run_diagnostic type="config_diff" WITH deploy_id → fix_type="rollback"

## CASCADE TRACING
When alert points to a gateway/frontend service:
1. check_dependencies on the alerted service
2. Trace toward LEAF services (databases)
3. check_metrics on EACH service in chain — highest error_rate = likely root cause
Example: api-gateway → order-service → payment-service → payment-db(pool exhausted) = ROOT CAUSE

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
- check_deployments: {"last_n": 5} or {"service": "<name>"} — Recent deploys
- check_dependencies: {"service": "<name>"} — Service dependency graph
- run_diagnostic: {"service": "<name>", "type": "<diag_type>"} — Run diagnostics
- apply_fix: {"service": "<name>", "fix_type": "<type>"} — Apply remediation
- verify_health: {"service": "<name>"} or {} — Verify resolution
- write_postmortem: {"content": "<detailed text>"} — Document the incident
- escalate: {} — Get a hint (costs points)

Output valid JSON only — no markdown fences, no commentary outside the JSON object.
"""


# ── Logging helpers (mandatory format) ───────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM helpers ──────────────────────────────────────────────────────────────

def extract_json(content: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences and extra text."""
    content = content.strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No valid JSON found in: {content[:200]}")


def call_llm(client: OpenAI, messages: list) -> str:
    """Call the LLM via OpenAI client and return content."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            top_p=0.9,
            max_tokens=4096,
            stream=False,
        )
    except Exception as e:
        if "max_tokens" in str(e) or "unsupported_parameter" in str(e):
            # Reasoning models need max_completion_tokens and don't support temperature/top_p
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_completion_tokens=4096,
                stream=False,
            )
        else:
            raise
    return completion.choices[0].message.content


def format_observation(obs: dict, step_num: int, reward: float, is_initial: bool = False) -> str:
    """Format observation to highlight key signals for the LLM."""
    parts = []
    if is_initial:
        parts.append(f"INCIDENT ALERT: {obs.get('alert_summary', 'Unknown incident')}")
        parts.append(f"\nAvailable actions: {obs.get('available_actions', [])}")
    else:
        result = obs.get("last_action_result", "")
        parts.append(f"Result: {result}")
        parts.append(f"Phase: {obs.get('incident_phase', '?')} | Step reward: {reward}")

    statuses = obs.get("service_statuses", {})
    degraded = []
    healthy = []
    items = statuses.values() if isinstance(statuses, dict) else statuses
    for s in items:
        if isinstance(s, str):
            healthy.append(s)
            continue
        name = s.get("name", "?")
        status = s.get("status", "")
        err = s.get("error_rate", 0)
        lat = s.get("latency_p99_ms", 0)
        restarts = s.get("restarts_last_hour", 0)
        pool = s.get("connection_pool_usage", 0)
        if status in ("degraded", "down"):
            extras = []
            if restarts > 0:
                extras.append(f"restarts={restarts}")
            if pool and pool > 0.8:
                extras.append(f"pool={pool} CRITICAL")
            extra_str = f", {', '.join(extras)}" if extras else ""
            degraded.append(f"  {name}: {status} (err={err}, lat={lat}ms{extra_str})")
        else:
            healthy.append(name)

    if degraded:
        parts.append("\nDEGRADED/DOWN services (investigate these):")
        parts.extend(degraded)
    if healthy:
        parts.append(f"Healthy: {', '.join(healthy)}")

    result_text = obs.get("last_action_result", "")
    if any(kw in result_text.lower() for kw in ["ssl", "tls", "certificate", "cert"]):
        parts.append("\nTLS/cert keywords detected -> consider run_diagnostic type='tls'")
    if any(kw in result_text.lower() for kw in ["dns", "nxdomain", "getaddrinfo", "stale ip"]):
        parts.append("\nDNS keywords detected -> consider run_diagnostic type='dns'")
    if any(kw in result_text.lower() for kw in ["connection pool", "hikaripool", "pool exhausted"]):
        parts.append("\nConnection pool issue -> trace to the DATABASE service (leaf node)")
    if any(kw in result_text.lower() for kw in ["network unreachable", "connection timed out", "iptables", "split-brain"]):
        parts.append("\nNetwork partition keywords -> consider run_diagnostic type='iptables'")

    if step_num > 0 and step_num % 6 == 0:
        parts.append("\nReminder: Have you checked deployments? Correlated timing? Run diagnostics?")
    if step_num > 0 and step_num % 10 == 0:
        parts.append("Reminder: After fixing root cause, ALWAYS verify_health then write_postmortem.")

    return "\n".join(parts)


# ── Main episode runner ──────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: str) -> float:
    """Run a single task episode. Returns the grader score."""
    max_steps = MAX_STEPS_MAP[task_id]

    # Reset environment
    reset_resp = httpx.post(
        f"{SPACE_URL}/reset",
        json={"task_id": task_id, "seed": SEED},
        timeout=30,
    )
    reset_resp.raise_for_status()
    obs = reset_resp.json()["observation"]

    initial_msg = format_observation(obs, 0, 0.0, is_initial=True)
    initial_msg += "\n\nFollow the investigation methodology. What do you do first?"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_msg},
    ]

    rewards: List[float] = []
    steps_taken = 0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step_num in range(1, max_steps + 1):
            # Get LLM action
            raw_action = None
            content = ""
            for attempt in range(3):
                try:
                    content = call_llm(client, messages)
                    if not content or not content.strip():
                        continue
                    raw_action = extract_json(content)
                    if "reasoning" not in raw_action:
                        raw_action["reasoning"] = "No explicit reasoning provided."
                    break
                except Exception as e:
                    print(f"[DEBUG] step={step_num} attempt={attempt+1} error: {e}", file=sys.stderr)
                    if attempt == 2:
                        break

            if raw_action is None:
                raw_action = {
                    "action_type": "escalate",
                    "parameters": {},
                    "reasoning": "Fallback after parse error",
                }

            action_str = f"{raw_action.get('action_type', '?')}({json.dumps(raw_action.get('parameters', {}))})"

            # Step environment
            try:
                step_resp = httpx.post(
                    f"{SPACE_URL}/step",
                    json={"action": raw_action},
                    timeout=30,
                )
                step_resp.raise_for_status()
                step_data = step_resp.json()
            except Exception as e:
                print(f"[DEBUG] HTTP step error: {e}", file=sys.stderr)
                log_step(step=step_num, action=action_str, reward=0.0, done=False, error=str(e))
                rewards.append(0.0)
                steps_taken = step_num
                break

            obs = step_data["observation"]
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", False)
            error = obs.get("last_action_error") if isinstance(obs, dict) else None

            rewards.append(reward)
            steps_taken = step_num

            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error)

            # Update message history
            messages.append({"role": "assistant", "content": content})
            obs_msg = format_observation(obs, step_num, reward)
            messages.append({"role": "user", "content": obs_msg})

            # Context window management
            if len(messages) > 20:
                old_actions = []
                services_checked = set()
                for m in messages[1:-14]:
                    if m["role"] == "assistant":
                        try:
                            act = extract_json(m["content"])
                            at = act.get("action_type", "?")
                            params = act.get("parameters", {})
                            svc = params.get("service", "")
                            old_actions.append(f"- {at}({json.dumps(params)})")
                            if svc:
                                services_checked.add(svc)
                        except Exception:
                            old_actions.append("- (parse error)")

                summary_parts = [
                    "INVESTIGATION LOG:",
                    f"Services checked: {', '.join(sorted(services_checked)) if services_checked else 'none'}",
                    f"Actions taken ({len(old_actions)} steps):",
                ]
                summary_parts.extend(old_actions[-12:])
                summary_parts.append("\nRemember: After fixing -> verify_health -> write_postmortem")
                summary = "\n".join(summary_parts)
                messages = [messages[0], {"role": "user", "content": summary}] + messages[-14:]

            if done:
                break

        # Get final grader score
        grader_resp = httpx.get(f"{SPACE_URL}/grader", timeout=30)
        grader_resp.raise_for_status()
        score = grader_resp.json().get("episode_score", 0.0)

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} exception: {exc}", file=sys.stderr)
        score = sum(rewards) / max(1, len(rewards)) if rewards else 0.001
        score = min(max(score, 0.001), 0.999)
        log_end(
            success=False,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )
        return score

    score = min(max(score, 0.001), 0.999)
    success = score > 0.001

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def main() -> None:
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
        timeout=120.0,
    )

    results = {}
    for task_id in TASKS:
        score = run_task(client, task_id)
        results[task_id] = round(score, 4)

    print("\n=== SREBench Final Scores ===", file=sys.stderr)
    for task_id, score in results.items():
        print(f"  {task_id:<30} {score:.4f}", file=sys.stderr)
    avg = sum(results.values()) / len(results)
    print(f"  {'Average':<30} {avg:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
