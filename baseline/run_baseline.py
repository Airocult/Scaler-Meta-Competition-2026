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
MODEL = os.getenv("BASELINE_MODEL", "deepseek-ai/deepseek-v3.1-terminus")
API_BASE = os.getenv("BASELINE_API_BASE", "https://integrate.api.nvidia.com/v1")

SYSTEM_PROMPT = """You are an expert SRE on-call engineer.
You will receive an incident alert and must investigate and resolve it.
On each turn, output a JSON action in this exact format:
{
  "action_type": "<one of the available actions>",
  "parameters": { ... },
  "reasoning": "Your step-by-step reasoning before acting"
}

Available action_types:
- read_logs: Read logs for a service. Params: {"service": "<name>"}
- check_metrics: Check metrics. Params: {"service": "<name>", "metric": "<type>"}
- list_services: List all services and their statuses. Params: {}
- check_alerts: View active alerts. Params: {}
- check_deployments: View recent deployments. Params: {"last_n": 5} or {"service": "<name>"}
- check_dependencies: View service dependency graph. Params: {"service": "<name>"}
- run_diagnostic: Run diagnostic on a service. Params: {"service": "<name>", "type": "<diag_type>"} (optionally "deploy_id")
- apply_fix: Apply a fix. Params: {"service": "<name>", "fix_type": "<type>"} (optionally "deploy_id")
- verify_health: Verify system health. Params: {"service": "<name>"} or {}
- write_postmortem: Write incident postmortem. Params: {"content": "<detailed postmortem>"}
- escalate: Get a hint. Params: {}

Always think carefully about the dependency graph before applying fixes.
Prefer to read_logs and check_metrics before applying any fix.
Only output valid JSON — no markdown, no commentary outside the JSON.
"""


def call_llm_streaming(client: OpenAI, messages: list) -> str:
    """Call the LLM and return the content (non-streaming for reliability)."""
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        top_p=0.7,
        max_tokens=8192,
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


def run_episode(client: OpenAI, task_id: str) -> float:
    """Runs one full episode. Returns final grader score."""
    # 1. Reset
    reset_resp = httpx.post(f"{BASE_URL}/reset",
                            json={"task_id": task_id, "seed": SEED},
                            timeout=30)
    obs = reset_resp.json()["observation"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"Incident alert: {obs['alert_summary']}\n"
            f"Available actions: {obs['available_actions']}\n"
            f"Service statuses: {json.dumps(obs['service_statuses'], indent=2)}\n"
            "What do you do first?"}
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
        messages.append({"role": "user", "content":
            f"Result: {obs['last_action_result']}\n"
            f"Incident phase: {obs['incident_phase']}\n"
            f"Step reward: {reward}\n"
            f"Service statuses: {json.dumps(obs['service_statuses'], indent=2)}\n"
            + (f"Available actions: {obs['available_actions']}"
               if step_num % 3 == 0 else "")
        })

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

    api_key = os.environ.get("NVIDIA_API_KEY",
              os.environ.get("OPENROUTER_API_KEY",
              os.environ.get("OPENAI_API_KEY", "")))
    if not api_key:
        print("ERROR: NVIDIA_API_KEY environment variable not set.", file=sys.stderr)
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
