#!/usr/bin/env python3
"""
Baseline agent: Runs each task with ReAct-style prompting via OpenRouter API.
Reads OPENROUTER_API_KEY from environment (falls back to OPENAI_API_KEY).
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
TASKS = ["task1_memory_leak", "task2_db_cascade", "task3_race_condition"]
SEED = 42
MODEL = os.getenv("BASELINE_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

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


def run_episode(client: OpenAI, task_id: str) -> float:
    """Runs one full episode. Returns final grader score."""
    # 1. Reset
    reset_resp = httpx.post(f"{BASE_URL}/reset",
                            json={"task_id": task_id, "seed": SEED},
                            timeout=30)
    obs = reset_resp.json()

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
                 "task3_race_condition": 40}[task_id]

    for step_num in range(max_steps):
        # 2. Get LLM action
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.2,
            )
            content = completion.choices[0].message.content
            # Strip markdown fences if model wraps JSON in ```json ... ```
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                content = "\n".join(lines)
            raw_action = json.loads(content)
            # Ensure reasoning field is present
            if "reasoning" not in raw_action:
                raw_action["reasoning"] = "No explicit reasoning provided."
        except Exception as e:
            print(f"  [step {step_num}] LLM error: {e}", file=sys.stderr)
            break

        # 3. Send to env
        try:
            step_resp = httpx.post(f"{BASE_URL}/step",
                                   json=raw_action, timeout=30)
            if step_resp.status_code != 200:
                print(f"  [step {step_num}] Step error: {step_resp.text}", file=sys.stderr)
                break
            step_data = step_resp.json()
        except Exception as e:
            print(f"  [step {step_num}] HTTP error: {e}", file=sys.stderr)
            break

        obs = step_data["observation"]
        reward = step_data["reward"]

        # 4. Add to message history
        messages.append({"role": "assistant",
                         "content": completion.choices[0].message.content})
        messages.append({"role": "user", "content":
            f"Result: {obs['last_action_result']}\n"
            f"Incident phase: {obs['incident_phase']}\n"
            f"Step reward: {reward['step_reward']}\n"
            f"Service statuses: {json.dumps(obs['service_statuses'], indent=2)}\n"
            + (f"Available actions: {obs['available_actions']}"
               if step_num % 3 == 0 else "")
        })

        if reward["done"]:
            return reward["episode_score"]

    # Episode ended without done=True — get grader score
    grader_resp = httpx.get(f"{BASE_URL}/grader", timeout=30)
    return grader_resp.json()["episode_score"]


def main():
    parser = argparse.ArgumentParser(description="SREBench baseline agent")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON only (for /baseline endpoint)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY (or OPENAI_API_KEY) environment variable not set.", file=sys.stderr)
        print("Get an OpenRouter key from https://openrouter.ai/keys", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE)
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
