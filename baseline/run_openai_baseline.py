#!/usr/bin/env python3
"""Runner script for baseline with OpenAI API config loaded from .env file."""
import os
import sys
import json

# Load .env file from project root
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_baseline import run_episode, MODEL, API_BASE
from openai import OpenAI

TASKS = ["task1_memory_leak", "task2_db_cascade", "task3_race_condition",
         "task4_dns_failure", "task5_cert_expiry", "task6_network_partition"]

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("ERROR: OPENAI_API_KEY not set. Create a .env file or set the environment variable.", file=sys.stderr)
    sys.exit(1)

print(f"Using model: {MODEL}")
print(f"Using API base: {API_BASE}")

client = OpenAI(
    api_key=api_key,
    base_url=API_BASE,
    timeout=120.0,
)

results = {}
results_file = "/tmp/baseline_results.json"

for task_id in TASKS:
    print(f"\n▶ Running {task_id}...", flush=True)
    try:
        score = run_episode(client, task_id)
        results[task_id] = round(score, 4)
        print(f"  Score: {score:.4f}", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        results[task_id] = -1.0

    # Save incrementally
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved to {results_file}]", flush=True)

print(f"\n=== Final Results ===")
for tid, s in results.items():
    bar = "█" * int(max(0, s) * 20)
    print(f"  {tid:<30} {s:.4f}  {bar}")
avg = sum(v for v in results.values() if v >= 0) / max(1, sum(1 for v in results.values() if v >= 0))
print(f"\n  Average: {avg:.4f}")
