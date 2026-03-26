#!/usr/bin/env python3
"""Runner script for baseline with NVIDIA API config."""
import os
import sys

os.environ["NVIDIA_API_KEY"] = "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"
os.environ["BASELINE_MODEL"] = "deepseek-ai/deepseek-v3.1-terminus"
os.environ["BASELINE_API_BASE"] = "https://integrate.api.nvidia.com/v1"
os.environ["SREBENCH_URL"] = "http://localhost:7860"

# Run one task at a time and save results incrementally
import json
import httpx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_baseline import run_episode
from openai import OpenAI

TASKS = ["task1_memory_leak", "task2_db_cascade", "task3_race_condition",
         "task4_dns_failure", "task5_cert_expiry", "task6_network_partition"]
SEED = 42

client = OpenAI(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url=os.environ["BASELINE_API_BASE"],
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
