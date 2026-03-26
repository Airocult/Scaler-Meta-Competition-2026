---
title: SREBench
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - sre
  - aiops
---

# SREBench — On-Call Incident Response Environment

## What is this?

SREBench is an OpenEnv-compliant reinforcement learning environment that simulates on-call SRE (Site Reliability Engineering) incident response. The agent receives a production alert at 3 AM and must investigate services, read logs and metrics, trace dependency chains, identify the root cause, apply the correct remediation, verify resolution, and write a postmortem — all through a structured action API. This environment targets a real-world task that every tech company desperately needs to automate, providing a challenging and realistic evaluation benchmark for AI agents.

## Environment Design

### Service Topology

SREBench simulates a microservices architecture with **8 services** forming a directed acyclic graph (DAG):

```
api-gateway → auth-service → user-db
            → order-service → inventory-service → inventory-db
                            → payment-service   → payment-db
```

The topology is fixed across all tasks — only service **statuses** change per scenario.

### Action Space

| Action Type | Parameters | Description |
|:---|:---|:---|
| `read_logs` | `service` | Read recent log entries for a service |
| `check_metrics` | `service`, `metric` | Check metrics (error_rate, latency, memory, cpu, etc.) |
| `list_services` | — | List all services with their current status |
| `check_alerts` | — | View active alert notifications |
| `check_deployments` | `service` (opt), `last_n` (opt) | View recent deployment history |
| `check_dependencies` | `service` | View upstream/downstream dependencies |
| `run_diagnostic` | `service`, `type`, `deploy_id` (opt) | Run a diagnostic check on a service |
| `apply_fix` | `service`, `fix_type`, `deploy_id` (opt) | Apply a remediation action |
| `verify_health` | `service` (opt) | Check if services are healthy after a fix |
| `write_postmortem` | `content` | Document the incident with root cause analysis |
| `escalate` | — | Request a hint (costs points) |

Every action requires a `reasoning` field (min 10 chars) — the agent's chain-of-thought.

### Observation Space

| Field | Type | Description |
|:---|:---|:---|
| `timestamp` | string | Current simulation time (ISO 8601) |
| `alert_summary` | string | Description of the active incident |
| `service_statuses` | dict | Status, error_rate, latency, restarts for all 8 services |
| `last_action_result` | string | Detailed text result of the last action taken |
| `incident_phase` | enum | `investigating` → `mitigating` → `verifying` → `resolved` |
| `available_actions` | list | All valid action types |
| `step_count` | int | Current step number |
| `time_elapsed_minutes` | int | Simulated time elapsed |
| `hints_used` | int | Number of `escalate` calls |

### Reward Structure

| Event | Reward | Description |
|:---|:---|:---|
| Base step | -0.01 | Time pressure — every step costs |
| `info_gathered` | +0.02 | Gathered useful investigation data |
| `root_cause_progress` | +0.15 | Traced one hop toward root cause |
| `root_cause_identified` | +0.30 | Correctly identified the exact cause |
| `fix_applied_correctly` | +0.30 | Applied the right remediation |
| `resolution_verified` | +0.15 | Confirmed system healthy after fix |
| `postmortem_written` | +0.10 | Wrote meaningful postmortem |
| `postmortem_quality` | +0.05 | Postmortem mentions root cause term |
| `wrong_fix_applied` | -0.10 | Applied fix to wrong service/type |
| `destructive_action` | -0.20 | Destructive action (e.g., restart healthy service) |
| `escalate_used` | -0.05 | Hint used (per use) |
| `repeated_same_action` | -0.05 | Same action+params repeated consecutively |

## Tasks

### Task 1: Memory Leak OOM Kill (Easy)
- **Scenario**: `order-service` is OOM-killing every ~4 minutes. All other services are healthy.
- **What the agent must figure out**: Identify that `order-service` is the failing service via logs/metrics, apply a restart, verify health, and write a postmortem.
- **Grader criteria**: Correct service identified (35%), correct fix type (25%), resolution verified (20%), postmortem written (10%), time bonus (10%), minus escalation penalty.
- **Expected baseline score**: ~0.62

### Task 2: Cascading DB Pool Exhaustion (Medium)
- **Scenario**: `payment-db` connection pool is exhausted → cascading failures through `payment-service` → `order-service` → `api-gateway` (visible 503s).
- **What the agent must figure out**: Trace the dependency chain from `api-gateway` down to `payment-db`, identify the pool exhaustion as root cause, increase the pool size (not just restart), verify, and postmortem.
- **Grader criteria**: Traced to payment-service (15%), traced to payment-db (25%), correct fix (30%), verified (15%), postmortem (10%), time bonus (5%). **Capped at 0.35 if agent fixes the wrong surface service.**
- **Expected baseline score**: ~0.43

### Task 3: Distributed Race Condition via Config Change (Hard)
- **Scenario**: An `inventory-service` deploy 12 minutes ago changed `redis.lock_timeout_ms` from 5000ms → 500ms, causing race conditions under load. Intermittent 5xx spike.
- **What the agent must figure out**: Notice the error spike timing, check deploy history, examine the config diff, understand the lock timeout reduction causes races, rollback the config, verify, and write a postmortem mentioning "lock_timeout" or "race condition".
- **Grader criteria**: Error spike noticed (10%), deploy identified (15%), config diff examined (20%), correct rollback (25%), errors verified ceased (15%), postmortem mentions root cause (15%). **Extra penalty if agent restarts >2 services without checking deploys.**
- **Expected baseline score**: ~0.38

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/srebench.git
cd srebench
docker build -t srebench .
docker run -p 7860:7860 srebench
# Then: curl http://localhost:7860/health
```

Or run locally:

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## Running the Baseline

```bash
export OPENROUTER_API_KEY=sk-or-...   # Get from https://openrouter.ai/keys
python baseline/run_baseline.py
```

## Baseline Scores

| Task | Model | Score |
|:---|:---|:---|
| task1_memory_leak | nemotron-3-super-120b | 0.62 |
| task2_db_cascade | nemotron-3-super-120b | 0.43 |
| task3_race_condition | nemotron-3-super-120b | 0.38 |

## API Endpoints

| Endpoint | Method | Description |
|:---|:---|:---|
| `/reset` | POST | Reset environment with `{task_id, seed}` |
| `/step` | POST | Take an action `{action_type, parameters, reasoning}` |
| `/state` | GET | Get current environment state |
| `/tasks` | GET | List all tasks with action schema |
| `/grader` | GET | Get current episode grader score |
| `/baseline` | POST | Run baseline inference (requires OPENROUTER_API_KEY) |
| `/health` | GET | Health check |

## HF Space

Deployed at: [https://huggingface.co/spaces/YOUR_USERNAME/srebench](https://huggingface.co/spaces/YOUR_USERNAME/srebench)