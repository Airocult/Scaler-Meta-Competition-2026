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
  - torchforge
---

# SREBench — On-Call Incident Response OpenEnv Environment

**Built on the [OpenEnv SDK](https://github.com/meta-pytorch/OpenEnv) · Compatible with [torchforge](https://github.com/meta-pytorch/torchforge) GRPO training**

## What is this?

SREBench is an OpenEnv-compliant reinforcement learning environment that simulates on-call SRE (Site Reliability Engineering) incident response. The agent receives a production alert at 3 AM and must investigate services, read logs and metrics, trace dependency chains, identify the root cause, apply the correct remediation, verify resolution, and write a postmortem — all through a structured action API.

### OpenEnv Integration

SREBench extends the official OpenEnv SDK base classes:
- **`SREAction(Action)`** — 11 SRE action types with parameters and reasoning
- **`SREObservation(Observation)`** — Rich incident state with service topology
- **`SREState(State)`** — Full environment state for checkpointing
- **`SREBenchEnvironment(Environment)`** — Gym-style `reset()`/`step()` interface

Standard OpenEnv endpoints (`/reset`, `/step`, `/state`, `/schema`, `/ws`, `/health`) are
auto-generated via `create_app()`, with custom SRE endpoints (`/tasks`, `/grader`, `/baseline`)
layered on top.

### torchforge GRPO Training

SREBench works out-of-the-box with torchforge's Group Relative Policy Optimization:
```bash
torchforge grpo --config examples/torchforge_grpo/config.yaml --env-url ws://localhost:7860
```
See [`examples/torchforge_grpo/`](examples/torchforge_grpo/) for full training config.

## Environment Design

### Service Topology

SREBench simulates a microservices architecture with **8 services** forming a directed acyclic graph:

```
api-gateway → auth-service → user-db
            → order-service → inventory-service → inventory-db
                            → payment-service   → payment-db
```

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
| `done` | bool | Whether the episode has terminated (OpenEnv standard) |
| `reward` | float | Step reward signal (OpenEnv standard) |

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
| `wrong_fix_applied` | -0.10 | Applied fix to wrong service/type |
| `destructive_action` | -0.20 | Destructive action on healthy service |
| `escalate_used` | -0.05 | Hint used (per use) |
| `repeated_same_action` | -0.05 | Same action+params repeated |

## Tasks

### Task 1: Memory Leak OOM Kill (Easy)
- **Scenario**: `order-service` is OOM-killing every ~4 minutes
- **Root cause**: Memory leak → just need to restart the correct service
- **Baseline score**: ~0.62

### Task 2: Cascading DB Pool Exhaustion (Medium)
- **Scenario**: `payment-db` pool exhausted → cascading failures through payment-service → order-service → api-gateway
- **Root cause**: Must trace dependency chain, not just fix the surface symptom
- **Baseline score**: ~0.43

### Task 3: Distributed Race Condition via Config Change (Hard)
- **Scenario**: Config deploy reduced `redis.lock_timeout_ms` causing intermittent races
- **Root cause**: Must correlate deploy timeline, diff config, rollback
- **Baseline score**: ~0.38

## Quickstart

### Docker
```bash
docker build -t srebench .
docker run -p 7860:7860 srebench
```

### Local
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### OpenEnv Client (Python)
```python
from client import SREBenchEnv
from app.models import SREAction

async with SREBenchEnv(base_url="ws://localhost:7860") as env:
    result = await env.reset(task_id="task1_memory_leak", seed=42)
    while not result.done:
        action = SREAction(action_type="list_services", reasoning="investigating")
        result = await env.step(action)
```

## API Endpoints

### OpenEnv Standard
| Endpoint | Method | Description |
|:---|:---|:---|
| `/reset` | POST | Reset environment `{"task_id": "...", "seed": 42}` |
| `/step` | POST | Take action `{"action": {"action_type": "...", "parameters": {}, "reasoning": "..."}}` |
| `/state` | GET | Current environment state |
| `/schema` | GET | JSON schemas for action/observation types |
| `/metadata` | GET | Environment metadata |
| `/health` | GET | Health check |
| `/ws` | WebSocket | Persistent session for RL training loops |

### SREBench Custom
| Endpoint | Method | Description |
|:---|:---|:---|
| `/tasks` | GET | List all tasks with action/observation schemas |
| `/grader` | GET | Get current episode grader score |
| `/baseline` | POST | Run baseline inference (requires `OPENROUTER_API_KEY`) |

## Baseline

```bash
export OPENROUTER_API_KEY=sk-or-...
python baseline/run_baseline.py
```

| Task | Model | Score |
|:---|:---|:---|
| task1_memory_leak | nemotron-3-super-120b | 0.62 |
| task2_db_cascade | nemotron-3-super-120b | 0.43 |
| task3_race_condition | nemotron-3-super-120b | 0.38 |