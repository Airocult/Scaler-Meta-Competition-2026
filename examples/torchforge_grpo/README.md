# torchforge GRPO Training for SREBench

Train an LLM agent to handle SRE incidents using [torchforge](https://github.com/meta-pytorch/torchforge) 
Group Relative Policy Optimization (GRPO) with SREBench as the OpenEnv environment.

## Prerequisites

- **GPU cluster** with ≥ 2 GPUs (H100/A100 recommended)
- PyTorch 2.7+ with [Monarch](https://github.com/pytorch/monarch) backend
- [torchforge](https://github.com/meta-pytorch/torchforge) installed
- SREBench server running (locally or remote)

## Quick Start

```bash
# 1. Start the SREBench environment server
cd /path/to/Quant-Quasars-2026
uvicorn app.main:app --host 0.0.0.0 --port 7860

# 2. Run GRPO training (requires 2+ GPUs)
torchforge grpo \
  --config examples/torchforge_grpo/config.yaml \
  --env-url ws://localhost:7860
```

## Architecture

```
┌──────────────────┐     WebSocket     ┌──────────────────┐
│   torchforge     │◄──────/ws────────►│   SREBench       │
│   GRPO Trainer   │                   │   OpenEnv Server  │
│                  │  reset(task_id,   │                  │
│  - Policy LLM    │    seed)          │  - 3 SRE tasks   │
│  - Value Head    │  step(action)     │  - 8 services    │
│  - Reward Model  │  ──────────────►  │  - Deterministic │
│                  │  observation +    │    graders       │
│                  │  reward + done    │                  │
│                  │  ◄──────────────  │                  │
└──────────────────┘                   └──────────────────┘
```

## Training Config

See [config.yaml](config.yaml) for the full GRPO configuration.

Key hyperparameters:
- **Group size**: 4 (generates 4 completions per prompt, ranks them by reward)
- **KL penalty**: 0.01 (prevents policy from diverging too far from reference)
- **Learning rate**: 1e-6 (conservative for policy optimization)
- **Max episode steps**: 20/30/40 (per task difficulty)

## Reward Signal

SREBench provides a dense reward signal ideal for GRPO:

| Event | Reward |
|-------|--------|
| Useful investigation step | +0.05 to +0.10 |
| Root cause identification | +0.15 |
| Correct fix applied | +0.25 |
| Resolution verified | +0.15 |
| Wrong service targeted | -0.10 |
| Repeated action | -0.05 |
| Step penalty (per step) | -0.01 |

Final episode score (0.0–1.0) from the deterministic grader is used as
the GRPO reward for ranking completions within each group.

## Using the Client

```python
from client import SREBenchEnv
from app.models import SREAction

async with SREBenchEnv(base_url="ws://localhost:7860") as env:
    result = await env.reset(task_id="task1_memory_leak", seed=42)
    
    while not result.done:
        # Your policy generates an action
        action = SREAction(
            action_type="read_logs",
            parameters={"service": "order-service"},
            reasoning="Investigating OOM alerts on order-service",
        )
        result = await env.step(action)
        print(f"Step reward: {result.reward}, Done: {result.done}")
```
