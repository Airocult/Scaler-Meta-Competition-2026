"""
SREBench FastAPI application — built on OpenEnv SDK's create_app
with custom SRE-specific endpoints layered on top.
"""
import json
import os
import subprocess
import sys

import yaml
from fastapi import HTTPException

from openenv.core.env_server.http_server import create_app

from app.environment import SREBenchEnvironment
from app.models import SREAction, SREObservation


# Singleton environment — shared across all HTTP requests
_env = SREBenchEnvironment()

# Create the OpenEnv-standard app (provides /reset, /step, /state, /schema, /health, /docs, etc.)
app = create_app(
    env=lambda: _env,
    action_cls=SREAction,
    observation_cls=SREObservation,
    env_name="srebench",
)


# ── Custom SREBench endpoints ─────────────────────────────


@app.get("/tasks")
async def tasks():
    """Returns task list + full action schema for agent consumption."""
    yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "openenv.yaml")
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return {
        "tasks": config["tasks"],
        "action_schema": config["action_space"],
        "observation_schema": config["observation_space"],
    }


@app.get("/grader")
async def grader():
    try:
        score = await _env.get_grader_score()
        return {"episode_score": score}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/baseline")
async def baseline():
    """Triggers the baseline inference script programmatically."""
    api_key = os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OPENROUTER_API_KEY (or OPENAI_API_KEY) not set.",
        )

    baseline_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "baseline", "run_baseline.py")
    try:
        result = subprocess.run(
            [sys.executable, baseline_path, "--json"],
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ},
        )
        if result.returncode != 0:
            raise HTTPException(500, f"Baseline script failed: {result.stderr}")
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "Baseline script timed out after 300s")
    except json.JSONDecodeError:
        raise HTTPException(500, "Baseline script returned invalid JSON")


def serve():
    """Entry point for `server` console_script (used by openenv serve)."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
