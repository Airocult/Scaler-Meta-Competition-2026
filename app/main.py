"""
FastAPI application — all OpenEnv endpoints.
"""
import json
import os
import subprocess
import sys

import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from app.environment import SREBenchEnvironment
from app.models import Action, Observation, ResetRequest, StepResponse

env = SREBenchEnvironment()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # startup/shutdown hooks if needed


app = FastAPI(
    title="SREBench",
    description="On-Call Incident Response OpenEnv Environment",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/reset", response_model=Observation)
async def reset(request: ResetRequest):
    try:
        return await env.reset(request.task_id, request.seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(action: Action):
    try:
        obs, reward = await env.step(action)
        return StepResponse(observation=obs, reward=reward)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
async def state():
    return await env.state()


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
        score = await env.get_grader_score()
        return {"episode_score": score}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/baseline")
async def baseline():
    """
    Triggers the baseline inference script programmatically.
    Runs all 3 tasks with seed=42, returns scores.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OPENROUTER_API_KEY (or OPENAI_API_KEY) not set. Set it in the environment to run the baseline.",
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


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "srebench", "version": "1.0.0"}
