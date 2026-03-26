from pydantic import BaseModel, Field
from typing import Literal, Optional, Any
from enum import Enum


class ServiceStatus(BaseModel):
    name: str
    status: Literal["healthy", "degraded", "down", "unknown"]
    error_rate: float  # 0.0–1.0
    latency_p99_ms: int
    restarts_last_hour: int


class IncidentPhase(str, Enum):
    INVESTIGATING = "investigating"
    MITIGATING = "mitigating"
    VERIFYING = "verifying"
    RESOLVED = "resolved"


class Action(BaseModel):
    action_type: Literal[
        "read_logs",
        "check_metrics",
        "list_services",
        "check_alerts",
        "check_deployments",
        "check_dependencies",
        "run_diagnostic",
        "apply_fix",
        "verify_health",
        "write_postmortem",
        "escalate",
    ]
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = Field(default="")


class Observation(BaseModel):
    timestamp: str
    alert_summary: str
    service_statuses: dict[str, ServiceStatus]
    last_action_result: str
    incident_phase: IncidentPhase
    available_actions: list[str]
    step_count: int
    time_elapsed_minutes: int
    hints_used: int  # tracks if agent called escalate


class Reward(BaseModel):
    step_reward: float  # reward for this step
    cumulative_reward: float  # total so far
    episode_score: float  # final 0.0–1.0 (only meaningful when done=True)
    done: bool
    info: dict[str, Any]


class ResetRequest(BaseModel):
    task_id: str
    seed: Optional[int] = 42  # reproducibility default


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
