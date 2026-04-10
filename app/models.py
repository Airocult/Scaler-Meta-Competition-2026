"""
SREBench data models — extends OpenEnv SDK base types for full compatibility.
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional, Any
from enum import Enum

from openenv.core.env_server.types import (
    Action as OpenEnvAction,
    Observation as OpenEnvObservation,
    State as OpenEnvState,
)


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


class SREAction(OpenEnvAction):
    """SRE incident response action — extends OpenEnv Action."""
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
        "trace_request",
        "check_slo",
        "classify_severity",
        "update_status_page",
    ]
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = Field(default="")


class SREObservation(OpenEnvObservation):
    """SRE incident observation — extends OpenEnv Observation (inherits done, reward, metadata)."""
    timestamp: str
    alert_summary: str
    service_statuses: dict[str, ServiceStatus]
    last_action_result: str
    incident_phase: IncidentPhase
    available_actions: list[str]
    step_count: int
    time_elapsed_minutes: int
    hints_used: int
    # Detailed reward info (also available via top-level reward field)
    step_reward: float = 0.0
    cumulative_reward: float = 0.0
    episode_score: float = 0.0


class SREState(OpenEnvState):
    """Full environment state — extends OpenEnv State (inherits episode_id, step_count)."""
    task_id: str = ""
    incident_phase: str = "investigating"
    cumulative_reward: float = 0.0
    done: bool = False
    hints_used: int = 0
    root_cause_identified: bool = False
    fix_applied: bool = False
    resolution_verified: bool = False
    postmortem_written: bool = False
    severity_classified: bool = False
    status_page_updated: bool = False


# Backward-compatible aliases
Action = SREAction
Observation = SREObservation
