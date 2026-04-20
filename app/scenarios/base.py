"""
Abstract base class for all scenarios.

Novel mechanics:
  - **Cascading degradation**: unresolved incidents get worse over time
    (error rates drift upward, latencies spike). The environment is
    non-stationary even without agent actions.
  - **Evidence tracking**: the grader records which *distinct* evidence
    sources the agent consulted before attempting a fix. Agents that
    gather broad evidence score higher than those that guess.
  - **Red-herring resilience**: mid-episode distracting alerts test
    whether the agent stays focused on the primary incident.
  - **Distributed tracing**: `trace_request` action reveals request flow
    through service graph, exposing root cause via span timing/errors.
  - **SLO / Error budget**: each service has an SLO target with a finite
    error budget that burns in real time. Creates observable urgency.
  - **Incident communication**: `classify_severity` and `update_status_page`
    test the communication / coordination side of incident response.
"""
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.data.service_graph import ServiceGraph
from app.data.slo import create_slo_tracker, ServiceSLO
from app.data.trace_templates import TraceGenerator
from app.reward import RewardShaper


class BaseScenario(ABC):
    task_id: str
    max_steps: int

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.step_count = 0
        self.incident_phase = IncidentPhase.INVESTIGATING
        self.cumulative_reward = 0.0
        self.hints_used = 0
        self.done = False
        self.graph = ServiceGraph()
        self.reward_shaper = RewardShaper()
        self._last_action: str | None = None
        self._last_params: dict | None = None
        self._action_history: list[dict] = []

        # Flags common to all scenarios
        self._root_cause_identified = False
        self._fix_applied = False
        self._resolution_verified = False
        self._postmortem_written = False

        # ── Novel: Evidence tracking ──────────────────────────────
        # Tracks distinct (action_type, service) pairs gathered before fix
        self._evidence_sources: set[str] = set()
        self._fix_attempted = False  # True once first apply_fix is called

        # ── Investigation gating ──────────────────────────────────
        # Minimum distinct evidence sources required before apply_fix is accepted
        self._min_evidence_required = 2

        # ── Novel: Degradation drift ──────────────────────────────
        # Services degrade further each step the incident is unresolved
        self._degradation_factor = 0.0  # increases per step

        # ── Novel: SLO / Error budget tracking ────────────────────
        self._slo_tracker = create_slo_tracker()
        self._slo_breaches_during_episode: list[str] = []
        self._fix_before_any_breach = False

        # ── Novel: Incident communication ─────────────────────────
        self._severity_classified = False
        self._severity_correct = False
        self._severity_value: str | None = None
        self._status_page_updated = False
        self._status_page_before_fix = False
        self._status_page_count = 0

    def _base_timestamp(self) -> str:
        """Incident start time."""
        return "2026-03-26T03:00:00Z"

    def _current_timestamp(self) -> str:
        base = datetime(2026, 3, 26, 3, 0, 0)
        current = base + timedelta(minutes=self.step_count * 2)
        return current.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _available_actions(self) -> list[str]:
        return [
            "read_logs", "check_metrics", "list_services", "check_alerts",
            "check_deployments", "check_dependencies", "run_diagnostic",
            "apply_fix", "verify_health", "write_postmortem", "escalate",
            "trace_request", "check_slo", "classify_severity", "update_status_page",
        ]

    def _build_observation(self, last_action_result: str) -> Observation:
        statuses = self._get_service_statuses()

        # ── Novel: Cascading degradation — services get worse over time ──
        if not self._fix_applied and self._degradation_factor > 0:
            for name, svc in statuses.items():
                if svc.status in ("degraded", "down"):
                    statuses[name] = ServiceStatus(
                        name=svc.name,
                        status=svc.status,
                        error_rate=min(0.99, round(svc.error_rate + self._degradation_factor, 2)),
                        latency_p99_ms=int(svc.latency_p99_ms * (1 + self._degradation_factor * 2)),
                        restarts_last_hour=svc.restarts_last_hour,
                    )

        obs = Observation(
            timestamp=self._current_timestamp(),
            alert_summary=self._get_alert_summary(),
            service_statuses=statuses,
            last_action_result=last_action_result,
            incident_phase=self.incident_phase,
            available_actions=self._available_actions(),
            step_count=self.step_count,
            time_elapsed_minutes=self.step_count * 2,
            hints_used=self.hints_used,
        )

        # ── Novel: Degradation warning when situation is worsening ──
        if self._degradation_factor >= 0.10 and not self._fix_applied:
            obs.last_action_result += (
                f"\n⚠️ DEGRADATION NOTICE: Incident severity increasing. "
                f"Error rates have risen {self._degradation_factor:.0%} since alert start."
            )

        # ── Novel: SLO burn rate warnings in observations ────────
        slo_warnings = []
        for svc_name, slo in self._slo_tracker.items():
            if slo.breached:
                slo_warnings.append(f"🔴 SLO BREACH: {slo.status_line()}")
            elif slo.budget_remaining < 0.50:
                slo_warnings.append(f"⚠️ SLO ALERT: {slo.status_line()}")
        if slo_warnings:
            obs.last_action_result += "\n" + "\n".join(slo_warnings)

        # ── Novel: Red-herring alert mid-episode (tests agent focus) ──
        import random as _rng_mod
        rng = _rng_mod.Random(self.seed + 9999)
        herring_step = rng.randint(3, max(4, self.max_steps // 3))
        if self.step_count == herring_step and not self._fix_applied:
            distractors = [
                "[NEW] WARN: Elevated slow-query count on user-db (p99 query time: 320ms). Possibly unrelated.",
                "[NEW] INFO: Scheduled maintenance window for auth-service starts in 45 minutes.",
                "[NEW] WARN: Disk usage on monitoring-stack at 78%. Consider cleanup.",
            ]
            distractor = distractors[rng.randint(0, len(distractors) - 1)]
            obs.alert_summary += f"\n{distractor}"

        return obs

    def _compute_reward(self, event: str, service: str = "") -> float:
        reward = self.reward_shaper.compute(
            event=event,
            step_count=self.step_count,
            max_steps=self.max_steps,
            previous_reward=self.cumulative_reward,
            service=service,
        )
        self.cumulative_reward += reward
        return reward

    def _record_evidence(self, action_type: str, service: str) -> None:
        """Track distinct evidence sources gathered before first fix attempt."""
        if not self._fix_attempted:
            self._evidence_sources.add(f"{action_type}:{service}")

    def _evidence_breadth_score(self) -> float:
        """Bonus for investigating multiple distinct sources before fixing.
        0 sources = 0.0, 1 = 0.02, 2 = 0.04, 3 = 0.06, 4+ = 0.08 (max)"""
        n = len(self._evidence_sources)
        return min(n * 0.02, 0.08)

    def _efficient_investigation_bonus(self) -> float:
        """Bonus for reaching root cause quickly relative to max steps.
        Rewards agents that identify the root cause efficiently.
        Returns 0.0–0.04 based on investigation speed."""
        if not self._root_cause_identified:
            return 0.0
        # Use current step count as proxy for when investigation concluded
        efficiency = 1.0 - (self.step_count / max(self.max_steps, 1))
        if efficiency > 0.7:
            return 0.04
        elif efficiency > 0.5:
            return 0.02
        elif efficiency > 0.3:
            return 0.01
        return 0.0

    def _blast_radius_bonus(self) -> float:
        """Bonus for assessing blast radius via dependency checks before fixing.
        Rewards agents that check downstream/upstream impact.
        Returns 0.0–0.03 based on dependency investigation breadth."""
        dep_checks = sum(
            1 for a in self._action_history
            if a.get("action_type") == "check_dependencies"
        )
        if dep_checks >= 2:
            return 0.03
        elif dep_checks >= 1:
            return 0.015
        return 0.0

    def _postmortem_quality_bonus(self, keywords: list[str]) -> float:
        """Score postmortem content for mentioning root-cause keywords.

        This is a novel grading mechanic: instead of binary "wrote postmortem",
        we evaluate whether the postmortem content demonstrates understanding
        of the root cause by checking for domain-specific keywords.

        Returns 0.0–0.06 based on keyword coverage.
        """
        if not self._postmortem_written:
            return 0.0
        # Find the postmortem content from action history
        content = ""
        for entry in self._action_history:
            if entry["action_type"] == "write_postmortem":
                content = entry["parameters"].get("content", "").lower()
                break
        if not content:
            return 0.0
        # Score based on keyword coverage
        matches = sum(1 for kw in keywords if kw.lower() in content)
        coverage = matches / max(len(keywords), 1)
        return round(min(coverage * 0.06, 0.06), 4)

    def _advance_degradation(self) -> None:
        """Worsen service health each step the incident is unresolved."""
        if not self._fix_applied:
            self._degradation_factor = min(
                0.30, self._degradation_factor + (0.02 * (self.step_count / self.max_steps + 0.5))
            )

    def _advance_slo_burns(self) -> None:
        """Advance SLO error budget burns based on current service statuses."""
        statuses = self._get_service_statuses()
        for svc_name, slo in self._slo_tracker.items():
            if slo.breached:
                continue
            err_rate = statuses.get(svc_name, ServiceStatus(
                name=svc_name, status="healthy", error_rate=0.01,
                latency_p99_ms=45, restarts_last_hour=0,
            )).error_rate
            slo.advance(err_rate, self.step_count, self.max_steps)
            if slo.breached and svc_name not in self._slo_breaches_during_episode:
                self._slo_breaches_during_episode.append(svc_name)

    def _correct_severity(self) -> str:
        """Override in subclass to set the correct severity for the scenario."""
        return "SEV2"  # default

    def _is_repeated_action(self, action: Action) -> bool:
        if (self._last_action == action.action_type
                and self._last_params == action.parameters):
            return True
        return False

    def apply_action(self, action: Action) -> tuple[Observation, float, bool]:
        """Process an action. Returns (observation, step_reward, done)."""
        if self.done:
            obs = self._build_observation("Episode already completed.")
            return obs, 0.0, True

        self.step_count += 1

        # Advance degradation drift (services get worse over time)
        self._advance_degradation()

        # Advance SLO error budget burns
        self._advance_slo_burns()

        # Track whether fix happens before any SLO breach
        if not self._fix_applied and not self._slo_breaches_during_episode:
            self._fix_before_any_breach = True

        # Track evidence gathering before fix attempts
        service = action.parameters.get("service", "")
        if action.action_type in ("list_services", "read_logs", "check_metrics",
                                   "check_alerts", "check_deployments",
                                   "check_dependencies", "run_diagnostic",
                                   "trace_request", "check_slo"):
            self._record_evidence(action.action_type, service)
        if action.action_type == "apply_fix":
            self._fix_attempted = True

        # Check for repeated action
        is_repeat = self._is_repeated_action(action)
        self._last_action = action.action_type
        self._last_params = action.parameters.copy()
        self._action_history.append({
            "action_type": action.action_type,
            "parameters": action.parameters.copy(),
            "step": self.step_count,
        })

        if is_repeat:
            repeat_penalty = self._compute_reward("repeated_same_action")
            obs = self._build_observation("Same action repeated — no new information.")
            # Check max steps
            if self.step_count >= self.max_steps:
                self.done = True
            return obs, repeat_penalty, self.done

        # ── Handle base-level actions (new mechanics) ─────────────
        if action.action_type == "trace_request":
            service = action.parameters.get("service", "api-gateway")
            trace_output = TraceGenerator.generate(
                self.task_id, service, self.seed, step_count=self.step_count
            )
            result = f"Distributed trace for request through {service}:\n{trace_output}"
            reward = self._compute_reward("info_gathered", service=service)
            obs = self._build_observation(result)
            if self.step_count >= self.max_steps:
                self.done = True
            return obs, reward, self.done

        if action.action_type == "check_slo":
            lines = ["SLO Status Dashboard:"]
            for svc_name, slo in self._slo_tracker.items():
                lines.append(f"  {slo.status_line()}")
            breaches = [s for s in self._slo_tracker.values() if s.breached]
            if breaches:
                lines.append(f"\n🔴 {len(breaches)} SLO(s) breached — prioritize affected services!")
            else:
                critical = [s for s in self._slo_tracker.values() if s.budget_remaining < 0.30]
                if critical:
                    lines.append(f"\n⚠️ {len(critical)} service(s) at critical error budget — act fast!")
            result = "\n".join(lines)
            reward = self._compute_reward("info_gathered", service="slo")
            obs = self._build_observation(result)
            if self.step_count >= self.max_steps:
                self.done = True
            return obs, reward, self.done

        if action.action_type == "classify_severity":
            sev = action.parameters.get("severity", "").upper()
            if sev not in ("SEV1", "SEV2", "SEV3", "SEV4"):
                result = f"Invalid severity '{sev}'. Must be SEV1, SEV2, SEV3, or SEV4."
                reward = self._compute_reward("no_effect")
            else:
                self._severity_classified = True
                self._severity_value = sev
                correct = self._correct_severity()
                self._severity_correct = (sev == correct)
                severity_descriptions = {
                    "SEV1": "Critical — complete service outage or data loss affecting all users",
                    "SEV2": "Major — significant degradation affecting many users",
                    "SEV3": "Minor — limited impact, workaround available",
                    "SEV4": "Low — cosmetic or minor issue, no user impact",
                }
                result = (f"Incident classified as {sev}: {severity_descriptions.get(sev, '')}\n"
                          f"Stakeholders notified. On-call escalation policy activated for {sev}.")
                reward = self._compute_reward("info_gathered", service="severity")
            obs = self._build_observation(result)
            if self.step_count >= self.max_steps:
                self.done = True
            return obs, reward, self.done

        if action.action_type == "update_status_page":
            status = action.parameters.get("status", "investigating")
            message = action.parameters.get("message", "")
            if len(message) < 20:
                result = "Status page update rejected — message must be at least 20 characters."
                reward = self._compute_reward("no_effect")
            else:
                self._status_page_updated = True
                self._status_page_count += 1
                if not self._fix_applied:
                    self._status_page_before_fix = True
                result = (f"Status page updated ({status}):\n"
                          f"  \"{message}\"\n"
                          f"  Visible to customers and stakeholders. Update #{self._status_page_count}.")
                reward = self._compute_reward("info_gathered", service="status_page")
            obs = self._build_observation(result)
            if self.step_count >= self.max_steps:
                self.done = True
            return obs, reward, self.done

        # ── Investigation gating: reject apply_fix without sufficient evidence ──
        if action.action_type == "apply_fix" and len(self._evidence_sources) < self._min_evidence_required:
            n = len(self._evidence_sources)
            result = (
                f"Fix attempt rejected — insufficient investigation. "
                f"You have gathered {n} evidence source(s), "
                f"but at least {self._min_evidence_required} are required before applying a fix.\n"
                f"Use investigation actions (read_logs, check_metrics, check_alerts, "
                f"run_diagnostic, check_deployments, trace_request, etc.) first."
            )
            reward = self._compute_reward("no_effect")
            obs = self._build_observation(result)
            if self.step_count >= self.max_steps:
                self.done = True
            return obs, reward, self.done

        # Delegate to scenario-specific handler
        obs, reward, done = self._handle_action(action)
        self.done = done

        # Check max steps
        if self.step_count >= self.max_steps and not self.done:
            self.done = True

        return obs, reward, self.done

    def get_current_state(self) -> dict:
        return {
            "task_id": self.task_id,
            "step_count": self.step_count,
            "incident_phase": self.incident_phase.value,
            "done": self.done,
            "cumulative_reward": self.cumulative_reward,
            "hints_used": self.hints_used,
            "root_cause_identified": self._root_cause_identified,
            "fix_applied": self._fix_applied,
            "resolution_verified": self._resolution_verified,
            "postmortem_written": self._postmortem_written,
            "severity_classified": self._severity_classified,
            "status_page_updated": self._status_page_updated,
        }

    @abstractmethod
    def get_initial_observation(self) -> Observation:
        ...

    @abstractmethod
    def _handle_action(self, action: Action) -> tuple[Observation, float, bool]:
        """Returns: (observation, step_reward, done)"""
        ...

    @abstractmethod
    def _get_alert_summary(self) -> str:
        ...

    @abstractmethod
    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        ...

    @abstractmethod
    def get_grader_score(self) -> float:
        ...
