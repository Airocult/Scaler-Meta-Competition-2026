"""
Task 2: Cascading DB Pool Exhaustion — Medium difficulty.

Root cause: payment-db connection pool exhausted (pool_usage=0.99)
  → payment-service failing (error_rate=0.87)
  → order-service failing (error_rate=0.62)
  → api-gateway returning 503 (visible to users)
"""
from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.scenarios.base import BaseScenario
from app.data.log_templates import LogGenerator
from app.data.metrics import MetricsSimulator
from app.data.service_graph import ServiceGraph


class DBCascadeScenario(BaseScenario):
    task_id = "task2_db_cascade"
    max_steps = 30

    def _correct_severity(self) -> str:
        return "SEV1"  # payment path completely down — revenue impact

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self._traced_to_payment_service = False
        self._traced_to_payment_db = False
        self._correct_fix_applied = False
        self._fix_applied_to_wrong_surface = False
        self._payment_db_fixed = False
        self._restart_count = 0

    def _get_alert_summary(self) -> str:
        return ("CRITICAL: api-gateway error rate 58% — customers seeing 503s. "
                "Multiple services degraded in cascade pattern. Trace upstream dependencies.")

    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        statuses = {}
        for svc in self.graph.get_all_services():
            if self._payment_db_fixed:
                statuses[svc] = ServiceStatus(
                    name=svc, status="healthy", error_rate=0.01,
                    latency_p99_ms=45, restarts_last_hour=0,
                )
            elif svc == "payment-db":
                statuses[svc] = ServiceStatus(
                    name=svc, status="degraded", error_rate=0.92,
                    latency_p99_ms=30000, restarts_last_hour=0,
                )
            elif svc == "payment-service":
                statuses[svc] = ServiceStatus(
                    name=svc, status="down", error_rate=0.87,
                    latency_p99_ms=25000, restarts_last_hour=2,
                )
            elif svc == "order-service":
                statuses[svc] = ServiceStatus(
                    name=svc, status="degraded", error_rate=0.62,
                    latency_p99_ms=18000, restarts_last_hour=0,
                )
            elif svc == "api-gateway":
                statuses[svc] = ServiceStatus(
                    name=svc, status="degraded", error_rate=0.58,
                    latency_p99_ms=15000, restarts_last_hour=0,
                )
            else:
                statuses[svc] = ServiceStatus(
                    name=svc, status="healthy", error_rate=0.01,
                    latency_p99_ms=45, restarts_last_hour=0,
                )
        return statuses

    def get_initial_observation(self) -> Observation:
        return self._build_observation("Incident alert triggered. Begin investigation.")

    def _handle_action(self, action: Action) -> tuple[Observation, float, bool]:
        at = action.action_type
        params = action.parameters

        if at == "list_services":
            result = "Services:\n"
            for svc, status in self._get_service_statuses().items():
                result += f"  {svc}: status={status.status}, error_rate={status.error_rate}, latency_p99={status.latency_p99_ms}ms\n"
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "read_logs":
            service = params.get("service", "")
            logs = LogGenerator.generate(self.task_id, service, self.seed, step_count=self.step_count)
            result = f"Logs for {service} (last 20 entries):\n" + "\n".join(logs)
            if service == "payment-service":
                self._traced_to_payment_service = True
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "check_metrics":
            service = params.get("service", "")
            summary = MetricsSimulator.get_metrics_summary(service, self.task_id, self.seed)
            result = f"Metrics for {service}:\n"
            result += f"  error_rate: {summary['error_rate']}\n"
            result += f"  latency_p99_ms: {summary['latency_p99_ms']}\n"
            result += f"  cpu_percent: {summary['cpu_percent']}\n"
            result += f"  memory_percent: {summary['memory_percent']}\n"
            result += f"  connection_pool_usage: {summary['connection_pool_usage']}\n"
            if service == "payment-db":
                self._traced_to_payment_db = True
                result += "\n  ⚠️  connection_pool_usage is CRITICAL (>0.95)"
            if service == "payment-service":
                self._traced_to_payment_service = True
            reward = self._compute_reward("info_gathered")
            if service == "payment-db":
                reward = self._compute_reward("root_cause_progress")
            return self._build_observation(result), reward, False

        elif at == "check_alerts":
            result = ("Active alerts:\n"
                      "  [CRITICAL] api-gateway: error rate 58% — 503 responses to users\n"
                      "  [CRITICAL] payment-service: DOWN — connection timeouts\n"
                      "  [WARN] order-service: degraded — upstream failures from payment-service")
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "check_deployments":
            from app.data.deploy_history import DeployHistory
            service_filter = params.get("service")
            last_n = params.get("last_n", 5)
            deploys = DeployHistory.get_deploys(self.seed, last_n=last_n, service_filter=service_filter)
            result = "Recent deployments:\n"
            for d in deploys:
                result += f"  [{d.deploy_id}] {d.service} — {d.description} ({d.minutes_ago}min ago)\n"
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "check_dependencies":
            service = params.get("service", "")
            upstream = self.graph.get_upstream_of(service)
            downstream = self.graph.get_downstream(service)
            result = (f"Dependencies for {service}:\n"
                      f"  Upstream (callers): {upstream}\n"
                      f"  Downstream (calls): {downstream}")
            # Track if agent traces the dependency chain
            if service == "payment-service":
                self._traced_to_payment_service = True
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "run_diagnostic":
            service = params.get("service", "")
            diag_type = params.get("type", "general")
            if service == "payment-db" and diag_type in ("connection_count", "connection_pool", "general"):
                self._traced_to_payment_db = True
                result = (f"Diagnostic ({diag_type}) for {service}:\n"
                          "  active_connections: 198 / 200 (CRITICAL)\n"
                          "  waiting_queries: 342\n"
                          "  avg_query_time: 4500ms\n"
                          "  pool_exhaustion: True\n"
                          "  oldest_connection_age: 45min")
                reward = self._compute_reward("root_cause_progress")
            elif service == "payment-service":
                self._traced_to_payment_service = True
                result = (f"Diagnostic ({diag_type}) for {service}:\n"
                          "  upstream_errors: payment-db connection timeouts\n"
                          "  retry_queue_depth: 1200\n"
                          "  circuit_breaker: OPEN")
                reward = self._compute_reward("info_gathered")
            else:
                result = (f"Diagnostic ({diag_type}) for {service}:\n"
                          "  All metrics within normal range.")
                reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "apply_fix":
            service = params.get("service", "")
            fix_type = params.get("fix_type", "")

            if service == "payment-db" and fix_type in ("increase_pool_size", "scale_pool", "drain_pool"):
                self._correct_fix_applied = True
                self._fix_applied = True
                self._payment_db_fixed = True
                self.incident_phase = IncidentPhase.VERIFYING
                result = ("Fix applied: payment-db connection pool increased from 200 → 500.\n"
                          "Active connections dropping. Payment-service recovering.\n"
                          "Cascade clearing: order-service and api-gateway error rates declining.\n"
                          "Verify health to confirm full resolution.")
                reward = self._compute_reward("fix_applied_correctly")
                return self._build_observation(result), reward, False

            elif service == "payment-service" and fix_type == "restart":
                self._restart_count += 1
                result = ("payment-service restarted. Briefly recovered but pool exhausted again.\n"
                          "Error rate climbing back. Root cause is upstream in payment-db.")
                reward = self._compute_reward("no_effect")
                return self._build_observation(result), reward, False

            elif service in ("api-gateway", "order-service"):
                self._fix_applied_to_wrong_surface = True
                result = (f"{service} is not the root cause — it's a symptom of upstream failures.\n"
                          "Root cause is further down the dependency chain.")
                reward = self._compute_reward("wrong_fix_applied")
                return self._build_observation(result), reward, False

            else:
                result = f"Fix applied to {service} ({fix_type}) — no observable effect."
                reward = self._compute_reward("no_effect")
                return self._build_observation(result), reward, False

        elif at == "verify_health":
            if self._payment_db_fixed:
                self._resolution_verified = True
                self.incident_phase = IncidentPhase.RESOLVED
                result = ("Health check: ALL SERVICES HEALTHY\n"
                          "  payment-db: pool_usage=0.35, connections normalized\n"
                          "  payment-service: error_rate=0.01, healthy\n"
                          "  order-service: error_rate=0.01, healthy\n"
                          "  api-gateway: error_rate=0.01, healthy\n"
                          "Incident resolved. Consider writing a postmortem.")
                reward = self._compute_reward("resolution_verified")
                done = True
                return self._build_observation(result), reward, done
            else:
                statuses = self._get_service_statuses()
                result = "Health check: ISSUES PERSIST\n"
                for svc, s in statuses.items():
                    if s.status != "healthy":
                        result += f"  {svc}: status={s.status}, error_rate={s.error_rate}\n"
                result += "Root cause has not been addressed."
                reward = self._compute_reward("no_effect")
                return self._build_observation(result), reward, False

        elif at == "write_postmortem":
            content = params.get("content", "")
            if len(content) > 50:
                self._postmortem_written = True
                result = "Postmortem recorded successfully."
                reward = self._compute_reward("postmortem_written")
            else:
                result = "Postmortem too short — please provide more detail."
                reward = self._compute_reward("no_effect")
            done = self.incident_phase == IncidentPhase.RESOLVED
            return self._build_observation(result), reward, done

        elif at == "escalate":
            self.hints_used += 1
            result = ("Escalation hint: The api-gateway errors are a symptom.\n"
                      "Trace the dependency chain: api-gateway → order-service → payment-service → payment-db.\n"
                      "Check connection pool metrics on payment-db.")
            reward = self._compute_reward("escalate_used")
            return self._build_observation(result), reward, False

        else:
            result = f"Unknown action type: {at}"
            reward = self._compute_reward("no_effect")
            return self._build_observation(result), reward, False

    def get_grader_score(self) -> float:
        if self._fix_applied_to_wrong_surface and not self._correct_fix_applied:
            return min(0.35, self._compute_partial_score())

        return self._compute_partial_score()

    def _compute_partial_score(self) -> float:
        score = 0.0
        score += self._traced_to_payment_service * 0.12
        score += self._traced_to_payment_db * 0.22
        score += self._correct_fix_applied * 0.28
        score += self._resolution_verified * 0.13
        score += self._postmortem_written * 0.08

        # Time bonus (smaller weight)
        time_bonus = max(0, 1 - (self.step_count / self.max_steps)) * 0.05
        score += time_bonus

        # Evidence breadth bonus
        score += self._evidence_breadth_score()

        # Postmortem quality
        score += self._postmortem_quality_bonus(
            ["payment-db", "pool", "connection", "cascade", "hikari"]
        )

        # Incident communication bonuses
        score += self._severity_correct * 0.02
        score += (self._status_page_updated and self._status_page_before_fix) * 0.02

        # SLO-aware bonus
        if self._fix_applied and self._fix_before_any_breach:
            score += 0.02

        # Escalation penalty
        score -= self.hints_used * 0.05

        return round(min(0.999, max(0.001, score)), 4)
