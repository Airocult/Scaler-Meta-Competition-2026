"""
Task 1: Memory Leak OOM Kill — Easy difficulty.

Root cause: order-service is OOMKilling every ~4 minutes.
All other services are healthy.
"""
from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.scenarios.base import BaseScenario
from app.data.log_templates import LogGenerator
from app.data.metrics import MetricsSimulator
from app.data.service_graph import ServiceGraph


class MemoryLeakScenario(BaseScenario):
    task_id = "task1_memory_leak"
    max_steps = 20

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self._correct_service_identified = False
        self._fix_applied_correctly = False
        self._fix_applied_to_wrong_service = False
        self._order_service_fixed = False

    def _get_alert_summary(self) -> str:
        return ("CRITICAL: order-service has restarted 4 times in 20 minutes. "
                "Memory usage spiking before each restart — suspected memory leak.")

    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        statuses = {}
        for svc in self.graph.get_all_services():
            if svc == "order-service" and not self._order_service_fixed:
                statuses[svc] = ServiceStatus(
                    name=svc, status="down", error_rate=0.58,
                    latency_p99_ms=12000, restarts_last_hour=4,
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
                result += f"  {svc}: status={status.status}, error_rate={status.error_rate}, restarts={status.restarts_last_hour}\n"
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "read_logs":
            service = params.get("service", "")
            logs = LogGenerator.generate(self.task_id, service, self.seed)
            result = f"Logs for {service} (last 20 entries):\n" + "\n".join(logs)
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "check_metrics":
            service = params.get("service", "")
            metric = params.get("metric", "")
            if metric and service:
                ts = MetricsSimulator.get_timeseries(service, metric, self.task_id, self.seed)
                summary = MetricsSimulator.get_metrics_summary(service, self.task_id, self.seed)
                result = f"Metrics for {service}:\n"
                result += f"  error_rate: {summary['error_rate']}\n"
                result += f"  latency_p99_ms: {summary['latency_p99_ms']}\n"
                result += f"  cpu_percent: {summary['cpu_percent']}\n"
                result += f"  memory_percent: {summary['memory_percent']}\n"
                result += f"  connection_pool_usage: {summary['connection_pool_usage']}\n"
            else:
                summary = MetricsSimulator.get_metrics_summary(
                    service or "order-service", self.task_id, self.seed
                )
                result = f"Metrics summary: {summary}"
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "check_alerts":
            result = ("Active alerts:\n"
                      "  [CRITICAL] order-service: OOMKilled — container restarted 4x in 20min\n"
                      "  [WARN] order-service: memory usage exceeding 95% threshold")
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "check_deployments":
            from app.data.deploy_history import DeployHistory
            service_filter = params.get("service")
            last_n = params.get("last_n", 5)
            deploys = DeployHistory.get_deploys(self.seed, last_n=last_n, service_filter=service_filter)
            result = "Recent deployments:\n"
            for d in deploys:
                result += f"  [{d.deploy_id}] {d.service} — {d.description} ({d.minutes_ago}min ago by {d.author})\n"
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "check_dependencies":
            service = params.get("service", "")
            upstream = self.graph.get_upstream_of(service)
            downstream = self.graph.get_downstream(service)
            chain = self.graph.get_dependency_chain(service)
            result = (f"Dependencies for {service}:\n"
                      f"  Upstream (callers): {upstream}\n"
                      f"  Downstream (calls): {downstream}\n"
                      f"  Full dependency chain: {chain}")
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "run_diagnostic":
            service = params.get("service", "")
            diag_type = params.get("type", "general")
            if service == "order-service":
                result = (f"Diagnostic ({diag_type}) for {service}:\n"
                          "  memory_usage: 98.2% (CRITICAL)\n"
                          "  heap_size: 490MB / 512MB\n"
                          "  gc_overhead: 97.3%\n"
                          "  open_connections: 12\n"
                          "  thread_count: 245 (HIGH)")
            else:
                result = (f"Diagnostic ({diag_type}) for {service}:\n"
                          "  All metrics within normal range.\n"
                          "  No issues detected.")
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "apply_fix":
            service = params.get("service", "")
            fix_type = params.get("fix_type", "")

            if service == "order-service":
                self._correct_service_identified = True
                self._root_cause_identified = True
                if fix_type == "restart":
                    self._fix_applied_correctly = True
                    self._fix_applied = True
                    self._order_service_fixed = True
                    self.incident_phase = IncidentPhase.VERIFYING
                    result = ("Fix applied: order-service restarted successfully.\n"
                              "Memory usage dropped to 23%. Service is responding.\n"
                              "Please verify health to confirm resolution.")
                    reward = self._compute_reward("fix_applied_correctly")
                    return self._build_observation(result), reward, False
                else:
                    self._fix_applied = True
                    self._order_service_fixed = True
                    self.incident_phase = IncidentPhase.VERIFYING
                    result = (f"Fix applied: {fix_type} on order-service.\n"
                              "Service is responding. Verify health to confirm.")
                    reward = self._compute_reward("fix_applied_correctly")
                    return self._build_observation(result), reward, False
            else:
                self._fix_applied_to_wrong_service = True
                result = (f"Fix applied to {service} — no effect observed.\n"
                          f"{service} was already healthy. Root cause is elsewhere.")
                reward = self._compute_reward("wrong_fix_applied")
                return self._build_observation(result), reward, False

        elif at == "verify_health":
            if self._fix_applied and self._order_service_fixed:
                self._resolution_verified = True
                self.incident_phase = IncidentPhase.RESOLVED
                result = ("Health check: ALL SERVICES HEALTHY\n"
                          "  order-service: status=healthy, error_rate=0.01, restarts=0\n"
                          "Incident resolved. Consider writing a postmortem.")
                reward = self._compute_reward("resolution_verified")
                done = True
                return self._build_observation(result), reward, done
            else:
                result = ("Health check: ISSUES REMAIN\n"
                          "  order-service: status=down, error_rate=0.58, restarts=4\n"
                          "Root cause has not been addressed yet.")
                reward = self._compute_reward("no_effect")
                return self._build_observation(result), reward, False

        elif at == "write_postmortem":
            content = params.get("content", "")
            if len(content) > 50:
                self._postmortem_written = True
                result = "Postmortem recorded successfully."
                reward = self._compute_reward("postmortem_written")
            else:
                result = "Postmortem too short — please provide more detail (min 50 chars)."
                reward = self._compute_reward("no_effect")
            # If we're already resolved, writing postmortem ends the episode
            done = self.incident_phase == IncidentPhase.RESOLVED
            return self._build_observation(result), reward, done

        elif at == "escalate":
            self.hints_used += 1
            result = ("Escalation hint: Check memory metrics on order-service.\n"
                      "The OOM kills suggest a memory leak in that service.\n"
                      "Try: check_metrics(service='order-service', metric='memory')")
            reward = self._compute_reward("escalate_used")
            return self._build_observation(result), reward, False

        else:
            result = f"Unknown action type: {at}"
            reward = self._compute_reward("no_effect")
            return self._build_observation(result), reward, False

    def get_grader_score(self) -> float:
        score = 0.0
        score += self._correct_service_identified * 0.35
        score += self._fix_applied_correctly * 0.25
        score += self._resolution_verified * 0.20
        score += self._postmortem_written * 0.10

        # Time bonus
        time_bonus = max(0, 1 - (self.step_count / self.max_steps)) * 0.10
        score += time_bonus

        # Escalation penalty
        score -= self.hints_used * 0.05

        epsilon = 1e-4
        return round(min(1 - epsilon, max(epsilon, score)), 4)
