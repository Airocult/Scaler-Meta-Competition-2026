"""
Task 3: Distributed Race Condition via Config Change — Hard difficulty.

Root cause: inventory-service deploy 12 minutes ago changed
redis.lock_timeout_ms from 5000ms → 500ms, causing cache read races
under load. 5xx spike appeared 12min after deploy.
"""
from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.scenarios.base import BaseScenario
from app.data.log_templates import LogGenerator
from app.data.metrics import MetricsSimulator
from app.data.deploy_history import DeployHistory


class RaceConditionScenario(BaseScenario):
    task_id = "task3_race_condition"
    max_steps = 40

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self._error_spike_noticed = False
        self._deploy_identified = False
        self._config_diff_examined = False
        self._correct_rollback_applied = False
        self._errors_verified_ceased = False
        self._postmortem_mentions_root_cause = False
        self._restarts_without_deploy_check = 0
        self._inventory_fixed = False

    def _get_alert_summary(self) -> str:
        return ("CRITICAL: Intermittent 5xx errors across multiple services. "
                "Error rate spiked ~12 minutes ago. Affecting inventory-service, "
                "order-service, and api-gateway.")

    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        statuses = {}
        for svc in self.graph.get_all_services():
            if self._inventory_fixed:
                statuses[svc] = ServiceStatus(
                    name=svc, status="healthy", error_rate=0.01,
                    latency_p99_ms=45, restarts_last_hour=0,
                )
            elif svc == "inventory-service":
                statuses[svc] = ServiceStatus(
                    name=svc, status="degraded", error_rate=0.38,
                    latency_p99_ms=5500, restarts_last_hour=1,
                )
            elif svc == "order-service":
                statuses[svc] = ServiceStatus(
                    name=svc, status="degraded", error_rate=0.22,
                    latency_p99_ms=3200, restarts_last_hour=0,
                )
            elif svc == "api-gateway":
                statuses[svc] = ServiceStatus(
                    name=svc, status="degraded", error_rate=0.18,
                    latency_p99_ms=2800, restarts_last_hour=0,
                )
            elif svc == "inventory-db":
                statuses[svc] = ServiceStatus(
                    name=svc, status="degraded", error_rate=0.12,
                    latency_p99_ms=1200, restarts_last_hour=0,
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
            logs = LogGenerator.generate(self.task_id, service, self.seed)
            result = f"Logs for {service} (last 20 entries):\n" + "\n".join(logs)
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "check_metrics":
            service = params.get("service", "")
            metric = params.get("metric", "")

            if service == "inventory-service" and metric == "error_rate":
                self._error_spike_noticed = True
                ts = MetricsSimulator.get_timeseries(service, "error_rate", self.task_id, self.seed)
                result = f"Error rate timeseries for {service} (last 60 min):\n"
                for point in ts[-15:]:  # Show last 15 data points
                    result += f"  {point['timestamp']}: {point['value']}\n"
                result += ("\n📈 Pattern: error rate was normal (<0.01) until 12 minutes ago, "
                           "then spiked to >0.35. Something changed ~12 min ago.")
                reward = self._compute_reward("root_cause_progress")
                return self._build_observation(result), reward, False

            summary = MetricsSimulator.get_metrics_summary(service, self.task_id, self.seed)
            result = f"Metrics for {service}:\n"
            result += f"  error_rate: {summary['error_rate']}\n"
            result += f"  latency_p99_ms: {summary['latency_p99_ms']}\n"
            result += f"  cpu_percent: {summary['cpu_percent']}\n"
            result += f"  memory_percent: {summary['memory_percent']}\n"
            result += f"  connection_pool_usage: {summary['connection_pool_usage']}\n"
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "check_alerts":
            result = ("Active alerts:\n"
                      "  [CRITICAL] inventory-service: error rate 38% — intermittent 500s\n"
                      "  [WARN] order-service: degraded — upstream inventory-service failing\n"
                      "  [WARN] api-gateway: elevated error rate 18%\n"
                      "  [INFO] Error spike began approximately 12 minutes ago")
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "check_deployments":
            service_filter = params.get("service")
            last_n = params.get("last_n", 5)
            deploys = DeployHistory.get_deploys(self.seed, last_n=last_n, service_filter=service_filter)
            self._deploy_identified = True
            result = "Recent deployments:\n"
            for d in deploys:
                marker = " ⚠️ RECENT" if d.minutes_ago <= 15 else ""
                result += (f"  [{d.deploy_id}] {d.service} — {d.description} "
                           f"({d.minutes_ago}min ago by {d.author}){marker}\n")
                if d.config_diff:
                    result += f"    Config changes detected: {list(d.config_diff.keys())}\n"
            reward = self._compute_reward("root_cause_progress")
            return self._build_observation(result), reward, False

        elif at == "check_dependencies":
            service = params.get("service", "")
            upstream = self.graph.get_upstream_of(service)
            downstream = self.graph.get_downstream(service)
            result = (f"Dependencies for {service}:\n"
                      f"  Upstream (callers): {upstream}\n"
                      f"  Downstream (calls): {downstream}")
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "run_diagnostic":
            service = params.get("service", "")
            diag_type = params.get("type", "general")
            deploy_id = params.get("deploy_id", "")

            if (service == "inventory-service" and diag_type == "config_diff"
                    and deploy_id == "deploy-a1b2c3"):
                self._config_diff_examined = True
                result = (f"Config diff for deploy {deploy_id} on {service}:\n"
                          "  redis.lock_timeout_ms: 5000 → 500  ⚠️ REDUCED BY 10x\n"
                          "  redis.connection_pool_size: 10 → 50\n"
                          "\n⚠️  The lock_timeout_ms reduction from 5000ms to 500ms is suspicious.\n"
                          "With high concurrency, 500ms may not be enough for lock acquisition,\n"
                          "causing stale reads and race conditions on inventory data.")
                reward = self._compute_reward("root_cause_identified")
                return self._build_observation(result), reward, False

            elif service == "inventory-service":
                result = (f"Diagnostic ({diag_type}) for {service}:\n"
                          "  redis_lock_timeouts: 847 in last 12min (CRITICAL)\n"
                          "  stale_read_count: 234\n"
                          "  cache_consistency_errors: 156\n"
                          "  concurrent_writers: avg 8.3 per key\n"
                          "  Recommendation: Check recent config changes to Redis settings")
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False

            else:
                result = (f"Diagnostic ({diag_type}) for {service}:\n"
                          "  All metrics within normal range.")
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False

        elif at == "apply_fix":
            service = params.get("service", "")
            fix_type = params.get("fix_type", "")
            deploy_id = params.get("deploy_id", "")

            if (service == "inventory-service"
                    and fix_type in ("rollback_config", "rollback")
                    and deploy_id == "deploy-a1b2c3"):
                self._correct_rollback_applied = True
                self._fix_applied = True
                self._root_cause_identified = True
                self._inventory_fixed = True
                self.incident_phase = IncidentPhase.VERIFYING
                result = ("Rollback applied: inventory-service config reverted.\n"
                          "  redis.lock_timeout_ms: 500 → 5000 (restored)\n"
                          "  redis.connection_pool_size: 50 → 10 (restored)\n"
                          "Error rate dropping. Lock timeouts ceased.\n"
                          "Verify health to confirm resolution.")
                reward = self._compute_reward("fix_applied_correctly")
                return self._build_observation(result), reward, False

            elif fix_type == "restart":
                if not self._deploy_identified:
                    self._restarts_without_deploy_check += 1
                result = (f"Restart applied to {service}. Service restarted.\n"
                          "However, errors are persisting — the restart did not resolve "
                          "the underlying issue. The race condition reoccurs immediately "
                          "after restart because the same config is loaded.")
                reward = self._compute_reward("wrong_fix_applied")
                return self._build_observation(result), reward, False

            else:
                result = f"Fix {fix_type} applied to {service} — no observable improvement."
                reward = self._compute_reward("no_effect")
                return self._build_observation(result), reward, False

        elif at == "verify_health":
            if self._inventory_fixed:
                self._errors_verified_ceased = True
                self._resolution_verified = True
                self.incident_phase = IncidentPhase.RESOLVED
                result = ("Health check: ALL SERVICES HEALTHY\n"
                          "  inventory-service: error_rate=0.01, lock_timeouts=0\n"
                          "  order-service: error_rate=0.01, healthy\n"
                          "  api-gateway: error_rate=0.01, healthy\n"
                          "Incident resolved. Write a postmortem mentioning the root cause.")
                reward = self._compute_reward("resolution_verified")
                done = True
                return self._build_observation(result), reward, done
            else:
                result = ("Health check: ISSUES PERSIST\n"
                          "  inventory-service: error_rate=0.38, lock_timeouts=847\n"
                          "  Intermittent 500s continuing.\n"
                          "Root cause has not been addressed.")
                reward = self._compute_reward("no_effect")
                return self._build_observation(result), reward, False

        elif at == "write_postmortem":
            content = params.get("content", "")
            content_lower = content.lower() if content else ""
            if len(content) > 50:
                self._postmortem_written = True
                if "lock_timeout" in content_lower or "race condition" in content_lower:
                    self._postmortem_mentions_root_cause = True
                    result = "Postmortem recorded. Root cause correctly identified in writeup."
                    reward = self._compute_reward("postmortem_quality")
                else:
                    result = "Postmortem recorded. Consider mentioning the specific root cause."
                    reward = self._compute_reward("postmortem_written")
            else:
                result = "Postmortem too short — please provide more detail."
                reward = self._compute_reward("no_effect")
            done = self.incident_phase == IncidentPhase.RESOLVED
            return self._build_observation(result), reward, done

        elif at == "escalate":
            self.hints_used += 1
            result = ("Escalation hint: The error spike started ~12 minutes ago.\n"
                      "Check recent deployments — something was deployed around that time.\n"
                      "Look for config changes that might affect concurrency or locking.")
            reward = self._compute_reward("escalate_used")
            return self._build_observation(result), reward, False

        else:
            result = f"Unknown action type: {at}"
            reward = self._compute_reward("no_effect")
            return self._build_observation(result), reward, False

    def get_grader_score(self) -> float:
        score = 0.0
        score += self._error_spike_noticed * 0.10
        score += self._deploy_identified * 0.15
        score += self._config_diff_examined * 0.20
        score += self._correct_rollback_applied * 0.25
        score += self._errors_verified_ceased * 0.15
        score += self._postmortem_mentions_root_cause * 0.15

        # Escalation penalty (harder on hard task)
        score -= self.hints_used * 0.075

        # Dead-end trap: restarts without deploy check
        if self._restarts_without_deploy_check > 2:
            score -= 0.10

        return round(min(0.999, max(0.001, score)), 4)
