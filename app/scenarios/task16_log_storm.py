"""
Task 16: Log Storm — Hard difficulty.

Root cause: A deploy to auth-service accidentally set LOG_LEVEL=DEBUG
instead of LOG_LEVEL=INFO. The debug logging generates ~50GB/hour,
saturating the shared logging pipeline. auth-service CPU at 95% from
log serialization. All services lose log shipping.
"""
from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.scenarios.base import BaseScenario
from app.data.log_templates import LogGenerator
from app.data.metrics import MetricsSimulator
from app.data.deploy_history import DeployHistory


class LogStormScenario(BaseScenario):
    task_id = "task16_log_storm"
    max_steps = 35

    def _correct_severity(self) -> str:
        return "SEV2"

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self._observability_loss_noticed = False
        self._auth_cpu_spike_noticed = False
        self._deploy_identified = False
        self._debug_logging_identified = False
        self._correct_fix_applied = False
        self._service_fixed = False
        self._wrong_fix_count = 0
        self._root_cause_service = "auth-service"

    def _get_alert_summary(self) -> str:
        if self._service_fixed:
            return "RESOLVED: auth-service log level restored, logging pipeline recovered."
        return ("WARN: Shared logging pipeline saturated — log ingestion failing for ALL services. "
                "auth-service CPU spiked to 95%. Observability severely degraded. "
                "Started ~12 minutes ago after auth-service deploy.")

    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        statuses = {}
        for svc in self.graph.get_all_services():
            if self._service_fixed:
                statuses[svc] = ServiceStatus(name=svc, status="healthy", error_rate=0.01, latency_p99_ms=50, restarts_last_hour=0)
            elif svc == "auth-service":
                statuses[svc] = ServiceStatus(name=svc, status="degraded", error_rate=0.15, latency_p99_ms=3500, restarts_last_hour=0)
            elif svc == "api-gateway":
                statuses[svc] = ServiceStatus(name=svc, status="degraded", error_rate=0.12, latency_p99_ms=3800, restarts_last_hour=0)
            else:
                statuses[svc] = ServiceStatus(name=svc, status="healthy", error_rate=0.02, latency_p99_ms=60, restarts_last_hour=0)
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
            result += "\n  ⚠️ Logging pipeline: SATURATED — log ingestion dropping for all services"
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "read_logs":
            service = params.get("service", "")
            if service == "auth-service":
                self._auth_cpu_spike_noticed = True
                result = (f"Logs for {service} (WARNING: extremely high volume — showing sample):\n"
                    "[2026-03-26T11:00:01.001Z] [DEBUG] Entering authenticate() for user_id=u-1234 [service=auth-service]\n"
                    "[2026-03-26T11:00:01.002Z] [DEBUG] Token validation: parsing JWT header [service=auth-service]\n"
                    "[2026-03-26T11:00:01.002Z] [DEBUG] Token validation: verifying signature with key ks-01 [service=auth-service]\n"
                    "[2026-03-26T11:00:01.003Z] [DEBUG] Token validation: checking expiry claim [service=auth-service]\n"
                    "[2026-03-26T11:00:01.003Z] [DEBUG] Token validation: verifying audience claim [service=auth-service]\n"
                    "[2026-03-26T11:00:01.004Z] [DEBUG] DB query: SELECT * FROM users WHERE id='u-1234' [service=auth-service]\n"
                    "[2026-03-26T11:00:01.005Z] [DEBUG] DB result: {user_id: 'u-1234', email: '...', roles: [...]} [service=auth-service]\n"
                    "[2026-03-26T11:00:01.005Z] [DEBUG] Permission check: role=admin, resource=/api/orders [service=auth-service]\n"
                    "[2026-03-26T11:00:01.006Z] [INFO]  Auth success for user u-1234 [service=auth-service]\n"
                    "  ... (50,000 log lines per second — 50GB/hour estimated) ⚠️\n"
                    "  LOG_LEVEL=DEBUG ⚠️ (normally INFO)")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            else:
                self._observability_loss_noticed = True
                result = (f"Logs for {service}:\n"
                    "  ⚠️ WARNING: Log retrieval incomplete — logging pipeline is saturated.\n"
                    "  Only 12% of logs from {service} are being ingested.\n"
                    "  Pipeline backlog: 5.2TB\n"
                    "  Cause: auth-service flooding pipeline with ~50GB/hour of debug logs.\n"
                    "  Recent logs that made it through:\n"
                    f"  [2026-03-26T10:58:00Z] [INFO] {service}: Normal operation [service={service}]")
                return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_metrics":
            service = params.get("service", "")
            if service == "auth-service":
                self._auth_cpu_spike_noticed = True
                result = (f"Metrics for {service}:\n"
                    "  error_rate: 0.15\n  latency_p99_ms: 3500\n"
                    "  cpu_percent: 95.2 ⚠️ CRITICAL (normally 25%)\n"
                    "  memory_percent: 78\n"
                    "  log_lines_per_sec: 50000 ⚠️ (normally 200)\n"
                    "  log_volume_gb_per_hour: 50 ⚠️ (normally 0.5)\n"
                    "  log_level: DEBUG ⚠️ (should be INFO)\n"
                    "  log_serialization_cpu_percent: 70 ⚠️")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            else:
                summary = MetricsSimulator.get_metrics_summary(service, self.task_id, self.seed)
                result = f"Metrics for {service}:\n"
                for k, v in summary.items():
                    if k != "service": result += f"  {k}: {v}\n"
                return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_alerts":
            result = ("Active alerts:\n"
                "  [CRIT] Logging pipeline: SATURATED — backlog 5.2TB, ingestion at 12% capacity\n"
                "  [WARN] auth-service: CPU 95% (normally 25%)\n"
                "  [WARN] auth-service: log volume 50GB/hr (normally 0.5GB/hr) ⚠️\n"
                "  [WARN] All services: log shipping degraded — observability impaired\n"
                "  [INFO] auth-service: LOG_LEVEL=DEBUG detected (should be INFO)")
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_deployments":
            service_filter = params.get("service")
            deploys = DeployHistory.get_deploys(self.seed, last_n=5, service_filter=service_filter)
            result = "Recent deployments:\n"
            for d in deploys:
                result += f"  [{d.deploy_id}] {d.service} — {d.description} ({d.minutes_ago}min ago)\n"
            result += ("\n  [deploy-auth-debug] auth-service — 'Add debug logging for auth investigation' (12min ago) ⚠️\n"
                       "    Changed: LOG_LEVEL=INFO → LOG_LEVEL=DEBUG\n"
                       "    Note: Meant for staging, accidentally deployed to production")
            self._deploy_identified = True
            return self._build_observation(result), self._compute_reward("root_cause_progress"), False

        elif at == "check_dependencies":
            service = params.get("service", "")
            upstream = self.graph.get_upstream_of(service)
            downstream = self.graph.get_downstream(service)
            result = f"Dependencies for {service}:\n  Upstream: {upstream}\n  Downstream: {downstream}"
            if service == "auth-service":
                result += "\n\n  Shared resource: logging pipeline (Elasticsearch cluster)"
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "run_diagnostic":
            service = params.get("service", "")
            diag_type = params.get("type", "general")
            if service == "auth-service" and diag_type in ("logging", "logs", "cpu", "general"):
                self._debug_logging_identified = True
                result = (f"Diagnostic (logging) for {service}:\n"
                    "  Log level: DEBUG ⚠️ (expected: INFO)\n"
                    "  Log output rate: 50,000 lines/sec\n"
                    "  Log volume: 50GB/hour\n"
                    "  Log serialization CPU: 70% of auth-service CPU\n\n"
                    "  Logging pipeline impact:\n"
                    "    Pipeline capacity: 10GB/hour\n"
                    "    Current input: 53GB/hour (auth=50GB + others=3GB)\n"
                    "    Pipeline status: SATURATED, dropping 80% of logs\n"
                    "    Backlog: 5.2TB in buffer\n\n"
                    "  Comparison:\n"
                    "    LOG_LEVEL=INFO: ~200 lines/sec, 0.5GB/hr ✓\n"
                    "    LOG_LEVEL=DEBUG: ~50,000 lines/sec, 50GB/hr ✗\n\n"
                    "  💡 Root cause: deploy-auth-debug set LOG_LEVEL=DEBUG for\n"
                    "  auth investigation, was meant for staging but hit production.\n"
                    "  50GB/hr of debug logs saturated the shared pipeline.\n"
                    "  Fix: rollback deploy to restore LOG_LEVEL=INFO or\n"
                    "  set LOG_LEVEL=INFO directly via config change.")
                return self._build_observation(result), self._compute_reward("root_cause_identified"), False
            elif diag_type in ("pipeline", "elasticsearch", "logging_pipeline"):
                self._observability_loss_noticed = True
                result = ("Diagnostic (logging_pipeline):\n"
                    "  Pipeline: Elasticsearch cluster\n"
                    "  Capacity: 10GB/hour\n  Current load: 53GB/hour ⚠️\n"
                    "  Ingestion rate: 12% (dropping 88% of logs)\n"
                    "  Backlog: 5.2TB\n"
                    "  Top contributor: auth-service — 50GB/hr (94% of pipeline load)")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            else:
                return self._build_observation(f"Diagnostic ({diag_type}) for {service}:\n  Normal."), self._compute_reward("info_gathered"), False

        elif at == "apply_fix":
            service = params.get("service", "")
            fix_type = params.get("fix_type", "")
            if service == "auth-service" and fix_type in ("rollback", "rollback_deploy",
                    "set_log_level_info", "restore_log_level", "disable_debug_logging"):
                self._correct_fix_applied = True
                self._fix_applied = True
                self._service_fixed = True
                self.incident_phase = IncidentPhase.VERIFYING
                result = ("Fix applied: auth-service LOG_LEVEL restored to INFO.\n"
                    "  Log volume: 50GB/hr → 0.5GB/hr\n"
                    "  auth-service CPU: 95% → 25%\n"
                    "  Logging pipeline backlog draining.\n"
                    "  All services will regain full log shipping within minutes.\n"
                    "  Verify health to confirm.")
                return self._build_observation(result), self._compute_reward("fix_applied_correctly"), False
            elif fix_type == "restart":
                result = f"Restart {service} — service restarted but loaded same DEBUG config. Log storm continues."
                return self._build_observation(result), self._compute_reward("wrong_fix_applied"), False
            elif fix_type in ("scale_pipeline", "increase_pipeline_capacity"):
                result = "Pipeline scaling initiated — but 50GB/hr far exceeds even scaled capacity. Address the source."
                return self._build_observation(result), self._compute_reward("wrong_fix_applied"), False
            else:
                self._wrong_fix_count += 1
                return self._build_observation(f"Fix {fix_type} on {service} — log storm continues."), self._compute_reward("no_effect"), False

        elif at == "verify_health":
            if self._service_fixed:
                self._resolution_verified = True
                self.incident_phase = IncidentPhase.RESOLVED
                result = ("Health check: ALL SERVICES HEALTHY\n"
                    "  auth-service: cpu=25%, log_level=INFO, log_rate=200/sec\n"
                    "  Logging pipeline: ingestion_rate=100%, backlog=0\n"
                    "  All services: full log shipping restored\n"
                    "Incident resolved.")
                return self._build_observation(result), self._compute_reward("resolution_verified"), True
            else:
                return self._build_observation("Health check: ISSUES PERSIST\n  auth-service CPU 95%, log storm ongoing."), self._compute_reward("no_effect"), False

        elif at == "write_postmortem":
            content = params.get("content", "")
            if len(content) > 50:
                self._postmortem_written = True
                reward = self._compute_reward("postmortem_written")
            else:
                reward = self._compute_reward("no_effect")
            return self._build_observation("Postmortem recorded." if len(content) > 50 else "Too short."), reward, self.incident_phase == IncidentPhase.RESOLVED

        elif at == "escalate":
            self.hints_used += 1
            result = "Hint: auth-service deploy set LOG_LEVEL=DEBUG. 50GB/hr of debug logs saturating pipeline. Rollback the deploy."
            return self._build_observation(result), self._compute_reward("escalate_used"), False

        else:
            return self._build_observation(f"Unknown action type: {at}"), self._compute_reward("no_effect"), False

    def get_grader_score(self) -> float:
        score = 0.0
        score += self._observability_loss_noticed * 0.08
        score += self._auth_cpu_spike_noticed * 0.10
        score += self._deploy_identified * 0.12
        score += self._debug_logging_identified * 0.15
        score += self._correct_fix_applied * 0.22
        score += self._resolution_verified * 0.13
        score += self._postmortem_written * 0.08
        time_bonus = max(0, 1 - (self.step_count / self.max_steps)) * 0.05
        score += time_bonus
        score += self._evidence_breadth_score()
        score += self._postmortem_quality_bonus(["debug", "log", "level", "pipeline", "saturated", "deploy", "volume"])
        score += self._severity_correct * 0.02
        score += (self._status_page_updated and self._status_page_before_fix) * 0.02
        if self._fix_applied and self._fix_before_any_breach:
            score += 0.02
        
        # Efficient investigation bonus
        score += self._efficient_investigation_bonus()
        # Blast radius assessment bonus
        score += self._blast_radius_bonus()
        score -= self._wrong_fix_count * 0.05
        score -= self.hints_used * 0.05
        return round(min(0.999, max(0.001, score)), 4)
