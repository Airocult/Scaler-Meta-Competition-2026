"""
Task 9: Disk Space Exhaustion — Medium difficulty.

Root cause: user-db WAL (Write-Ahead Log) files accumulated after
an automated log rotation cron was disabled 2 days ago. Disk hit 100%,
user-db can't write, auth-service login queries fail.
"""
from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.scenarios.base import BaseScenario
from app.data.log_templates import LogGenerator
from app.data.metrics import MetricsSimulator
from app.data.deploy_history import DeployHistory


class DiskFullScenario(BaseScenario):
    task_id = "task9_disk_full"
    max_steps = 25

    def _correct_severity(self) -> str:
        return "SEV1"

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self._auth_failure_noticed = False
        self._user_db_investigated = False
        self._disk_full_identified = False
        self._correct_fix_applied = False
        self._service_fixed = False
        self._wrong_fix_count = 0

    def _get_alert_summary(self) -> str:
        if self._service_fixed:
            return "RESOLVED: user-db disk space recovered, auth-service fully operational."
        return ("CRIT: auth-service login failures spiking. 80% of logins return 500. "
                "user-db health check failing. Started ~5 minutes ago.")

    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        statuses = {}
        for svc in self.graph.get_all_services():
            if self._service_fixed:
                statuses[svc] = ServiceStatus(name=svc, status="healthy", error_rate=0.01, latency_p99_ms=40, restarts_last_hour=0)
            elif svc == "user-db":
                statuses[svc] = ServiceStatus(name=svc, status="down", error_rate=0.80, latency_p99_ms=12000, restarts_last_hour=2)
            elif svc == "auth-service":
                statuses[svc] = ServiceStatus(name=svc, status="down", error_rate=0.80, latency_p99_ms=11000, restarts_last_hour=0)
            elif svc == "api-gateway":
                statuses[svc] = ServiceStatus(name=svc, status="degraded", error_rate=0.65, latency_p99_ms=11500, restarts_last_hour=0)
            else:
                statuses[svc] = ServiceStatus(name=svc, status="healthy", error_rate=0.01, latency_p99_ms=40, restarts_last_hour=0)
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
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "read_logs":
            service = params.get("service", "")
            if service == "user-db":
                self._user_db_investigated = True
                result = (f"Logs for {service}:\n"
                    "[2026-03-26T04:00:05Z] [ERROR] PANIC: could not write to WAL file: No space left on device [service=user-db]\n"
                    "[2026-03-26T04:00:08Z] [ERROR] FATAL: data directory /var/lib/postgresql/data has no free space [service=user-db]\n"
                    "[2026-03-26T04:00:10Z] [ERROR] checkpointer process failed: No space left on device [service=user-db]\n"
                    "[2026-03-26T04:00:12Z] [WARN]  Disk usage: /var/lib/postgresql/data — 100% used (499.9G/500G) [service=user-db]\n"
                    "[2026-03-26T04:00:15Z] [WARN]  WAL directory size: 285G — pg_wal/ has 11,423 unarchived segments [service=user-db]\n"
                    "[2026-03-26T04:00:18Z] [ERROR] All write operations rejected — read-only mode forced [service=user-db]")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            elif service == "auth-service":
                self._auth_failure_noticed = True
                result = (f"Logs for {service}:\n"
                    "[2026-03-26T04:00:07Z] [ERROR] Login failed for user alice@example.com: database write error [service=auth-service]\n"
                    "[2026-03-26T04:00:09Z] [ERROR] Failed to update last_login — user-db returned 'disk full' error [service=auth-service]\n"
                    "[2026-03-26T04:00:12Z] [ERROR] 847/1000 login attempts failed in last minute [service=auth-service]\n"
                    "[2026-03-26T04:00:15Z] [WARN]  Session token creation failing — cannot write to sessions table [service=auth-service]")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            else:
                logs = LogGenerator.generate(self.task_id, service, self.seed, step_count=self.step_count)
                return self._build_observation(f"Logs for {service}:\n" + "\n".join(logs[:10])), self._compute_reward("info_gathered"), False

        elif at == "check_metrics":
            service = params.get("service", "")
            if service == "user-db":
                self._user_db_investigated = True
                result = (f"Metrics for {service}:\n"
                    "  error_rate: 0.80\n  latency_p99_ms: 12000\n"
                    "  disk_usage_percent: 100.0 ⚠️ CRITICAL\n"
                    "  disk_total_gb: 500\n  disk_free_gb: 0.1\n"
                    "  wal_directory_gb: 285 ⚠️\n  wal_segments_unarchived: 11423 ⚠️\n"
                    "  cpu_percent: 15\n  memory_percent: 62\n"
                    "  connections_active: 45\n  writes_rejected: true ⚠️")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            elif service == "auth-service":
                self._auth_failure_noticed = True
                result = (f"Metrics for {service}:\n"
                    "  error_rate: 0.80\n  latency_p99_ms: 11000\n"
                    "  login_success_rate: 0.20 ⚠️\n"
                    "  cpu_percent: 30\n  memory_percent: 45")
                return self._build_observation(result), self._compute_reward("info_gathered"), False
            else:
                summary = MetricsSimulator.get_metrics_summary(service, self.task_id, self.seed)
                result = f"Metrics for {service}:\n"
                for k, v in summary.items():
                    if k != "service": result += f"  {k}: {v}\n"
                return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_alerts":
            result = ("Active alerts:\n"
                "  [CRIT] user-db: Disk usage 100% — write operations failing\n"
                "  [CRIT] auth-service: Login error rate 80%\n"
                "  [WARN] api-gateway: Elevated 500 error rate on /auth endpoints\n"
                "  [INFO] user-db: WAL directory 285GB — log rotation may be broken")
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_deployments":
            service_filter = params.get("service")
            deploys = DeployHistory.get_deploys(self.seed, last_n=5, service_filter=service_filter)
            result = "Recent deployments:\n"
            for d in deploys:
                result += f"  [{d.deploy_id}] {d.service} — {d.description} ({d.minutes_ago}min ago)\n"
            result += ("\n  [ops-cron-disable] user-db — 'Disabled pg_archivecleanup cron for maintenance' (2 days ago)\n"
                       "    Note: Cron was supposed to be re-enabled after 1-hour maintenance window.")
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_dependencies":
            service = params.get("service", "")
            upstream = self.graph.get_upstream_of(service)
            downstream = self.graph.get_downstream(service)
            return self._build_observation(f"Dependencies for {service}:\n  Upstream: {upstream}\n  Downstream: {downstream}"), self._compute_reward("info_gathered"), False

        elif at == "run_diagnostic":
            service = params.get("service", "")
            diag_type = params.get("type", "general")
            if service == "user-db" and diag_type in ("disk", "storage", "wal", "general"):
                self._disk_full_identified = True
                result = (f"Diagnostic (disk/WAL) for {service}:\n"
                    "  filesystem: /var/lib/postgresql/data\n"
                    "  total_space: 500G\n  used: 499.9G (100%)\n  free: 0.1G\n\n"
                    "  Top space consumers:\n"
                    "    /var/lib/postgresql/data/pg_wal/ — 285G (11,423 WAL segments) ⚠️\n"
                    "    /var/lib/postgresql/data/base/ — 180G (normal)\n"
                    "    /var/lib/postgresql/data/pg_xlog/ — 30G\n\n"
                    "  WAL archival status:\n"
                    "    archive_mode: on\n"
                    "    archive_command: /usr/bin/pg_archivecleanup (NOT RUNNING) ⚠️\n"
                    "    last_archive_success: 2 days ago ⚠️\n"
                    "    WAL retention cron: DISABLED ⚠️\n\n"
                    "  💡 Root cause: pg_archivecleanup cron disabled 2 days ago.\n"
                    "  WAL segments accumulated to 285GB, filling disk.\n"
                    "  Fix: clean old WAL segments + re-enable archival cron.")
                return self._build_observation(result), self._compute_reward("root_cause_identified"), False
            else:
                return self._build_observation(f"Diagnostic ({diag_type}) for {service}:\n  Normal."), self._compute_reward("info_gathered"), False

        elif at == "apply_fix":
            service = params.get("service", "")
            fix_type = params.get("fix_type", "")
            if service == "user-db" and fix_type in ("clean_wal", "truncate_wal",
                    "enable_archival", "enable_rotation", "clean_wal_enable_cron"):
                self._correct_fix_applied = True
                self._fix_applied = True
                self._root_cause_identified = True
                self._service_fixed = True
                self.incident_phase = IncidentPhase.VERIFYING
                result = ("Fix applied:\n"
                    "  Cleaned 10,500 old WAL segments (freed 250GB)\n"
                    "  Re-enabled pg_archivecleanup cron job\n"
                    "  Disk usage: 50% (249G/500G)\n"
                    "  user-db write operations resumed\n"
                    "  Verify health to confirm auth-service recovery.")
                return self._build_observation(result), self._compute_reward("fix_applied_correctly"), False
            elif fix_type == "restart":
                result = f"Restart applied to {service}. Database restarted but disk still full — writes fail again."
                return self._build_observation(result), self._compute_reward("wrong_fix_applied"), False
            else:
                self._wrong_fix_count += 1
                return self._build_observation(f"Fix {fix_type} on {service} — no effect."), self._compute_reward("no_effect"), False

        elif at == "verify_health":
            if self._service_fixed:
                self._resolution_verified = True
                self.incident_phase = IncidentPhase.RESOLVED
                result = ("Health check: ALL SERVICES HEALTHY\n"
                    "  user-db: disk_usage=50%, writes=OK, latency_p99=30ms\n"
                    "  auth-service: login_success_rate=99.5%, error_rate=0.005\n"
                    "Incident resolved.")
                return self._build_observation(result), self._compute_reward("resolution_verified"), True
            else:
                return self._build_observation("Health check: ISSUES PERSIST\n  user-db: disk still at 100%."), self._compute_reward("no_effect"), False

        elif at == "write_postmortem":
            content = params.get("content", "")
            if len(content) > 50:
                self._postmortem_written = True
                reward = self._compute_reward("postmortem_written")
            else:
                reward = self._compute_reward("no_effect")
            return self._build_observation("Postmortem recorded." if len(content) > 50 else "Postmortem too short."), reward, self.incident_phase == IncidentPhase.RESOLVED

        elif at == "escalate":
            self.hints_used += 1
            result = "Escalation hint: user-db disk is full. Check WAL log retention — archival cron may be disabled."
            return self._build_observation(result), self._compute_reward("escalate_used"), False

        else:
            return self._build_observation(f"Unknown action type: {at}"), self._compute_reward("no_effect"), False

    def get_grader_score(self) -> float:
        score = 0.0
        score += self._auth_failure_noticed * 0.10
        score += self._user_db_investigated * 0.15
        score += self._disk_full_identified * 0.18
        score += self._correct_fix_applied * 0.25
        score += self._resolution_verified * 0.13
        score += self._postmortem_written * 0.08
        time_bonus = max(0, 1 - (self.step_count / self.max_steps)) * 0.05
        score += time_bonus
        score += self._evidence_breadth_score()
        score += self._postmortem_quality_bonus(["disk", "wal", "full", "archival", "rotation", "cron"])
        score += self._severity_correct * 0.02
        score += (self._status_page_updated and self._status_page_before_fix) * 0.02
        if self._fix_applied and self._fix_before_any_breach:
            score += 0.02
        score -= self._wrong_fix_count * 0.05
        score -= self.hints_used * 0.05
        return round(min(0.999, max(0.001, score)), 4)
