"""
Task 11: Database Migration Lock — Hard difficulty.

Root cause: A DBA ran ALTER TABLE payment_transactions ADD COLUMN on
payment-db during peak hours without setting lock_timeout. The DDL
acquired an exclusive lock blocking all writes for 12+ minutes.
"""
from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.scenarios.base import BaseScenario
from app.data.log_templates import LogGenerator
from app.data.metrics import MetricsSimulator
from app.data.deploy_history import DeployHistory


class DbMigrationLockScenario(BaseScenario):
    task_id = "task11_db_migration_lock"
    max_steps = 35

    def _correct_severity(self) -> str:
        return "SEV1"

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self._payment_failure_noticed = False
        self._payment_db_investigated = False
        self._lock_detected = False
        self._migration_identified = False
        self._correct_fix_applied = False
        self._service_fixed = False
        self._wrong_fix_count = 0
        self._root_cause_service = "payment-db"

    def _get_alert_summary(self) -> str:
        if self._service_fixed:
            return "RESOLVED: payment-db lock released, payment processing restored."
        return ("CRIT: payment-service writes timing out. All payment transactions failing. "
                "payment-db exclusive lock detected. Started ~12 minutes ago.")

    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        statuses = {}
        for svc in self.graph.get_all_services():
            if self._service_fixed:
                statuses[svc] = ServiceStatus(name=svc, status="healthy", error_rate=0.01, latency_p99_ms=45, restarts_last_hour=0)
            elif svc == "payment-db":
                statuses[svc] = ServiceStatus(name=svc, status="down", error_rate=0.95, latency_p99_ms=30000, restarts_last_hour=0)
            elif svc == "payment-service":
                statuses[svc] = ServiceStatus(name=svc, status="down", error_rate=0.92, latency_p99_ms=30500, restarts_last_hour=0)
            elif svc == "order-service":
                statuses[svc] = ServiceStatus(name=svc, status="degraded", error_rate=0.35, latency_p99_ms=31000, restarts_last_hour=0)
            else:
                statuses[svc] = ServiceStatus(name=svc, status="healthy", error_rate=0.01, latency_p99_ms=45, restarts_last_hour=0)
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
            if service == "payment-db":
                self._payment_db_investigated = True
                result = (f"Logs for {service}:\n"
                    "[2026-03-26T06:00:05Z] [INFO]  ALTER TABLE payment_transactions ADD COLUMN refund_id UUID started (pid 8234) [service=payment-db]\n"
                    "[2026-03-26T06:00:05Z] [WARN]  Exclusive lock acquired on table 'payment_transactions' by pid 8234 [service=payment-db]\n"
                    "[2026-03-26T06:00:30Z] [WARN]  147 queries waiting for lock on payment_transactions [service=payment-db]\n"
                    "[2026-03-26T06:02:00Z] [ERROR] Lock wait timeout for INSERT INTO payment_transactions — waited 120s [service=payment-db]\n"
                    "[2026-03-26T06:05:00Z] [ERROR] 892 blocked statements waiting on AccessExclusiveLock [service=payment-db]\n"
                    "[2026-03-26T06:10:00Z] [WARN]  ALTER TABLE still running — rewriting table (65% complete, ETA 8 min) [service=payment-db]\n"
                    "[2026-03-26T06:12:00Z] [ERROR] Connection pool exhausted: 300/300 connections, all blocked [service=payment-db]")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            elif service == "payment-service":
                self._payment_failure_noticed = True
                result = (f"Logs for {service}:\n"
                    "[2026-03-26T06:00:35Z] [ERROR] INSERT INTO payment_transactions failed: lock timeout [service=payment-service]\n"
                    "[2026-03-26T06:01:00Z] [ERROR] Payment transaction 00123 failed after 30s wait [service=payment-service]\n"
                    "[2026-03-26T06:05:00Z] [ERROR] 892 payment failures in last 5 minutes ⚠️ [service=payment-service]\n"
                    "[2026-03-26T06:10:00Z] [WARN]  Retry queue depth: 1500 — all retries also timing out [service=payment-service]")
                return self._build_observation(result), self._compute_reward("info_gathered"), False
            else:
                logs = LogGenerator.generate(self.task_id, service, self.seed, step_count=self.step_count)
                return self._build_observation(f"Logs for {service}:\n" + "\n".join(logs[:10])), self._compute_reward("info_gathered"), False

        elif at == "check_metrics":
            service = params.get("service", "")
            if service == "payment-db":
                self._payment_db_investigated = True
                result = (f"Metrics for {service}:\n"
                    "  error_rate: 0.95\n  latency_p99_ms: 30000\n"
                    "  active_connections: 300/300 ⚠️\n"
                    "  blocked_queries: 892 ⚠️\n"
                    "  locks_exclusive_count: 1 ⚠️\n"
                    "  longest_running_query_sec: 720 ⚠️ (ALTER TABLE, 12 min)\n"
                    "  cpu_percent: 85\n  disk_io_util: 95%")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            elif service == "payment-service":
                self._payment_failure_noticed = True
                result = (f"Metrics for {service}:\n"
                    "  error_rate: 0.92\n  latency_p99_ms: 30500\n"
                    "  payment_success_rate: 0.08 ⚠️\n  retry_queue_depth: 1500")
                return self._build_observation(result), self._compute_reward("info_gathered"), False
            else:
                summary = MetricsSimulator.get_metrics_summary(service, self.task_id, self.seed)
                result = f"Metrics for {service}:\n"
                for k, v in summary.items():
                    if k != "service": result += f"  {k}: {v}\n"
                return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_alerts":
            result = ("Active alerts:\n"
                "  [CRIT] payment-db: 892 queries blocked by exclusive lock\n"
                "  [CRIT] payment-service: payment success rate 8%\n"
                "  [WARN] payment-db: ALTER TABLE running for 12+ minutes (pid 8234)\n"
                "  [WARN] order-service: payment timeout affecting order completion")
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_deployments":
            service_filter = params.get("service")
            deploys = DeployHistory.get_deploys(self.seed, last_n=5, service_filter=service_filter)
            result = "Recent deployments:\n"
            for d in deploys:
                result += f"  [{d.deploy_id}] {d.service} — {d.description} ({d.minutes_ago}min ago)\n"
            result += ("\n  No code deploy — but DBA migration ticket DB-4521 found:\n"
                       "    'Add refund_id column to payment_transactions'\n"
                       "    Scheduled: off-peak window (2 AM) but ran at 6 AM (peak) ⚠️\n"
                       "    lock_timeout: NOT SET ⚠️")
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_dependencies":
            service = params.get("service", "")
            upstream = self.graph.get_upstream_of(service)
            downstream = self.graph.get_downstream(service)
            return self._build_observation(f"Dependencies for {service}:\n  Upstream: {upstream}\n  Downstream: {downstream}"), self._compute_reward("info_gathered"), False

        elif at == "run_diagnostic":
            service = params.get("service", "")
            diag_type = params.get("type", "general")
            if service == "payment-db" and diag_type in ("locks", "blocking", "queries", "general"):
                self._lock_detected = True
                self._migration_identified = True
                result = (f"Diagnostic (locks) for {service}:\n"
                    "  Blocking query tree:\n"
                    "    PID 8234 (ACTIVE, 12min) — ALTER TABLE payment_transactions ADD COLUMN refund_id UUID\n"
                    "      Lock type: AccessExclusiveLock on payment_transactions\n"
                    "      User: dba_admin\n"
                    "      lock_timeout: 0 (disabled) ⚠️\n"
                    "      Progress: rewriting table, 65% complete\n"
                    "      ETA at current rate: ~8 more minutes\n"
                    "    ├── PID 8301 (BLOCKED, 11min) — INSERT INTO payment_transactions ...\n"
                    "    ├── PID 8302 (BLOCKED, 11min) — INSERT INTO payment_transactions ...\n"
                    "    └── ... 890 more blocked queries\n\n"
                    "  Table size: payment_transactions — 45GB, 120M rows\n"
                    "  ALTER TABLE requires full table rewrite (Postgres <12 behavior)\n\n"
                    "  💡 Root cause: DBA ran table migration during peak hours.\n"
                    "  AccessExclusiveLock blocks ALL reads and writes.\n"
                    "  No lock_timeout set — migration will run until complete or killed.\n"
                    "  Options:\n"
                    "    1. Kill migration (pg_cancel_backend(8234)) — immediate relief\n"
                    "    2. Wait ~8 min for completion\n"
                    "  Recommended: Kill migration, reschedule for off-peak with lock_timeout.")
                return self._build_observation(result), self._compute_reward("root_cause_identified"), False
            else:
                return self._build_observation(f"Diagnostic ({diag_type}) for {service}:\n  Normal."), self._compute_reward("info_gathered"), False

        elif at == "apply_fix":
            service = params.get("service", "")
            fix_type = params.get("fix_type", "")
            if service == "payment-db" and fix_type in ("kill_migration", "cancel_migration",
                    "pg_cancel_backend", "kill_query", "terminate_alter"):
                self._correct_fix_applied = True
                self._fix_applied = True
                self._service_fixed = True
                self.incident_phase = IncidentPhase.VERIFYING
                result = ("Fix applied: ALTER TABLE migration cancelled (pid 8234).\n"
                    "  AccessExclusiveLock released.\n"
                    "  892 blocked queries now executing.\n"
                    "  payment_transactions table intact (changes rolled back).\n"
                    "  Migration to be rescheduled for off-peak with lock_timeout.\n"
                    "  Verify health to confirm recovery.")
                return self._build_observation(result), self._compute_reward("fix_applied_correctly"), False
            elif fix_type == "restart":
                result = f"Restart {service} — this would terminate ALL connections. Migration PID 8234 killed along with all active queries. Recovery slow."
                self._fix_applied = True
                self._service_fixed = True
                self.incident_phase = IncidentPhase.VERIFYING
                return self._build_observation(result), self._compute_reward("wrong_fix_applied"), False
            else:
                self._wrong_fix_count += 1
                return self._build_observation(f"Fix {fix_type} on {service} — no effect on lock."), self._compute_reward("no_effect"), False

        elif at == "verify_health":
            if self._service_fixed:
                self._resolution_verified = True
                self.incident_phase = IncidentPhase.RESOLVED
                result = ("Health check: ALL SERVICES HEALTHY\n"
                    "  payment-db: blocked_queries=0, connections=85/300, latency_p99=35ms\n"
                    "  payment-service: payment_success_rate=99.5%\n"
                    "Incident resolved.")
                return self._build_observation(result), self._compute_reward("resolution_verified"), True
            else:
                return self._build_observation("Health check: ISSUES PERSIST\n  payment-db: 892 queries still blocked."), self._compute_reward("no_effect"), False

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
            result = "Hint: payment-db has a long-running ALTER TABLE holding an exclusive lock. Check pg_locks and consider killing the migration."
            return self._build_observation(result), self._compute_reward("escalate_used"), False

        else:
            return self._build_observation(f"Unknown action type: {at}"), self._compute_reward("no_effect"), False

    def get_grader_score(self) -> float:
        score = 0.0
        score += self._payment_failure_noticed * 0.08
        score += self._payment_db_investigated * 0.10
        score += self._lock_detected * 0.15
        score += self._migration_identified * 0.12
        score += self._correct_fix_applied * 0.22
        score += self._resolution_verified * 0.13
        score += self._postmortem_written * 0.08
        time_bonus = max(0, 1 - (self.step_count / self.max_steps)) * 0.05
        score += time_bonus
        score += self._evidence_breadth_score()
        score += self._postmortem_quality_bonus(["alter table", "lock", "migration", "exclusive", "payment", "peak"])
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
