"""
Task 8: Redis Cluster Failover — Medium-Hard difficulty.

Root cause: inventory-service Redis primary node failed, Sentinel
promoted a replica but is misconfigured (quorum=3, only 2 sentinels alive).
Failover takes 5+ minutes → cache miss storm hits inventory-db.
"""
from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.scenarios.base import BaseScenario
from app.data.log_templates import LogGenerator
from app.data.metrics import MetricsSimulator
from app.data.deploy_history import DeployHistory


class RedisFailoverScenario(BaseScenario):
    task_id = "task8_redis_failover"
    max_steps = 30

    def _correct_severity(self) -> str:
        return "SEV2"

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self._cache_miss_noticed = False
        self._redis_down_identified = False
        self._sentinel_issue_identified = False
        self._correct_fix_applied = False
        self._service_fixed = False
        self._wrong_fix_count = 0
        self._root_cause_service = "inventory-service"

    def _get_alert_summary(self) -> str:
        if self._service_fixed:
            return "RESOLVED: inventory-service cache recovered, inventory-db load normalized."
        return ("WARN: inventory-service latency spike. inventory-db query rate 5x normal. "
                "Cache hit rate dropped from 95% to 2%. Started ~8 minutes ago.")

    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        statuses = {}
        for svc in self.graph.get_all_services():
            if self._service_fixed:
                statuses[svc] = ServiceStatus(name=svc, status="healthy", error_rate=0.01, latency_p99_ms=50, restarts_last_hour=0)
            elif svc == "inventory-service":
                statuses[svc] = ServiceStatus(name=svc, status="degraded", error_rate=0.25, latency_p99_ms=8500, restarts_last_hour=0)
            elif svc == "inventory-db":
                statuses[svc] = ServiceStatus(name=svc, status="degraded", error_rate=0.05, latency_p99_ms=3200, restarts_last_hour=0)
            elif svc == "order-service":
                statuses[svc] = ServiceStatus(name=svc, status="degraded", error_rate=0.18, latency_p99_ms=9200, restarts_last_hour=0)
            else:
                statuses[svc] = ServiceStatus(name=svc, status="healthy", error_rate=0.01, latency_p99_ms=50, restarts_last_hour=0)
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
            if service == "inventory-service":
                result = (f"Logs for {service}:\n"
                    "[2026-03-26T03:10:05Z] [ERROR] Redis connection refused: primary node 10.0.2.5:6379 unreachable [service=inventory-service]\n"
                    "[2026-03-26T03:10:08Z] [WARN]  Falling back to inventory-db for all lookups — cache unavailable [service=inventory-service]\n"
                    "[2026-03-26T03:10:12Z] [ERROR] Connection pool to inventory-db saturated (150/150 connections) [service=inventory-service]\n"
                    "[2026-03-26T03:10:15Z] [WARN]  Redis Sentinel reports: +sdown master mymaster 10.0.2.5 6379 [service=inventory-service]\n"
                    "[2026-03-26T03:10:20Z] [ERROR] Sentinel failover NOT proceeding — insufficient quorum (need 3, have 2) [service=inventory-service]\n"
                    "[2026-03-26T03:10:25Z] [WARN]  Cache hit rate: 2.1% (normally 95%) — all reads hitting database [service=inventory-service]\n"
                    "[2026-03-26T03:10:30Z] [ERROR] inventory-db response time 3200ms (threshold: 500ms) [service=inventory-service]")
                self._cache_miss_noticed = True
                self._redis_down_identified = True
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            elif service == "inventory-db":
                result = (f"Logs for {service}:\n"
                    "[2026-03-26T03:10:10Z] [WARN]  Query rate spiked from 200/s → 1100/s [service=inventory-db]\n"
                    "[2026-03-26T03:10:15Z] [WARN]  Active connections: 148/150 (near limit) [service=inventory-db]\n"
                    "[2026-03-26T03:10:20Z] [ERROR] Query timeout: SELECT * FROM inventory WHERE sku=... (waited 3s) [service=inventory-db]\n"
                    "[2026-03-26T03:10:25Z] [WARN]  Buffer pool hit rate dropped to 62% under load [service=inventory-db]")
                return self._build_observation(result), self._compute_reward("info_gathered"), False
            else:
                logs = LogGenerator.generate(self.task_id, service, self.seed, step_count=self.step_count)
                return self._build_observation(f"Logs for {service}:\n" + "\n".join(logs[:10])), self._compute_reward("info_gathered"), False

        elif at == "check_metrics":
            service = params.get("service", "")
            if service == "inventory-service":
                self._cache_miss_noticed = True
                result = (f"Metrics for {service}:\n"
                    "  error_rate: 0.25\n  latency_p99_ms: 8500\n  cpu_percent: 45\n  memory_percent: 50\n"
                    "  redis_cache_hit_rate: 0.021 ⚠️ CRITICAL (normally 0.95)\n"
                    "  redis_connection_status: DISCONNECTED ⚠️\n"
                    "  db_connection_pool_used: 148/150 ⚠️\n"
                    "  queries_per_sec_to_db: 1100 (normally 200)")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            else:
                summary = MetricsSimulator.get_metrics_summary(service, self.task_id, self.seed)
                result = f"Metrics for {service}:\n"
                for k, v in summary.items():
                    if k != "service": result += f"  {k}: {v}\n"
                return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_alerts":
            result = ("Active alerts:\n"
                "  [CRIT] inventory-service: Redis primary unreachable for 8 minutes\n"
                "  [WARN] inventory-service: cache hit rate 2.1% (SLA: >90%)\n"
                "  [WARN] inventory-db: connection pool 98% utilized\n"
                "  [WARN] order-service: inventory lookup latency 8.5s (SLA: 1s)")
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_deployments":
            service_filter = params.get("service")
            deploys = DeployHistory.get_deploys(self.seed, last_n=5, service_filter=service_filter)
            result = "Recent deployments:\n"
            for d in deploys:
                result += f"  [{d.deploy_id}] {d.service} — {d.description} ({d.minutes_ago}min ago)\n"
            result += "\n  No deployment in past 24h relates to Redis or cache config."
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_dependencies":
            service = params.get("service", "")
            upstream = self.graph.get_upstream_of(service)
            downstream = self.graph.get_downstream(service)
            result = f"Dependencies for {service}:\n  Upstream: {upstream}\n  Downstream: {downstream}"
            if service == "inventory-service":
                result += "\n\n  ⚠️  inventory-service also uses Redis cluster (redis-sentinel://10.0.2.{5,6,7}:26379)"
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "run_diagnostic":
            service = params.get("service", "")
            diag_type = params.get("type", "general")
            if service == "inventory-service" and diag_type in ("redis", "cache", "sentinel", "general"):
                self._sentinel_issue_identified = True
                result = (f"Diagnostic (redis/sentinel) for {service}:\n"
                    "  redis_primary: 10.0.2.5:6379 — DOWN (unreachable since 8min ago)\n"
                    "  redis_replica_1: 10.0.2.6:6379 — UP (read-only, 0s repl lag)\n"
                    "  redis_replica_2: 10.0.2.7:6379 — UP (read-only, 0s repl lag)\n\n"
                    "  sentinel_1: 10.0.2.5:26379 — DOWN (on same node as primary) ⚠️\n"
                    "  sentinel_2: 10.0.2.6:26379 — UP\n"
                    "  sentinel_3: 10.0.2.7:26379 — UP\n\n"
                    "  Sentinel quorum required: 3 ⚠️\n"
                    "  Sentinel quorum available: 2 (sentinel_1 is down with primary)\n"
                    "  Failover status: BLOCKED — cannot reach quorum\n\n"
                    "  💡 Root cause: Redis primary and sentinel_1 are co-located.\n"
                    "  When primary died, sentinel_1 also went down.\n"
                    "  Quorum requires 3 but only 2 sentinels alive.\n"
                    "  Fix: reduce quorum to 2 OR manually promote replica.")
                return self._build_observation(result), self._compute_reward("root_cause_identified"), False
            else:
                return self._build_observation(f"Diagnostic ({diag_type}) for {service}:\n  All within normal range."), self._compute_reward("info_gathered"), False

        elif at == "apply_fix":
            service = params.get("service", "")
            fix_type = params.get("fix_type", "")
            if service == "inventory-service" and fix_type in ("reduce_quorum", "fix_sentinel_quorum",
                    "manual_failover", "promote_replica", "force_failover"):
                self._correct_fix_applied = True
                self._fix_applied = True
                self._service_fixed = True
                self.incident_phase = IncidentPhase.VERIFYING
                result = ("Fix applied: Redis failover completed.\n"
                    "  Sentinel quorum reduced to 2 — failover triggered.\n"
                    "  Replica 10.0.2.6 promoted to primary.\n"
                    "  inventory-service reconnected to new primary.\n"
                    "  Cache rebuilding — hit rate climbing.\n"
                    "  Verify health to confirm resolution.")
                return self._build_observation(result), self._compute_reward("fix_applied_correctly"), False
            elif fix_type == "restart":
                result = f"Restart applied to {service}. Cache miss storm continues — Redis primary still down."
                return self._build_observation(result), self._compute_reward("wrong_fix_applied"), False
            else:
                self._wrong_fix_count += 1
                return self._build_observation(f"Fix {fix_type} on {service} — no effect on Redis failover."), self._compute_reward("no_effect"), False

        elif at == "verify_health":
            if self._service_fixed:
                self._resolution_verified = True
                self.incident_phase = IncidentPhase.RESOLVED
                result = ("Health check: ALL SERVICES HEALTHY\n"
                    "  inventory-service: cache_hit_rate=0.88 (recovering), latency_p99=120ms\n"
                    "  inventory-db: query_rate=250/s (normal), connections=35/150\n"
                    "Incident resolved.")
                return self._build_observation(result), self._compute_reward("resolution_verified"), True
            else:
                result = "Health check: ISSUES PERSIST\n  inventory-service: Redis primary still unreachable."
                return self._build_observation(result), self._compute_reward("no_effect"), False

        elif at == "write_postmortem":
            content = params.get("content", "")
            if len(content) > 50:
                self._postmortem_written = True
                result = "Postmortem recorded successfully."
                reward = self._compute_reward("postmortem_written")
            else:
                result = "Postmortem too short."
                reward = self._compute_reward("no_effect")
            return self._build_observation(result), reward, self.incident_phase == IncidentPhase.RESOLVED

        elif at == "escalate":
            self.hints_used += 1
            result = ("Escalation hint: Redis primary is down and sentinel cannot failover.\n"
                "Check sentinel quorum config — sentinel_1 may be co-located with the failed primary.")
            return self._build_observation(result), self._compute_reward("escalate_used"), False

        else:
            return self._build_observation(f"Unknown action type: {at}"), self._compute_reward("no_effect"), False

    def get_grader_score(self) -> float:
        score = 0.0
        score += self._cache_miss_noticed * 0.12
        score += self._redis_down_identified * 0.12
        score += self._sentinel_issue_identified * 0.15
        score += self._correct_fix_applied * 0.22
        score += self._resolution_verified * 0.13
        score += self._postmortem_written * 0.08
        time_bonus = max(0, 1 - (self.step_count / self.max_steps)) * 0.05
        score += time_bonus
        score += self._evidence_breadth_score()
        score += self._postmortem_quality_bonus(["redis", "sentinel", "quorum", "failover", "primary", "cache"])
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
