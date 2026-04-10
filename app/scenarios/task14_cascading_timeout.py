"""
Task 14: Cascading Timeout — Medium-Hard difficulty.

Root cause: inventory-service has a slow query (25s) due to missing index.
order-service has 30s timeout, api-gateway has 10s timeout.
Users see 504s because api-gateway times out before the response arrives.
"""
from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.scenarios.base import BaseScenario
from app.data.log_templates import LogGenerator
from app.data.metrics import MetricsSimulator
from app.data.deploy_history import DeployHistory


class CascadingTimeoutScenario(BaseScenario):
    task_id = "task14_cascading_timeout"
    max_steps = 30

    def _correct_severity(self) -> str:
        return "SEV2"

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self._timeout_noticed = False
        self._inventory_slow_detected = False
        self._timeout_hierarchy_identified = False
        self._correct_fix_applied = False
        self._service_fixed = False
        self._wrong_fix_count = 0

    def _get_alert_summary(self) -> str:
        if self._service_fixed:
            return "RESOLVED: timeout chain resolved, user-facing latency normalized."
        return ("WARN: api-gateway returning 504 Gateway Timeout for /orders endpoints. "
                "Users reporting slow and failing requests. Intermittent — some succeed, most timeout. "
                "Started ~20 minutes ago after inventory-db index rebuild.")

    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        statuses = {}
        for svc in self.graph.get_all_services():
            if self._service_fixed:
                statuses[svc] = ServiceStatus(name=svc, status="healthy", error_rate=0.01, latency_p99_ms=80, restarts_last_hour=0)
            elif svc == "api-gateway":
                statuses[svc] = ServiceStatus(name=svc, status="degraded", error_rate=0.70, latency_p99_ms=10000, restarts_last_hour=0)
            elif svc == "order-service":
                statuses[svc] = ServiceStatus(name=svc, status="degraded", error_rate=0.65, latency_p99_ms=25500, restarts_last_hour=0)
            elif svc == "inventory-service":
                statuses[svc] = ServiceStatus(name=svc, status="degraded", error_rate=0.10, latency_p99_ms=25000, restarts_last_hour=0)
            elif svc == "inventory-db":
                statuses[svc] = ServiceStatus(name=svc, status="degraded", error_rate=0.02, latency_p99_ms=24000, restarts_last_hour=0)
            else:
                statuses[svc] = ServiceStatus(name=svc, status="healthy", error_rate=0.01, latency_p99_ms=80, restarts_last_hour=0)
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
            if service == "api-gateway":
                self._timeout_noticed = True
                result = (f"Logs for {service}:\n"
                    "[2026-03-26T09:00:05Z] [ERROR] 504 Gateway Timeout: /orders/ORD-1234 — upstream did not respond within 10000ms [service=api-gateway]\n"
                    "[2026-03-26T09:00:08Z] [ERROR] 504 Gateway Timeout: /orders/ORD-1235 — upstream timeout [service=api-gateway]\n"
                    "[2026-03-26T09:00:10Z] [WARN]  Upstream timeout config: order-service=10000ms [service=api-gateway]\n"
                    "[2026-03-26T09:00:12Z] [ERROR] 70% of /orders requests timing out at 10s [service=api-gateway]\n"
                    "[2026-03-26T09:00:15Z] [INFO]  Non-order endpoints (/auth, /user) responding normally [service=api-gateway]")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            elif service == "order-service":
                result = (f"Logs for {service}:\n"
                    "[2026-03-26T09:00:06Z] [WARN]  inventory-service call took 25200ms — near timeout limit (30000ms) [service=order-service]\n"
                    "[2026-03-26T09:00:09Z] [ERROR] Upstream reply arrived but api-gateway already timed out (client disconnected) [service=order-service]\n"
                    "[2026-03-26T09:00:12Z] [WARN]  Timeout hierarchy: api-gateway(10s) < order-service→inventory(30s) ⚠️ [service=order-service]\n"
                    "[2026-03-26T09:00:15Z] [ERROR] Wasted work: 12 inventory lookups completed but response discarded by gateway timeout [service=order-service]")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            elif service == "inventory-service":
                self._inventory_slow_detected = True
                result = (f"Logs for {service}:\n"
                    "[2026-03-26T09:00:04Z] [WARN]  Slow query: SELECT * FROM inventory WHERE sku LIKE '%XY%' — 24800ms [service=inventory-service]\n"
                    "[2026-03-26T09:00:07Z] [WARN]  Missing index on inventory.sku column ⚠️ — full table scan (5M rows) [service=inventory-service]\n"
                    "[2026-03-26T09:00:10Z] [INFO]  Previous index dropped during last night's index rebuild maintenance [service=inventory-service]\n"
                    "[2026-03-26T09:00:15Z] [WARN]  Average query time 24s (normally 15ms with index) [service=inventory-service]")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            else:
                logs = LogGenerator.generate(self.task_id, service, self.seed, step_count=self.step_count)
                return self._build_observation(f"Logs for {service}:\n" + "\n".join(logs[:10])), self._compute_reward("info_gathered"), False

        elif at == "check_metrics":
            service = params.get("service", "")
            if service == "inventory-service":
                self._inventory_slow_detected = True
                result = (f"Metrics for {service}:\n"
                    "  error_rate: 0.10\n  latency_p99_ms: 25000 ⚠️\n"
                    "  db_query_p99_ms: 24800 ⚠️\n  cpu_percent: 60\n  memory_percent: 55\n"
                    "  slow_queries_10m: 847 ⚠️")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            elif service == "api-gateway":
                self._timeout_noticed = True
                result = (f"Metrics for {service}:\n"
                    "  error_rate: 0.70 ⚠️\n  latency_p99_ms: 10000 (hitting timeout)\n"
                    "  http_504_rate: 0.70 ⚠️\n  upstream_timeout_ms: 10000\n"
                    "  cpu_percent: 15\n  memory_percent: 25")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            else:
                summary = MetricsSimulator.get_metrics_summary(service, self.task_id, self.seed)
                result = f"Metrics for {service}:\n"
                for k, v in summary.items():
                    if k != "service": result += f"  {k}: {v}\n"
                return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_alerts":
            result = ("Active alerts:\n"
                "  [WARN] api-gateway: 70% of /orders requests returning 504\n"
                "  [WARN] order-service: inventory lookups averaging 25s\n"
                "  [WARN] inventory-service: query latency p99=25s (normal: 15ms)\n"
                "  [INFO] Timeout mismatch: api-gateway(10s) < order-service(30s)")
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_deployments":
            service_filter = params.get("service")
            deploys = DeployHistory.get_deploys(self.seed, last_n=5, service_filter=service_filter)
            result = "Recent deployments:\n"
            for d in deploys:
                result += f"  [{d.deploy_id}] {d.service} — {d.description} ({d.minutes_ago}min ago)\n"
            result += ("\n  [ops-index-rebuild] inventory-db — 'Nightly index rebuild maintenance' (45min ago)\n"
                       "    Note: index on inventory.sku was dropped for rebuild but NOT recreated ⚠️\n"
                       "    Maintenance script failed silently at CREATE INDEX step")
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_dependencies":
            service = params.get("service", "")
            upstream = self.graph.get_upstream_of(service)
            downstream = self.graph.get_downstream(service)
            result = f"Dependencies for {service}:\n  Upstream: {upstream}\n  Downstream: {downstream}"
            if service in ("api-gateway", "order-service"):
                result += ("\n\n  Timeout configuration:\n"
                           "    api-gateway → order-service: 10000ms\n"
                           "    order-service → inventory-service: 30000ms ⚠️ (higher than gateway!)\n"
                           "    order-service → payment-service: 15000ms\n"
                           "    inventory-service → inventory-db: 60000ms")
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "run_diagnostic":
            service = params.get("service", "")
            diag_type = params.get("type", "general")
            if service in ("inventory-service", "inventory-db") and diag_type in ("query", "slow_query", "index", "general"):
                self._timeout_hierarchy_identified = True
                result = (f"Diagnostic (query/index) for inventory stack:\n"
                    "  Slow query analysis:\n"
                    "    Query: SELECT * FROM inventory WHERE sku LIKE '%XY%'\n"
                    "    Execution plan: Seq Scan on inventory (5,000,000 rows)\n"
                    "    Duration: 24,800ms\n"
                    "    Expected with index: 15ms\n\n"
                    "  Missing indexes:\n"
                    "    inventory.sku — INDEX MISSING ⚠️\n"
                    "    (dropped by nightly maintenance, not recreated)\n\n"
                    "  Timeout hierarchy analysis:\n"
                    "    api-gateway → order-service: 10s\n"
                    "    order-service → inventory-service: 30s\n"
                    "    inventory-service → inventory-db: 60s\n"
                    "    Actual inventory query time: ~25s\n\n"
                    "  💡 Root cause: Missing index on inventory.sku causing 25s queries.\n"
                    "  Combined with timeout mismatch (gateway 10s < inventory 30s),\n"
                    "  users see 504 even though queries eventually complete.\n"
                    "  Fix: recreate the missing index (immediate relief).\n"
                    "  Also: adjust timeout hierarchy so downstream < upstream.")
                return self._build_observation(result), self._compute_reward("root_cause_identified"), False
            elif service == "api-gateway" and diag_type in ("timeout", "general"):
                self._timeout_hierarchy_identified = True
                result = (f"Diagnostic (timeout) for {service}:\n"
                    "  Timeout config:\n"
                    "    api-gateway → order-service: 10000ms\n"
                    "    order-service → inventory-service: 30000ms ⚠️\n\n"
                    "  Problem: downstream timeout (30s) > upstream timeout (10s)\n"
                    "  Work is wasted when gateway times out but inventory query completes.\n"
                    "  Fix: either speed up inventory queries or adjust timeout hierarchy.")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            else:
                return self._build_observation(f"Diagnostic ({diag_type}) for {service}:\n  Normal."), self._compute_reward("info_gathered"), False

        elif at == "apply_fix":
            service = params.get("service", "")
            fix_type = params.get("fix_type", "")
            if fix_type in ("recreate_index", "create_index", "add_index") or \
               (service in ("inventory-service", "inventory-db") and fix_type in ("fix_index", "rebuild_index")):
                self._correct_fix_applied = True
                self._fix_applied = True
                self._root_cause_identified = True
                self._service_fixed = True
                self.incident_phase = IncidentPhase.VERIFYING
                result = ("Fix applied: Index on inventory.sku recreated.\n"
                    "  CREATE INDEX idx_inventory_sku ON inventory(sku) — completed in 45s.\n"
                    "  inventory query time: 25000ms → 15ms\n"
                    "  order-service latency: 25s → 80ms\n"
                    "  api-gateway 504 rate dropping rapidly.\n"
                    "  Verify health to confirm.")
                return self._build_observation(result), self._compute_reward("fix_applied_correctly"), False
            elif fix_type in ("adjust_timeout", "fix_timeout_hierarchy", "increase_gateway_timeout"):
                result = ("Timeout adjusted — api-gateway now has 35s timeout.\n"
                    "  504 errors reduced but users still wait 25s per request (poor UX).\n"
                    "  Root cause (slow query) still present.")
                return self._build_observation(result), self._compute_reward("wrong_fix_applied"), False
            else:
                self._wrong_fix_count += 1
                return self._build_observation(f"Fix {fix_type} on {service} — no effect on timeouts."), self._compute_reward("no_effect"), False

        elif at == "verify_health":
            if self._service_fixed:
                self._resolution_verified = True
                self.incident_phase = IncidentPhase.RESOLVED
                result = ("Health check: ALL SERVICES HEALTHY\n"
                    "  inventory-service: query_p99=15ms, error_rate=0.01\n"
                    "  api-gateway: 504_rate=0.0, latency_p99=80ms\n"
                    "Incident resolved.")
                return self._build_observation(result), self._compute_reward("resolution_verified"), True
            else:
                return self._build_observation("Health check: ISSUES PERSIST\n  70% of /orders requests still timing out."), self._compute_reward("no_effect"), False

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
            result = "Hint: inventory-service has a missing index causing 25s queries. Index was dropped during nightly maintenance."
            return self._build_observation(result), self._compute_reward("escalate_used"), False

        else:
            return self._build_observation(f"Unknown action type: {at}"), self._compute_reward("no_effect"), False

    def get_grader_score(self) -> float:
        score = 0.0
        score += self._timeout_noticed * 0.10
        score += self._inventory_slow_detected * 0.12
        score += self._timeout_hierarchy_identified * 0.18
        score += self._correct_fix_applied * 0.25
        score += self._resolution_verified * 0.13
        score += self._postmortem_written * 0.08
        time_bonus = max(0, 1 - (self.step_count / self.max_steps)) * 0.05
        score += time_bonus
        score += self._evidence_breadth_score()
        score += self._postmortem_quality_bonus(["timeout", "index", "missing", "cascade", "504", "inventory", "query"])
        score += self._severity_correct * 0.02
        score += (self._status_page_updated and self._status_page_before_fix) * 0.02
        if self._fix_applied and self._fix_before_any_breach:
            score += 0.02
        score -= self._wrong_fix_count * 0.05
        score -= self.hints_used * 0.05
        return round(min(0.999, max(0.001, score)), 4)
