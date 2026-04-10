"""
Task 12: Health Check Flapping — Medium-Hard difficulty.

Root cause: order-service health check endpoint performs a deep
dependency check that calls inventory-service. When inventory-service
has intermittent latency spikes (GC pauses), the health check times
out and the load balancer removes the instance.
"""
from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.scenarios.base import BaseScenario
from app.data.log_templates import LogGenerator
from app.data.metrics import MetricsSimulator
from app.data.deploy_history import DeployHistory


class HealthFlapScenario(BaseScenario):
    task_id = "task12_health_flap"
    max_steps = 30

    def _correct_severity(self) -> str:
        return "SEV2"

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self._flapping_noticed = False
        self._health_check_investigated = False
        self._deep_check_identified = False
        self._correct_fix_applied = False
        self._service_fixed = False
        self._wrong_fix_count = 0

    def _get_alert_summary(self) -> str:
        if self._service_fixed:
            return "RESOLVED: order-service instances stable in load balancer."
        return ("WARN: order-service instances flapping in/out of load balancer. "
                "3 of 5 instances currently removed. Intermittent 503 errors. "
                "Pattern: fail → remove → recover → add → fail again. ~15 minutes.")

    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        statuses = {}
        for svc in self.graph.get_all_services():
            if self._service_fixed:
                statuses[svc] = ServiceStatus(name=svc, status="healthy", error_rate=0.01, latency_p99_ms=50, restarts_last_hour=0)
            elif svc == "order-service":
                statuses[svc] = ServiceStatus(name=svc, status="degraded", error_rate=0.30, latency_p99_ms=4500, restarts_last_hour=0)
            elif svc == "inventory-service":
                statuses[svc] = ServiceStatus(name=svc, status="degraded", error_rate=0.05, latency_p99_ms=2200, restarts_last_hour=0)
            elif svc == "api-gateway":
                statuses[svc] = ServiceStatus(name=svc, status="degraded", error_rate=0.25, latency_p99_ms=5000, restarts_last_hour=0)
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
            if service == "order-service":
                self._flapping_noticed = True
                result = (f"Logs for {service}:\n"
                    "[2026-03-26T07:10:01Z] [WARN]  Health check timeout: /health/deep took 3200ms (limit: 2000ms) [service=order-service]\n"
                    "[2026-03-26T07:10:02Z] [WARN]  Load balancer removed instance order-svc-3 (failed 3 consecutive checks) [service=order-service]\n"
                    "[2026-03-26T07:10:15Z] [INFO]  Health check passed: /health/deep responded in 450ms [service=order-service]\n"
                    "[2026-03-26T07:10:30Z] [INFO]  Load balancer re-added instance order-svc-3 [service=order-service]\n"
                    "[2026-03-26T07:10:45Z] [WARN]  Health check timeout: /health/deep took 4100ms (limit: 2000ms) [service=order-service]\n"
                    "[2026-03-26T07:10:46Z] [WARN]  Load balancer removed instance order-svc-3 again (flap #8 in 15min) [service=order-service]\n"
                    "[2026-03-26T07:10:50Z] [INFO]  /health/deep calls: inventory-service /inventory/ping (p99: 1800ms) ⚠️ [service=order-service]\n"
                    "[2026-03-26T07:11:00Z] [WARN]  Instances in LB: 2/5 active (3 removed by health flapping) [service=order-service]")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            elif service == "inventory-service":
                result = (f"Logs for {service}:\n"
                    "[2026-03-26T07:10:00Z] [WARN]  GC pause: 1200ms (full GC) [service=inventory-service]\n"
                    "[2026-03-26T07:10:40Z] [WARN]  GC pause: 980ms (full GC) [service=inventory-service]\n"
                    "[2026-03-26T07:11:20Z] [INFO]  Heap usage: 85% — approaching GC threshold [service=inventory-service]\n"
                    "[2026-03-26T07:11:30Z] [WARN]  /inventory/ping response time: 1800ms (includes GC pause) [service=inventory-service]")
                return self._build_observation(result), self._compute_reward("info_gathered"), False
            else:
                logs = LogGenerator.generate(self.task_id, service, self.seed, step_count=self.step_count)
                return self._build_observation(f"Logs for {service}:\n" + "\n".join(logs[:10])), self._compute_reward("info_gathered"), False

        elif at == "check_metrics":
            service = params.get("service", "")
            if service == "order-service":
                self._flapping_noticed = True
                result = (f"Metrics for {service}:\n"
                    "  error_rate: 0.30\n  latency_p99_ms: 4500\n"
                    "  instances_in_lb: 2/5 ⚠️\n"
                    "  health_check_flap_count_15m: 24 ⚠️\n"
                    "  health_check_timeout_rate: 0.60 ⚠️\n"
                    "  health_endpoint: /health/deep\n"
                    "  health_timeout_ms: 2000\n"
                    "  cpu_percent: 25\n  memory_percent: 40")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            elif service == "inventory-service":
                result = (f"Metrics for {service}:\n"
                    "  error_rate: 0.05\n  latency_p99_ms: 2200\n"
                    "  gc_pause_p99_ms: 1200 ⚠️\n"
                    "  gc_collections_15m: 18\n"
                    "  heap_usage_percent: 85")
                return self._build_observation(result), self._compute_reward("info_gathered"), False
            else:
                summary = MetricsSimulator.get_metrics_summary(service, self.task_id, self.seed)
                result = f"Metrics for {service}:\n"
                for k, v in summary.items():
                    if k != "service": result += f"  {k}: {v}\n"
                return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_alerts":
            result = ("Active alerts:\n"
                "  [WARN] order-service: 3/5 instances removed from load balancer\n"
                "  [WARN] order-service: health check flapping — 24 flaps in 15 minutes\n"
                "  [WARN] api-gateway: order endpoints returning intermittent 503s\n"
                "  [INFO] inventory-service: GC pauses up to 1200ms")
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_deployments":
            service_filter = params.get("service")
            deploys = DeployHistory.get_deploys(self.seed, last_n=5, service_filter=service_filter)
            result = "Recent deployments:\n"
            for d in deploys:
                result += f"  [{d.deploy_id}] {d.service} — {d.description} ({d.minutes_ago}min ago)\n"
            result += ("\n  [deploy-health-01] order-service — 'Enhanced health check: add deep dependency validation' (2 hours ago) ⚠️\n"
                       "    Added: /health/deep now calls inventory-service /inventory/ping, payment-service /payment/ping\n"
                       "    Previous: /health/shallow only checked local readiness")
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_dependencies":
            service = params.get("service", "")
            upstream = self.graph.get_upstream_of(service)
            downstream = self.graph.get_downstream(service)
            result = f"Dependencies for {service}:\n  Upstream: {upstream}\n  Downstream: {downstream}"
            if service == "order-service":
                result += ("\n\n  ⚠️  order-service /health/deep endpoint calls:\n"
                           "    - inventory-service /inventory/ping\n"
                           "    - payment-service /payment/ping\n"
                           "  If either dependency is slow, health check fails.")
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "run_diagnostic":
            service = params.get("service", "")
            diag_type = params.get("type", "general")
            if service == "order-service" and diag_type in ("health_check", "health", "flap", "general"):
                self._health_check_investigated = True
                self._deep_check_identified = True
                result = (f"Diagnostic (health_check) for {service}:\n"
                    "  Health endpoint: /health/deep\n"
                    "  Timeout: 2000ms\n"
                    "  LB failure threshold: 3 consecutive failures\n"
                    "  LB recovery threshold: 2 consecutive passes\n\n"
                    "  /health/deep breakdown:\n"
                    "    Local checks: 5ms (CPU, memory, disk) ✓\n"
                    "    DB connection check: 8ms ✓\n"
                    "    inventory-service /inventory/ping: 400-1800ms ⚠️ (intermittent >2000ms)\n"
                    "    payment-service /payment/ping: 15ms ✓\n\n"
                    "  Problem analysis:\n"
                    "    inventory-service has GC pauses up to 1200ms\n"
                    "    /inventory/ping during GC: ~1800ms\n"
                    "    Combined /health/deep: sometimes exceeds 2000ms timeout\n"
                    "    → LB removes instance → traffic redistributed → overload other instances\n\n"
                    "  💡 Root cause: /health/deep includes synchronous call to inventory-service.\n"
                    "  inventory-service GC pauses cause intermittent health check timeouts.\n"
                    "  Fix: switch to /health/shallow (local-only checks) OR increase timeout OR\n"
                    "  make dependency checks async/non-blocking.")
                return self._build_observation(result), self._compute_reward("root_cause_identified"), False
            else:
                return self._build_observation(f"Diagnostic ({diag_type}) for {service}:\n  Normal."), self._compute_reward("info_gathered"), False

        elif at == "apply_fix":
            service = params.get("service", "")
            fix_type = params.get("fix_type", "")
            if service == "order-service" and fix_type in ("use_shallow_health_check",
                    "simplify_health_check", "switch_shallow", "increase_health_timeout",
                    "remove_deep_check", "decouple_health_check"):
                self._correct_fix_applied = True
                self._fix_applied = True
                self._service_fixed = True
                self.incident_phase = IncidentPhase.VERIFYING
                result = ("Fix applied: order-service health check reconfigured.\n"
                    "  Health endpoint changed to /health/shallow (local checks only).\n"
                    "  All 5 instances passing health checks.\n"
                    "  LB re-adding previously removed instances.\n"
                    "  Verify health to confirm stability.")
                return self._build_observation(result), self._compute_reward("fix_applied_correctly"), False
            elif fix_type == "restart":
                result = f"Restart {service} — instances restarted but flapping resumes within 2 minutes."
                return self._build_observation(result), self._compute_reward("wrong_fix_applied"), False
            else:
                self._wrong_fix_count += 1
                return self._build_observation(f"Fix {fix_type} on {service} — no effect on flapping."), self._compute_reward("no_effect"), False

        elif at == "verify_health":
            if self._service_fixed:
                self._resolution_verified = True
                self.incident_phase = IncidentPhase.RESOLVED
                result = ("Health check: ALL SERVICES HEALTHY\n"
                    "  order-service: 5/5 instances in LB, flap_count=0, error_rate=0.01\n"
                    "  api-gateway: no more 503 errors\n"
                    "Incident resolved.")
                return self._build_observation(result), self._compute_reward("resolution_verified"), True
            else:
                return self._build_observation("Health check: ISSUES PERSIST\n  order-service: still flapping, 2/5 in LB."), self._compute_reward("no_effect"), False

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
            result = "Hint: order-service health check calls inventory-service. GC pauses cause timeout. Switch to shallow health check."
            return self._build_observation(result), self._compute_reward("escalate_used"), False

        else:
            return self._build_observation(f"Unknown action type: {at}"), self._compute_reward("no_effect"), False

    def get_grader_score(self) -> float:
        score = 0.0
        score += self._flapping_noticed * 0.10
        score += self._health_check_investigated * 0.15
        score += self._deep_check_identified * 0.18
        score += self._correct_fix_applied * 0.25
        score += self._resolution_verified * 0.13
        score += self._postmortem_written * 0.08
        time_bonus = max(0, 1 - (self.step_count / self.max_steps)) * 0.05
        score += time_bonus
        score += self._evidence_breadth_score()
        score += self._postmortem_quality_bonus(["health check", "flap", "deep", "shallow", "inventory", "timeout"])
        score += self._severity_correct * 0.02
        score += (self._status_page_updated and self._status_page_before_fix) * 0.02
        if self._fix_applied and self._fix_before_any_breach:
            score += 0.02
        score -= self._wrong_fix_count * 0.05
        score -= self.hints_used * 0.05
        return round(min(0.999, max(0.001, score)), 4)
