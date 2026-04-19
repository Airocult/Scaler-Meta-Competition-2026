"""
Task 10: API Rate Limiting Misconfiguration — Medium difficulty.

Root cause: A deploy to api-gateway changed the rate limit from
10000 req/s to 100 req/s (typo — missing zero). 90% of legitimate
traffic is being throttled with 429 responses.
"""
from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.scenarios.base import BaseScenario
from app.data.log_templates import LogGenerator
from app.data.metrics import MetricsSimulator
from app.data.deploy_history import DeployHistory


class RateLimitScenario(BaseScenario):
    task_id = "task10_rate_limit"
    max_steps = 25

    def _correct_severity(self) -> str:
        return "SEV1"

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self._throttling_noticed = False
        self._deploy_identified = False
        self._rate_limit_misconfigured = False
        self._correct_fix_applied = False
        self._service_fixed = False
        self._wrong_fix_count = 0
        self._root_cause_service = "api-gateway"

    def _get_alert_summary(self) -> str:
        if self._service_fixed:
            return "RESOLVED: api-gateway rate limiting restored, traffic flowing normally."
        return ("CRIT: api-gateway returning HTTP 429 for 90% of requests. "
                "All downstream services seeing 90% drop in traffic. Started ~7 minutes ago.")

    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        statuses = {}
        for svc in self.graph.get_all_services():
            if self._service_fixed:
                statuses[svc] = ServiceStatus(name=svc, status="healthy", error_rate=0.01, latency_p99_ms=50, restarts_last_hour=0)
            elif svc == "api-gateway":
                statuses[svc] = ServiceStatus(name=svc, status="down", error_rate=0.90, latency_p99_ms=5, restarts_last_hour=0)
            else:
                statuses[svc] = ServiceStatus(name=svc, status="healthy", error_rate=0.02, latency_p99_ms=50, restarts_last_hour=0)
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
                self._throttling_noticed = True
                result = (f"Logs for {service}:\n"
                    "[2026-03-26T05:05:02Z] [WARN]  Rate limit exceeded for client 10.1.0.50: 429 Too Many Requests [service=api-gateway]\n"
                    "[2026-03-26T05:05:02Z] [WARN]  Rate limit exceeded for client 10.1.0.51: 429 Too Many Requests [service=api-gateway]\n"
                    "[2026-03-26T05:05:03Z] [WARN]  9247 requests throttled in last 10 seconds (limit: 100/s) [service=api-gateway]\n"
                    "[2026-03-26T05:05:05Z] [INFO]  Rate limiter config: max_requests_per_second=100, burst=10 [service=api-gateway]\n"
                    "[2026-03-26T05:05:08Z] [WARN]  Legitimate traffic estimate: 1050 req/s — current limit permits only 100/s [service=api-gateway]\n"
                    "[2026-03-26T05:05:10Z] [ERROR] Customer reports: 'Cannot access API — getting 429 errors' [service=api-gateway]")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            else:
                logs = LogGenerator.generate(self.task_id, service, self.seed, step_count=self.step_count)
                return self._build_observation(f"Logs for {service}:\n" + "\n".join(logs[:10])), self._compute_reward("info_gathered"), False

        elif at == "check_metrics":
            service = params.get("service", "")
            if service == "api-gateway":
                self._throttling_noticed = True
                result = (f"Metrics for {service}:\n"
                    "  error_rate: 0.90 ⚠️ CRITICAL (almost all 429s)\n"
                    "  latency_p99_ms: 5 (fast — just returning 429)\n"
                    "  requests_per_sec: 1050\n"
                    "  requests_throttled_per_sec: 950 ⚠️\n"
                    "  rate_limit_max: 100 ⚠️ (normally 10000)\n"
                    "  http_429_count_1m: 57000 ⚠️\n"
                    "  cpu_percent: 12\n  memory_percent: 30")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            else:
                summary = MetricsSimulator.get_metrics_summary(service, self.task_id, self.seed)
                result = f"Metrics for {service}:\n"
                for k, v in summary.items():
                    if k != "service": result += f"  {k}: {v}\n"
                return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_alerts":
            result = ("Active alerts:\n"
                "  [CRIT] api-gateway: 90% of requests returning HTTP 429\n"
                "  [WARN] api-gateway: rate_limit_max=100 (expected: 10000)\n"
                "  [WARN] All downstream services: traffic dropped 90%\n"
                "  [INFO] Customer complaints: 'API returning 429 errors'")
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_deployments":
            service_filter = params.get("service")
            deploys = DeployHistory.get_deploys(self.seed, last_n=5, service_filter=service_filter)
            result = "Recent deployments:\n"
            for d in deploys:
                result += f"  [{d.deploy_id}] {d.service} — {d.description} ({d.minutes_ago}min ago)\n"
            result += ("\n  [deploy-ratelimit-01] api-gateway — 'Update rate limiter config' (7min ago) ⚠️\n"
                       "    Changed: rate_limit.max_requests_per_second: 10000 → 100\n"
                       "    Changed: rate_limit.burst_size: 1000 → 10")
            self._deploy_identified = True
            return self._build_observation(result), self._compute_reward("root_cause_progress"), False

        elif at == "check_dependencies":
            service = params.get("service", "")
            upstream = self.graph.get_upstream_of(service)
            downstream = self.graph.get_downstream(service)
            result = f"Dependencies for {service}:\n  Upstream: {upstream}\n  Downstream: {downstream}"
            if service == "api-gateway":
                result += "\n\n  api-gateway is the entry point — all external traffic flows through it."
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "run_diagnostic":
            service = params.get("service", "")
            diag_type = params.get("type", "general")
            if service == "api-gateway" and diag_type in ("rate_limit", "config", "general"):
                self._rate_limit_misconfigured = True
                result = (f"Diagnostic (rate_limit) for {service}:\n"
                    "  Current config:\n"
                    "    max_requests_per_second: 100 ⚠️\n"
                    "    burst_size: 10\n"
                    "    window_seconds: 1\n\n"
                    "  Previous config (before deploy-ratelimit-01):\n"
                    "    max_requests_per_second: 10000\n"
                    "    burst_size: 1000\n"
                    "    window_seconds: 1\n\n"
                    "  Traffic analysis:\n"
                    "    avg_legitimate_traffic: 1050 req/s\n"
                    "    current_limit: 100 req/s\n"
                    "    throttle_rate: 90.5%\n\n"
                    "  💡 Root cause: deploy-ratelimit-01 set rate limit to 100\n"
                    "  instead of 10000 (missing a zero — likely typo).\n"
                    "  Fix: rollback deploy-ratelimit-01 or set limit to 10000.")
                return self._build_observation(result), self._compute_reward("root_cause_identified"), False
            else:
                return self._build_observation(f"Diagnostic ({diag_type}) for {service}:\n  Normal."), self._compute_reward("info_gathered"), False

        elif at == "apply_fix":
            service = params.get("service", "")
            fix_type = params.get("fix_type", "")
            if service == "api-gateway" and fix_type in ("rollback", "rollback_config",
                    "restore_rate_limit", "set_rate_limit_10000"):
                self._correct_fix_applied = True
                self._fix_applied = True
                self._service_fixed = True
                self.incident_phase = IncidentPhase.VERIFYING
                result = ("Fix applied: api-gateway rate limit restored.\n"
                    "  max_requests_per_second: 100 → 10000\n"
                    "  burst_size: 10 → 1000\n"
                    "  Throttled traffic cleared immediately.\n"
                    "  All downstream services receiving normal traffic.")
                return self._build_observation(result), self._compute_reward("fix_applied_correctly"), False
            else:
                self._wrong_fix_count += 1
                return self._build_observation(f"Fix {fix_type} on {service} — no effect on rate limiting."), self._compute_reward("no_effect"), False

        elif at == "verify_health":
            if self._service_fixed:
                self._resolution_verified = True
                self.incident_phase = IncidentPhase.RESOLVED
                result = ("Health check: ALL SERVICES HEALTHY\n"
                    "  api-gateway: error_rate=0.01, throttled=0, requests_per_sec=1050\n"
                    "Incident resolved.")
                return self._build_observation(result), self._compute_reward("resolution_verified"), True
            else:
                return self._build_observation("Health check: ISSUES PERSIST\n  api-gateway still throttling 90% of traffic."), self._compute_reward("no_effect"), False

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
            result = "Hint: api-gateway rate limit was changed by a recent deploy. Check if 100 should be 10000."
            return self._build_observation(result), self._compute_reward("escalate_used"), False

        else:
            return self._build_observation(f"Unknown action type: {at}"), self._compute_reward("no_effect"), False

    def get_grader_score(self) -> float:
        score = 0.0
        score += self._throttling_noticed * 0.12
        score += self._deploy_identified * 0.15
        score += self._rate_limit_misconfigured * 0.18
        score += self._correct_fix_applied * 0.25
        score += self._resolution_verified * 0.13
        score += self._postmortem_written * 0.08
        time_bonus = max(0, 1 - (self.step_count / self.max_steps)) * 0.05
        score += time_bonus
        score += self._evidence_breadth_score()
        score += self._postmortem_quality_bonus(["rate", "limit", "429", "throttl", "10000", "100", "deploy"])
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
