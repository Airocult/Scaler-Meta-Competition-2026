"""
Task 4: DNS Resolution Failure — Easy-Medium difficulty.

Root cause: auth-service DNS resolver cache is stale/corrupted,
causing lookups for user-db to fail intermittently.
auth-service → user-db DNS fails → auth failures cascade to api-gateway.

Agent must: check DNS config, identify stale DNS cache on auth-service,
flush DNS cache or restart DNS resolver, and verify resolution.
"""
from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.scenarios.base import BaseScenario
from app.data.log_templates import LogGenerator
from app.data.metrics import MetricsSimulator


class DNSFailureScenario(BaseScenario):
    task_id = "task4_dns_failure"
    max_steps = 25

    def _correct_severity(self) -> str:
        return "SEV2"  # auth degraded, users can't log in

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self._dns_issue_noticed = False
        self._auth_service_investigated = False
        self._dns_cache_identified = False
        self._correct_fix_applied = False
        self._auth_fixed = False
        self._wrong_fix_count = 0
        self._root_cause_service = "auth-service"

    def _get_alert_summary(self) -> str:
        if self._auth_fixed:
            return "RESOLVED: auth-service connectivity restored."
        return ("CRITICAL: auth-service returning 503 for 40% of requests. "
                "Users unable to log in. Started 8 minutes ago.")

    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        statuses = {}
        for svc in self.graph.get_all_services():
            if self._auth_fixed:
                statuses[svc] = ServiceStatus(
                    name=svc, status="healthy", error_rate=0.01,
                    latency_p99_ms=45, restarts_last_hour=0,
                )
            elif svc == "auth-service":
                statuses[svc] = ServiceStatus(
                    name=svc, status="degraded", error_rate=0.42,
                    latency_p99_ms=8500, restarts_last_hour=0,
                )
            elif svc == "user-db":
                statuses[svc] = ServiceStatus(
                    name=svc, status="healthy", error_rate=0.01,
                    latency_p99_ms=25, restarts_last_hour=0,
                )
            elif svc == "api-gateway":
                statuses[svc] = ServiceStatus(
                    name=svc, status="degraded", error_rate=0.30,
                    latency_p99_ms=6000, restarts_last_hour=0,
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
            if service == "auth-service":
                self._auth_service_investigated = True
                result = (f"Logs for {service} (last 20 entries):\n"
                          "[2026-03-26T02:52:14.331Z] [ERROR] DNS resolution failed for user-db.internal: NXDOMAIN [service=auth-service]\n"
                          "[2026-03-26T02:52:18.442Z] [ERROR] getaddrinfo ENOTFOUND user-db.internal [service=auth-service]\n"
                          "[2026-03-26T02:52:22.119Z] [WARN]  DNS cache stale — TTL expired 480s ago for user-db.internal [service=auth-service]\n"
                          "[2026-03-26T02:52:25.553Z] [ERROR] Connection refused to user-db: DNS resolution returned stale/invalid IP 10.0.0.99 [service=auth-service]\n"
                          "[2026-03-26T02:52:30.667Z] [ERROR] Auth request failed: upstream user-db unreachable [service=auth-service]\n"
                          "[2026-03-26T02:52:35.112Z] [INFO]  Retrying DNS lookup for user-db.internal (attempt 2/3) [service=auth-service]\n"
                          "[2026-03-26T02:52:38.998Z] [ERROR] DNS retry failed — resolver 10.96.0.10 not responding [service=auth-service]\n"
                          "[2026-03-26T02:52:42.001Z] [WARN]  Falling back to cached DNS entry (stale) [service=auth-service]\n"
                          "[2026-03-26T02:52:45.334Z] [ERROR] Connection timeout to 10.0.0.99:5432 (stale IP) [service=auth-service]\n"
                          "[2026-03-26T02:52:50.221Z] [ERROR] HTTP 503 returned to client — auth unavailable [service=auth-service]")
                reward = self._compute_reward("root_cause_progress")
                return self._build_observation(result), reward, False
            elif service == "user-db":
                result = (f"Logs for {service} (last 20 entries):\n"
                          "[2026-03-26T02:52:10.100Z] [INFO]  Query executed successfully in 12ms [service=user-db]\n"
                          "[2026-03-26T02:52:15.200Z] [INFO]  Health check passed — all subsystems OK [service=user-db]\n"
                          "[2026-03-26T02:52:20.300Z] [INFO]  Connection pool: 3/20 active connections [service=user-db]\n"
                          "[2026-03-26T02:52:25.400Z] [INFO]  Replication lag: 0ms [service=user-db]\n"
                          "[2026-03-26T02:52:30.500Z] [INFO]  No errors detected [service=user-db]")
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False
            else:
                logs = LogGenerator.generate(self.task_id, service, self.seed, step_count=self.step_count)
                result = f"Logs for {service} (last 20 entries):\n" + "\n".join(logs[:10])
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False

        elif at == "check_metrics":
            service = params.get("service", "")
            if service == "auth-service":
                self._auth_service_investigated = True
                result = (f"Metrics for {service}:\n"
                          "  error_rate: 0.42\n"
                          "  latency_p99_ms: 8500\n"
                          "  cpu_percent: 22.3\n"
                          "  memory_percent: 35.1\n"
                          "  connection_pool_usage: 0.15\n"
                          "  dns_resolution_failures: 847 (last 10 min) ⚠️ CRITICAL\n"
                          "  dns_cache_hit_rate: 0.12 (normally >0.95) ⚠️")
                reward = self._compute_reward("root_cause_progress")
                return self._build_observation(result), reward, False
            else:
                summary = MetricsSimulator.get_metrics_summary(service, self.task_id, self.seed)
                result = f"Metrics for {service}:\n"
                for k, v in summary.items():
                    if k != "service":
                        result += f"  {k}: {v}\n"
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False

        elif at == "check_alerts":
            result = ("Active alerts:\n"
                      "  [CRITICAL] auth-service: 42% error rate — DNS resolution failures\n"
                      "  [WARN] api-gateway: elevated error rate 30% — upstream auth failures\n"
                      "  [INFO] user-db: healthy — accepting connections normally\n"
                      "  [INFO] DNS resolution failures started ~8 minutes ago")
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
            result += "\n  No recent deployments to auth-service or DNS infrastructure."
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "check_dependencies":
            service = params.get("service", "")
            upstream = self.graph.get_upstream_of(service)
            downstream = self.graph.get_downstream(service)
            result = (f"Dependencies for {service}:\n"
                      f"  Upstream (callers): {upstream}\n"
                      f"  Downstream (calls): {downstream}")
            if service == "auth-service":
                result += ("\n\n  ⚠️  auth-service depends on user-db via DNS name 'user-db.internal'\n"
                           "  DNS resolver: kube-dns at 10.96.0.10")
                self._auth_service_investigated = True
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "run_diagnostic":
            service = params.get("service", "")
            diag_type = params.get("type", "general")

            if service == "auth-service" and diag_type in ("dns", "dns_check", "network", "general"):
                self._dns_cache_identified = True
                self._dns_issue_noticed = True
                result = (f"Diagnostic (dns) for {service}:\n"
                          "  dns_resolver: 10.96.0.10 (kube-dns)\n"
                          "  dns_cache_entries: 847 (stale)\n"
                          "  dns_cache_ttl_expired: True ⚠️\n"
                          "  user-db.internal → 10.0.0.99 (STALE — actual IP is 10.0.1.15)\n"
                          "  dns_resolution_time: timeout (>5000ms)\n"
                          "  kube-dns_status: healthy\n\n"
                          "  💡 Root cause: auth-service local DNS cache has stale entry for user-db.\n"
                          "  The cached IP 10.0.0.99 is no longer valid. user-db moved to 10.0.1.15.\n"
                          "  Fix: flush DNS cache on auth-service or restart DNS resolver.")
                reward = self._compute_reward("root_cause_identified")
                return self._build_observation(result), reward, False
            elif service == "user-db":
                result = (f"Diagnostic ({diag_type}) for {service}:\n"
                          "  All metrics within normal range.\n"
                          "  Database is accepting connections on 10.0.1.15:5432.\n"
                          "  Replication: healthy, lag=0ms")
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False
            else:
                result = f"Diagnostic ({diag_type}) for {service}:\n  All metrics within normal range."
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False

        elif at == "apply_fix":
            service = params.get("service", "")
            fix_type = params.get("fix_type", "")

            if service == "auth-service" and fix_type in ("flush_dns", "flush_dns_cache",
                                                           "restart_dns", "restart",
                                                           "clear_dns_cache"):
                self._correct_fix_applied = True
                self._fix_applied = True
                self._auth_fixed = True
                self.incident_phase = IncidentPhase.VERIFYING
                result = ("Fix applied: auth-service DNS cache flushed.\n"
                          "  DNS resolver re-queried kube-dns for user-db.internal.\n"
                          "  Resolved: user-db.internal → 10.0.1.15 (correct IP)\n"
                          "  auth-service reconnecting to user-db successfully.\n"
                          "  Error rate dropping rapidly.\n"
                          "Verify health to confirm full resolution.")
                reward = self._compute_reward("fix_applied_correctly")
                return self._build_observation(result), reward, False

            elif service == "api-gateway":
                self._wrong_fix_count += 1
                result = ("api-gateway is symptomatic — it's failing because auth-service is down.\n"
                          "Root cause is in the auth-service → user-db connection path.")
                reward = self._compute_reward("wrong_fix_applied")
                return self._build_observation(result), reward, False

            elif service == "user-db":
                self._wrong_fix_count += 1
                result = ("user-db is healthy and accepting connections normally.\n"
                          "The issue is that auth-service can't resolve user-db's DNS name.")
                reward = self._compute_reward("no_effect")
                return self._build_observation(result), reward, False

            else:
                self._wrong_fix_count += 1
                result = f"Fix applied to {service} ({fix_type}) — no observable effect on the incident."
                reward = self._compute_reward("no_effect")
                return self._build_observation(result), reward, False

        elif at == "verify_health":
            if self._auth_fixed:
                self._resolution_verified = True
                self.incident_phase = IncidentPhase.RESOLVED
                result = ("Health check: ALL SERVICES HEALTHY\n"
                          "  auth-service: error_rate=0.01, dns_resolution=OK\n"
                          "  user-db: healthy, connections normal\n"
                          "  api-gateway: error_rate=0.01, healthy\n"
                          "Incident resolved. Consider writing a postmortem.")
                reward = self._compute_reward("resolution_verified")
                return self._build_observation(result), reward, True
            else:
                result = ("Health check: ISSUES PERSIST\n"
                          "  auth-service: error_rate=0.42, DNS resolution failures ongoing\n"
                          "  api-gateway: error_rate=0.30, upstream auth failures\n"
                          "Root cause has not been addressed.")
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
            result = ("Escalation hint: auth-service can't reach user-db.\n"
                      "Check the DNS resolution path — auth-service uses DNS to find user-db.\n"
                      "Look at DNS cache staleness and resolver connectivity.")
            reward = self._compute_reward("escalate_used")
            return self._build_observation(result), reward, False

        else:
            result = f"Unknown action type: {at}"
            reward = self._compute_reward("no_effect")
            return self._build_observation(result), reward, False

    def get_grader_score(self) -> float:
        score = 0.0
        score += self._auth_service_investigated * 0.12
        score += self._dns_issue_noticed * 0.13
        score += self._dns_cache_identified * 0.18
        score += self._correct_fix_applied * 0.23
        score += self._resolution_verified * 0.13
        score += self._postmortem_written * 0.08

        # Time bonus
        time_bonus = max(0, 1 - (self.step_count / self.max_steps)) * 0.05
        score += time_bonus

        # Evidence breadth bonus
        score += self._evidence_breadth_score()

        # Postmortem quality
        score += self._postmortem_quality_bonus(
            ["dns", "cache", "stale", "auth-service", "nxdomain", "10.0.0.99"]
        )

        # Incident communication bonuses
        score += self._severity_correct * 0.02
        score += (self._status_page_updated and self._status_page_before_fix) * 0.02

        # SLO-aware bonus
        if self._fix_applied and self._fix_before_any_breach:
            score += 0.02

        # Wrong fix penalty
        
        # Efficient investigation bonus
        score += self._efficient_investigation_bonus()
        # Blast radius assessment bonus
        score += self._blast_radius_bonus()
        score -= self._wrong_fix_count * 0.05

        # Escalation penalty
        score -= self.hints_used * 0.05

        return round(min(0.999, max(0.001, score)), 4)
