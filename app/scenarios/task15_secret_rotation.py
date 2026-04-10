"""
Task 15: Secret Rotation Failure — Medium difficulty.

Root cause: payment-service external API key was rotated in the vault,
but the service was not restarted/reloaded to pick up the new key.
All payment API calls to the external gateway return 401 Unauthorized.
"""
from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.scenarios.base import BaseScenario
from app.data.log_templates import LogGenerator
from app.data.metrics import MetricsSimulator
from app.data.deploy_history import DeployHistory


class SecretRotationScenario(BaseScenario):
    task_id = "task15_secret_rotation"
    max_steps = 25

    def _correct_severity(self) -> str:
        return "SEV1"

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self._payment_failure_noticed = False
        self._auth_401_identified = False
        self._stale_key_identified = False
        self._correct_fix_applied = False
        self._service_fixed = False
        self._wrong_fix_count = 0

    def _get_alert_summary(self) -> str:
        if self._service_fixed:
            return "RESOLVED: payment-service API key refreshed, payments processing normally."
        return ("CRIT: payment-service returning 100% errors. All payment transactions "
                "failing with 'unauthorized' from external payment gateway. "
                "Started ~6 minutes ago after scheduled secret rotation.")

    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        statuses = {}
        for svc in self.graph.get_all_services():
            if self._service_fixed:
                statuses[svc] = ServiceStatus(name=svc, status="healthy", error_rate=0.01, latency_p99_ms=50, restarts_last_hour=1)
            elif svc == "payment-service":
                statuses[svc] = ServiceStatus(name=svc, status="down", error_rate=1.0, latency_p99_ms=200, restarts_last_hour=0)
            elif svc == "order-service":
                statuses[svc] = ServiceStatus(name=svc, status="degraded", error_rate=0.45, latency_p99_ms=5000, restarts_last_hour=0)
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
            if service == "payment-service":
                self._payment_failure_noticed = True
                self._auth_401_identified = True
                result = (f"Logs for {service}:\n"
                    "[2026-03-26T10:00:05Z] [ERROR] External payment gateway returned 401 Unauthorized [service=payment-service]\n"
                    "[2026-03-26T10:00:05Z] [ERROR] API key validation failed: 'Invalid API key' from gateway.payments.com [service=payment-service]\n"
                    "[2026-03-26T10:00:08Z] [ERROR] Payment TXN-0012 failed: external gateway auth error [service=payment-service]\n"
                    "[2026-03-26T10:00:10Z] [WARN]  Using API key loaded at startup: key_prefix=sk_live_A3x... (loaded 2 hours ago) [service=payment-service]\n"
                    "[2026-03-26T10:00:12Z] [ERROR] 100% payment failure rate — all transactions returning 401 [service=payment-service]\n"
                    "[2026-03-26T10:00:15Z] [INFO]  Vault secret version: v2 (current in vault: v3) ⚠️ key mismatch [service=payment-service]\n"
                    "[2026-03-26T10:00:18Z] [WARN]  Service has not reloaded secrets since startup — no hot-reload configured [service=payment-service]")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            elif service == "order-service":
                self._payment_failure_noticed = True
                result = (f"Logs for {service}:\n"
                    "[2026-03-26T10:00:08Z] [ERROR] Payment step failed for ORD-2345: payment-service returned 'unauthorized' [service=order-service]\n"
                    "[2026-03-26T10:00:12Z] [WARN]  All orders with payment failing — 450 failures in 6 minutes [service=order-service]")
                return self._build_observation(result), self._compute_reward("info_gathered"), False
            else:
                logs = LogGenerator.generate(self.task_id, service, self.seed, step_count=self.step_count)
                return self._build_observation(f"Logs for {service}:\n" + "\n".join(logs[:10])), self._compute_reward("info_gathered"), False

        elif at == "check_metrics":
            service = params.get("service", "")
            if service == "payment-service":
                self._payment_failure_noticed = True
                result = (f"Metrics for {service}:\n"
                    "  error_rate: 1.0 ⚠️ CRITICAL\n  latency_p99_ms: 200 (fast failure)\n"
                    "  payment_success_rate: 0.0 ⚠️\n"
                    "  external_gateway_401_rate: 1.0 ⚠️\n"
                    "  vault_secret_version_loaded: v2\n"
                    "  vault_secret_version_current: v3 ⚠️ MISMATCH\n"
                    "  last_secret_reload: 2 hours ago\n"
                    "  cpu_percent: 20\n  memory_percent: 35")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            else:
                summary = MetricsSimulator.get_metrics_summary(service, self.task_id, self.seed)
                result = f"Metrics for {service}:\n"
                for k, v in summary.items():
                    if k != "service": result += f"  {k}: {v}\n"
                return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_alerts":
            result = ("Active alerts:\n"
                "  [CRIT] payment-service: 100% payment failure rate\n"
                "  [CRIT] payment-service: external gateway returning 401 for all requests\n"
                "  [WARN] Vault: secret 'payment-api-key' rotated 6 minutes ago (v2→v3)\n"
                "  [WARN] order-service: order completion impacted by payment failures")
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_deployments":
            service_filter = params.get("service")
            deploys = DeployHistory.get_deploys(self.seed, last_n=5, service_filter=service_filter)
            result = "Recent deployments:\n"
            for d in deploys:
                result += f"  [{d.deploy_id}] {d.service} — {d.description} ({d.minutes_ago}min ago)\n"
            result += ("\n  [vault-rotation-01] vault — 'Scheduled API key rotation: payment-api-key v2→v3' (6min ago) ⚠️\n"
                       "    Note: payment-service does NOT have hot-reload for vault secrets\n"
                       "    Requires: restart or SIGHUP to reload new key")
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_dependencies":
            service = params.get("service", "")
            upstream = self.graph.get_upstream_of(service)
            downstream = self.graph.get_downstream(service)
            result = f"Dependencies for {service}:\n  Upstream: {upstream}\n  Downstream: {downstream}"
            if service == "payment-service":
                result += "\n\n  External dependency: gateway.payments.com (payment API, requires API key)"
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "run_diagnostic":
            service = params.get("service", "")
            diag_type = params.get("type", "general")
            if service == "payment-service" and diag_type in ("secrets", "vault", "auth", "api_key", "general"):
                self._stale_key_identified = True
                result = (f"Diagnostic (secrets) for {service}:\n"
                    "  Vault secret: payment-api-key\n"
                    "    Version loaded by service: v2 (sk_live_A3x...)\n"
                    "    Current version in Vault: v3 (sk_live_K9m...) ⚠️ MISMATCH\n"
                    "    Rotation time: 6 minutes ago\n\n"
                    "  Secret loading:\n"
                    "    Method: environment variable at startup\n"
                    "    Hot-reload: NOT CONFIGURED ⚠️\n"
                    "    Last reload: 2 hours ago (at last deployment)\n\n"
                    "  External gateway test:\n"
                    "    With v2 key: 401 Unauthorized ✗\n"
                    "    With v3 key: 200 OK ✓\n\n"
                    "  💡 Root cause: Vault rotated payment-api-key from v2→v3,\n"
                    "  but payment-service loaded the key at startup and has no hot-reload.\n"
                    "  Old key (v2) was revoked, all API calls fail with 401.\n"
                    "  Fix: restart payment-service to pick up new key (v3).")
                return self._build_observation(result), self._compute_reward("root_cause_identified"), False
            else:
                return self._build_observation(f"Diagnostic ({diag_type}) for {service}:\n  Normal."), self._compute_reward("info_gathered"), False

        elif at == "apply_fix":
            service = params.get("service", "")
            fix_type = params.get("fix_type", "")
            if service == "payment-service" and fix_type in ("restart", "reload_secrets",
                    "rolling_restart", "refresh_api_key", "reload_vault_secret"):
                self._correct_fix_applied = True
                self._fix_applied = True
                self._service_fixed = True
                self.incident_phase = IncidentPhase.VERIFYING
                result = ("Fix applied: payment-service restarted with new secret.\n"
                    "  Loaded vault secret version: v3 (sk_live_K9m...)\n"
                    "  External gateway auth: 200 OK ✓\n"
                    "  Payment processing resuming.\n"
                    "  Verify health to confirm.")
                return self._build_observation(result), self._compute_reward("fix_applied_correctly"), False
            elif fix_type == "rollback":
                result = "Rollback attempted — no deploy to rollback. Issue is secret rotation, not a code change."
                return self._build_observation(result), self._compute_reward("wrong_fix_applied"), False
            else:
                self._wrong_fix_count += 1
                return self._build_observation(f"Fix {fix_type} on {service} — payments still failing with 401."), self._compute_reward("no_effect"), False

        elif at == "verify_health":
            if self._service_fixed:
                self._resolution_verified = True
                self.incident_phase = IncidentPhase.RESOLVED
                result = ("Health check: ALL SERVICES HEALTHY\n"
                    "  payment-service: error_rate=0.01, payment_success_rate=99.5%\n"
                    "  vault_secret_version: v3 (current)\n"
                    "Incident resolved.")
                return self._build_observation(result), self._compute_reward("resolution_verified"), True
            else:
                return self._build_observation("Health check: ISSUES PERSIST\n  payment-service: 100% 401 errors from gateway."), self._compute_reward("no_effect"), False

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
            result = "Hint: payment-service API key was rotated in Vault but service wasn't restarted. Restart to pick up new key."
            return self._build_observation(result), self._compute_reward("escalate_used"), False

        else:
            return self._build_observation(f"Unknown action type: {at}"), self._compute_reward("no_effect"), False

    def get_grader_score(self) -> float:
        score = 0.0
        score += self._payment_failure_noticed * 0.12
        score += self._auth_401_identified * 0.15
        score += self._stale_key_identified * 0.18
        score += self._correct_fix_applied * 0.25
        score += self._resolution_verified * 0.13
        score += self._postmortem_written * 0.08
        time_bonus = max(0, 1 - (self.step_count / self.max_steps)) * 0.05
        score += time_bonus
        score += self._evidence_breadth_score()
        score += self._postmortem_quality_bonus(["secret", "rotation", "vault", "api key", "401", "reload", "restart"])
        score += self._severity_correct * 0.02
        score += (self._status_page_updated and self._status_page_before_fix) * 0.02
        if self._fix_applied and self._fix_before_any_breach:
            score += 0.02
        score -= self._wrong_fix_count * 0.05
        score -= self.hints_used * 0.05
        return round(min(0.999, max(0.001, score)), 4)
