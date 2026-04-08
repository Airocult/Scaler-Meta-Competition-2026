"""
Task 5: Certificate Expiry Chain — Medium-Hard difficulty.

Root cause: TLS certificate on payment-service expired 2 hours ago.
SSL handshake failures cascade: order-service → payment-service fails,
then payment-service → payment-db also failing (mutual TLS).

Agent must: trace the cascading SSL errors, identify the expired cert
on payment-service, renew/replace the certificate, and verify.
"""
from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.scenarios.base import BaseScenario
from app.data.log_templates import LogGenerator
from app.data.metrics import MetricsSimulator


class CertExpiryScenario(BaseScenario):
    task_id = "task5_cert_expiry"
    max_steps = 35

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self._payment_svc_investigated = False
        self._cert_issue_noticed = False
        self._cert_expiry_identified = False
        self._cert_renewed = False
        self._services_reconnected = False
        self._wrong_fix_count = 0
        self._restart_without_cert_fix = 0

    def _get_alert_summary(self) -> str:
        if self._services_reconnected:
            return "RESOLVED: payment-service TLS connectivity restored."
        return ("CRITICAL: order-service receiving SSL handshake failures from payment-service. "
                "Payment processing at 0% success rate. Started 25 minutes ago. "
                "Revenue impact: ~$12,000/min.")

    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        statuses = {}
        for svc in self.graph.get_all_services():
            if self._services_reconnected:
                statuses[svc] = ServiceStatus(
                    name=svc, status="healthy", error_rate=0.01,
                    latency_p99_ms=40, restarts_last_hour=0,
                )
            elif svc == "payment-service":
                statuses[svc] = ServiceStatus(
                    name=svc, status="down", error_rate=0.95,
                    latency_p99_ms=50, restarts_last_hour=2,
                )
            elif svc == "order-service":
                statuses[svc] = ServiceStatus(
                    name=svc, status="degraded", error_rate=0.68,
                    latency_p99_ms=12000, restarts_last_hour=0,
                )
            elif svc == "payment-db":
                statuses[svc] = ServiceStatus(
                    name=svc, status="degraded", error_rate=0.90,
                    latency_p99_ms=50, restarts_last_hour=0,
                )
            elif svc == "api-gateway":
                statuses[svc] = ServiceStatus(
                    name=svc, status="degraded", error_rate=0.45,
                    latency_p99_ms=9000, restarts_last_hour=0,
                )
            else:
                statuses[svc] = ServiceStatus(
                    name=svc, status="healthy", error_rate=0.02,
                    latency_p99_ms=50, restarts_last_hour=0,
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
            if service == "payment-service":
                self._payment_svc_investigated = True
                self._cert_issue_noticed = True
                result = (f"Logs for {service} (last 20 entries):\n"
                          "[2026-03-26T01:30:14Z] [ERROR] TLS handshake failed: certificate has expired [service=payment-service]\n"
                          "[2026-03-26T01:30:18Z] [ERROR] SSL_ERROR_CERTIFICATE_EXPIRED: /CN=payment-service.internal expired at 2026-03-25T23:59:59Z [service=payment-service]\n"
                          "[2026-03-26T01:30:22Z] [WARN]  Incoming TLS connection from order-service rejected: cert expired [service=payment-service]\n"
                          "[2026-03-26T01:30:25Z] [ERROR] mTLS to payment-db failed: client cert expired [service=payment-service]\n"
                          "[2026-03-26T01:30:30Z] [ERROR] Unable to connect to payment-db: SSL handshake failure (code: CERT_HAS_EXPIRED) [service=payment-service]\n"
                          "[2026-03-26T01:30:35Z] [WARN]  Connection pool to payment-db: 0/20 active (all failed SSL) [service=payment-service]\n"
                          "[2026-03-26T01:30:40Z] [ERROR] Payment request from order-service rejected: TLS termination failed [service=payment-service]\n"
                          "[2026-03-26T01:30:45Z] [INFO]  Certificate info: subject=/CN=payment-service.internal, issuer=/CN=internal-ca, not_after=2026-03-25T23:59:59Z [service=payment-service]\n"
                          "[2026-03-26T01:30:50Z] [ERROR] 0 of last 100 payment requests succeeded [service=payment-service]")
                reward = self._compute_reward("root_cause_progress")
                return self._build_observation(result), reward, False

            elif service == "order-service":
                self._payment_svc_investigated = True
                result = (f"Logs for {service} (last 20 entries):\n"
                          "[2026-03-26T01:30:12Z] [ERROR] Failed to connect to payment-service: SSL handshake error [service=order-service]\n"
                          "[2026-03-26T01:30:16Z] [ERROR] javax.net.ssl.SSLHandshakeException: PKIX path validation failed: certificate expired [service=order-service]\n"
                          "[2026-03-26T01:30:20Z] [WARN]  Payment service circuit breaker OPEN — 100% failure rate [service=order-service]\n"
                          "[2026-03-26T01:30:25Z] [ERROR] Order #ORD-9982 failed: payment processing unavailable [service=order-service]\n"
                          "[2026-03-26T01:30:30Z] [ERROR] Order #ORD-9983 failed: payment processing unavailable [service=order-service]\n"
                          "[2026-03-26T01:30:35Z] [INFO]  Retrying payment-service connection in 30s [service=order-service]\n"
                          "[2026-03-26T01:30:40Z] [ERROR] Retry failed: same SSL handshake error [service=order-service]")
                reward = self._compute_reward("root_cause_progress")
                return self._build_observation(result), reward, False

            elif service == "payment-db":
                result = (f"Logs for {service} (last 20 entries):\n"
                          "[2026-03-26T01:30:10Z] [INFO]  Database healthy, 0 connected clients (normally 5-10) [service=payment-db]\n"
                          "[2026-03-26T01:30:15Z] [WARN]  mTLS client authentication: rejected 47 connections — client cert expired [service=payment-db]\n"
                          "[2026-03-26T01:30:20Z] [INFO]  Storage: 45% used, replication: healthy [service=payment-db]")
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False

            else:
                logs = LogGenerator.generate(self.task_id, service, self.seed)
                result = f"Logs for {service} (last 20 entries):\n" + "\n".join(logs[:8])
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False

        elif at == "check_metrics":
            service = params.get("service", "")
            if service == "payment-service":
                self._payment_svc_investigated = True
                result = (f"Metrics for {service}:\n"
                          "  error_rate: 0.95\n"
                          "  latency_p99_ms: 50 (requests rejected immediately)\n"
                          "  cpu_percent: 5.2\n"
                          "  memory_percent: 20.1\n"
                          "  tls_handshake_failures: 1247 (last 30 min) ⚠️ CRITICAL\n"
                          "  tls_cert_days_until_expiry: -0.08 (EXPIRED) ⚠️\n"
                          "  connection_pool_to_payment_db: 0/20 (all SSL failed)\n"
                          "  successful_payments_last_30m: 0")
                reward = self._compute_reward("root_cause_progress")
                return self._build_observation(result), reward, False
            elif service == "order-service":
                result = (f"Metrics for {service}:\n"
                          "  error_rate: 0.68\n"
                          "  latency_p99_ms: 12000\n"
                          "  cpu_percent: 15.3\n"
                          "  memory_percent: 40.2\n"
                          "  payment_circuit_breaker: OPEN\n"
                          "  orders_failed_last_30m: 892")
                reward = self._compute_reward("info_gathered")
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
                      "  [CRITICAL] payment-service: TLS certificate EXPIRED — 95% error rate\n"
                      "  [CRITICAL] order-service: payment circuit breaker OPEN — 68% error rate\n"
                      "  [WARN] payment-db: 0 connected clients — mTLS rejections\n"
                      "  [WARN] api-gateway: 45% error rate — payment-related requests failing\n"
                      "  [INFO] Certificate /CN=payment-service.internal expired 2h ago")
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
            result += ("\n  Note: No cert rotation deploys recorded. "
                       "Last cert rotation for payment-service was 365 days ago (auto-renewal failed).")
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "check_dependencies":
            service = params.get("service", "")
            upstream = self.graph.get_upstream_of(service)
            downstream = self.graph.get_downstream(service)
            result = (f"Dependencies for {service}:\n"
                      f"  Upstream (callers): {upstream}\n"
                      f"  Downstream (calls): {downstream}")
            if service == "payment-service":
                result += ("\n\n  ⚠️  payment-service uses mTLS for all connections.\n"
                           "  Cert: /CN=payment-service.internal, issued by /CN=internal-ca\n"
                           "  Same cert used for: server TLS (from order-service) + client mTLS (to payment-db)")
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "run_diagnostic":
            service = params.get("service", "")
            diag_type = params.get("type", "general")

            if service == "payment-service" and diag_type in ("tls", "cert", "ssl", "certificate", "general"):
                self._cert_expiry_identified = True
                result = (f"Diagnostic (tls) for {service}:\n"
                          "  certificate_subject: /CN=payment-service.internal\n"
                          "  certificate_issuer: /CN=internal-ca\n"
                          "  not_before: 2025-03-26T00:00:00Z\n"
                          "  not_after: 2026-03-25T23:59:59Z  ⚠️ EXPIRED\n"
                          "  current_time: 2026-03-26T02:00:00Z\n"
                          "  time_since_expiry: 2h 0m\n"
                          "  cert_serial: 0x1A2B3C4D\n"
                          "  key_type: RSA-2048\n"
                          "  tls_handshake_success_rate: 0.0%\n"
                          "  auto_renewal_status: FAILED (cert-manager pod OOMKilled 3 days ago)\n\n"
                          "  💡 Root cause: payment-service TLS certificate expired 2 hours ago.\n"
                          "  Auto-renewal failed because cert-manager pod was OOMKilled.\n"
                          "  Fix: renew the certificate manually or restart cert-manager.")
                reward = self._compute_reward("root_cause_identified")
                return self._build_observation(result), reward, False
            else:
                result = f"Diagnostic ({diag_type}) for {service}:\n  All metrics within normal range."
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False

        elif at == "apply_fix":
            service = params.get("service", "")
            fix_type = params.get("fix_type", "")

            if service == "payment-service" and fix_type in ("renew_cert", "renew_certificate",
                                                              "replace_cert", "rotate_cert",
                                                              "update_cert", "install_cert"):
                self._cert_renewed = True
                self._fix_applied = True
                self._root_cause_identified = True
                self._services_reconnected = True
                self.incident_phase = IncidentPhase.VERIFYING
                result = ("Fix applied: payment-service TLS certificate renewed.\n"
                          "  New cert: /CN=payment-service.internal\n"
                          "  not_after: 2027-03-26T00:00:00Z\n"
                          "  TLS handshakes succeeding.\n"
                          "  payment-db mTLS connections re-establishing.\n"
                          "  order-service → payment-service connections recovering.\n"
                          "Verify health to confirm full resolution.")
                reward = self._compute_reward("fix_applied_correctly")
                return self._build_observation(result), reward, False

            elif service == "payment-service" and fix_type in ("restart", "restart_service"):
                self._restart_without_cert_fix += 1
                if not self._cert_renewed:
                    result = ("payment-service restarted, but TLS handshakes still failing.\n"
                              "  The expired certificate is still loaded — restart doesn't fix it.\n"
                              "  You need to renew/replace the certificate.")
                    reward = self._compute_reward("wrong_fix_applied")
                else:
                    result = "payment-service restarted. Cert already renewed — connections healthy."
                    reward = self._compute_reward("no_effect")
                return self._build_observation(result), reward, False

            elif service == "order-service":
                self._wrong_fix_count += 1
                result = ("order-service is a victim — it's failing because payment-service has an expired cert.\n"
                          "Fix must be applied to payment-service's TLS certificate.")
                reward = self._compute_reward("wrong_fix_applied")
                return self._build_observation(result), reward, False

            else:
                self._wrong_fix_count += 1
                result = f"Fix applied to {service} ({fix_type}) — no effect on the TLS issue."
                reward = self._compute_reward("no_effect")
                return self._build_observation(result), reward, False

        elif at == "verify_health":
            if self._services_reconnected:
                self._resolution_verified = True
                self.incident_phase = IncidentPhase.RESOLVED
                result = ("Health check: ALL SERVICES HEALTHY\n"
                          "  payment-service: TLS OK, error_rate=0.01, cert valid until 2027-03-26\n"
                          "  order-service: circuit breaker CLOSED, error_rate=0.02\n"
                          "  payment-db: 8 connected clients, healthy\n"
                          "  api-gateway: error_rate=0.01\n"
                          "Incident resolved. Consider writing a postmortem.")
                reward = self._compute_reward("resolution_verified")
                return self._build_observation(result), reward, True
            else:
                result = ("Health check: ISSUES PERSIST\n"
                          "  payment-service: TLS handshake failures — cert still expired\n"
                          "  order-service: circuit breaker OPEN\n"
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
            result = ("Escalation hint: Look at TLS certificates in the payment path.\n"
                      "payment-service uses mTLS for both incoming (from order-service) and "
                      "outgoing (to payment-db) connections. Check certificate expiry dates.")
            reward = self._compute_reward("escalate_used")
            return self._build_observation(result), reward, False

        else:
            result = f"Unknown action type: {at}"
            reward = self._compute_reward("no_effect")
            return self._build_observation(result), reward, False

    def get_grader_score(self) -> float:
        score = 0.0
        score += self._payment_svc_investigated * 0.10
        score += self._cert_issue_noticed * 0.10
        score += self._cert_expiry_identified * 0.20
        score += self._cert_renewed * 0.30
        score += self._resolution_verified * 0.15
        score += self._postmortem_written * 0.10

        # Time bonus
        time_bonus = max(0, 1 - (self.step_count / self.max_steps)) * 0.05
        score += time_bonus

        # Penalties
        score -= self._wrong_fix_count * 0.05
        score -= self._restart_without_cert_fix * 0.05
        score -= self.hints_used * 0.05

        epsilon = 1e-4
        return round(min(1 - epsilon, max(epsilon, score)), 4)
