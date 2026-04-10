"""
Task 7: Kafka Consumer Lag — Medium difficulty.

Root cause: A deploy to order-service changed Kafka consumer
session.timeout.ms from 30000 → 3000ms, causing constant
consumer group rebalances. Orders queue up and processing
delays cascade to users.
"""
from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.scenarios.base import BaseScenario
from app.data.log_templates import LogGenerator
from app.data.metrics import MetricsSimulator
from app.data.deploy_history import DeployHistory


class KafkaLagScenario(BaseScenario):
    task_id = "task7_kafka_lag"
    max_steps = 25

    def _correct_severity(self) -> str:
        return "SEV2"  # degraded order processing, not full outage

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self._consumer_lag_noticed = False
        self._deploy_identified = False
        self._session_timeout_identified = False
        self._correct_fix_applied = False
        self._order_service_fixed = False
        self._wrong_fix_count = 0

    def _get_alert_summary(self) -> str:
        if self._order_service_fixed:
            return "RESOLVED: order-service Kafka consumer lag cleared."
        return ("WARN: order-service processing delays increasing. "
                "Order completion time p99 went from 2s to 45s. "
                "Kafka consumer lag growing. Started ~10 minutes ago.")

    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        statuses = {}
        for svc in self.graph.get_all_services():
            if self._order_service_fixed:
                statuses[svc] = ServiceStatus(
                    name=svc, status="healthy", error_rate=0.01,
                    latency_p99_ms=45, restarts_last_hour=0)
            elif svc == "order-service":
                statuses[svc] = ServiceStatus(
                    name=svc, status="degraded", error_rate=0.15,
                    latency_p99_ms=45000, restarts_last_hour=0)
            elif svc == "api-gateway":
                statuses[svc] = ServiceStatus(
                    name=svc, status="degraded", error_rate=0.12,
                    latency_p99_ms=46000, restarts_last_hour=0)
            else:
                statuses[svc] = ServiceStatus(
                    name=svc, status="healthy", error_rate=0.01,
                    latency_p99_ms=45, restarts_last_hour=0)
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
            if service == "order-service":
                result = (f"Logs for {service} (last 20 entries):\n"
                    "[2026-03-26T02:50:12Z] [WARN]  Kafka consumer rebalance triggered (reason: member heartbeat timeout) [service=order-service]\n"
                    "[2026-03-26T02:50:15Z] [ERROR] Consumer group rebalance #47 in last 10 minutes — processing stalled [service=order-service]\n"
                    "[2026-03-26T02:50:18Z] [WARN]  Consumer lag on orders-topic partition 0: 12847 messages behind [service=order-service]\n"
                    "[2026-03-26T02:50:22Z] [ERROR] Order ORD-8821 processing delayed 42s (normally <2s) [service=order-service]\n"
                    "[2026-03-26T02:50:25Z] [WARN]  session.timeout.ms=3000 — consumer kicked from group before heartbeat [service=order-service]\n"
                    "[2026-03-26T02:50:30Z] [ERROR] Kafka ConsumerCoordinator: member removed from group, rejoining [service=order-service]\n"
                    "[2026-03-26T02:50:35Z] [WARN]  Partition assignment revoked and re-assigned (rebalance loop) [service=order-service]\n"
                    "[2026-03-26T02:50:40Z] [INFO]  Processing resumed briefly — 230 messages processed before next rebalance [service=order-service]")
                self._consumer_lag_noticed = True
                reward = self._compute_reward("root_cause_progress")
                return self._build_observation(result), reward, False
            else:
                logs = LogGenerator.generate(self.task_id, service, self.seed, step_count=self.step_count)
                result = f"Logs for {service} (last 20 entries):\n" + "\n".join(logs[:10])
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False

        elif at == "check_metrics":
            service = params.get("service", "")
            if service == "order-service":
                self._consumer_lag_noticed = True
                result = (f"Metrics for {service}:\n"
                    "  error_rate: 0.15\n"
                    "  latency_p99_ms: 45000\n"
                    "  cpu_percent: 35.2\n"
                    "  memory_percent: 42.1\n"
                    "  kafka_consumer_lag: 12847 ⚠️ CRITICAL\n"
                    "  kafka_rebalance_count_10m: 47 ⚠️ CRITICAL\n"
                    "  kafka_session_timeout_ms: 3000\n"
                    "  orders_processed_per_sec: 12 (normally 450)")
                reward = self._compute_reward("root_cause_progress")
                return self._build_observation(result), reward, False
            else:
                summary = MetricsSimulator.get_metrics_summary(service, self.task_id, self.seed)
                result = f"Metrics for {service}:\n"
                for k, v in summary.items():
                    if k != "service": result += f"  {k}: {v}\n"
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False

        elif at == "check_alerts":
            result = ("Active alerts:\n"
                "  [WARN] order-service: Kafka consumer lag 12847 messages on orders-topic\n"
                "  [WARN] order-service: 47 consumer rebalances in last 10 minutes\n"
                "  [WARN] api-gateway: order completion latency p99=45s (SLA: 5s)\n"
                "  [INFO] Consumer lag started growing ~10 minutes ago")
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "check_deployments":
            service_filter = params.get("service")
            deploys = DeployHistory.get_deploys(self.seed, last_n=5, service_filter=service_filter)
            result = "Recent deployments:\n"
            for d in deploys:
                result += f"  [{d.deploy_id}] {d.service} — {d.description} ({d.minutes_ago}min ago)\n"
            result += ("\n  [deploy-kafka-01] order-service — 'Kafka consumer tuning: reduced session.timeout.ms' (10min ago) ⚠️\n"
                       "    Changed: kafka.session.timeout.ms: 30000 → 3000\n"
                       "    Changed: kafka.max.poll.interval.ms: 300000 → 300000 (unchanged)")
            self._deploy_identified = True
            reward = self._compute_reward("root_cause_progress")
            return self._build_observation(result), reward, False

        elif at == "check_dependencies":
            service = params.get("service", "")
            upstream = self.graph.get_upstream_of(service)
            downstream = self.graph.get_downstream(service)
            result = (f"Dependencies for {service}:\n"
                      f"  Upstream (callers): {upstream}\n"
                      f"  Downstream (calls): {downstream}")
            if service == "order-service":
                result += ("\n\n  ⚠️  order-service also consumes from Kafka topic 'orders-topic'\n"
                           "  Consumer group: order-processors (3 instances)")
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "run_diagnostic":
            service = params.get("service", "")
            diag_type = params.get("type", "general")
            if service == "order-service" and diag_type in ("kafka", "consumer", "general"):
                self._session_timeout_identified = True
                result = (f"Diagnostic (kafka) for {service}:\n"
                    "  consumer_group: order-processors\n"
                    "  partition_count: 6\n"
                    "  consumer_instances: 3\n"
                    "  session_timeout_ms: 3000 ⚠️ TOO LOW (was 30000)\n"
                    "  heartbeat_interval_ms: 1000\n"
                    "  max_poll_interval_ms: 300000\n"
                    "  rebalance_count_10m: 47 ⚠️\n"
                    "  avg_poll_duration_ms: 4200 > session_timeout 3000 ⚠️\n\n"
                    "  💡 Root cause: session.timeout.ms (3000ms) is shorter than\n"
                    "  avg poll duration (4200ms). Consumer gets kicked from group\n"
                    "  before it finishes processing, causing constant rebalances.\n"
                    "  Fix: rollback deploy-kafka-01 to restore session.timeout.ms=30000")
                reward = self._compute_reward("root_cause_identified")
                return self._build_observation(result), reward, False
            else:
                result = f"Diagnostic ({diag_type}) for {service}:\n  All metrics within normal range."
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False

        elif at == "apply_fix":
            service = params.get("service", "")
            fix_type = params.get("fix_type", "")
            if service == "order-service" and fix_type in ("rollback", "rollback_config",
                    "increase_session_timeout", "restore_session_timeout"):
                self._correct_fix_applied = True
                self._fix_applied = True
                self._order_service_fixed = True
                self.incident_phase = IncidentPhase.VERIFYING
                result = ("Fix applied: order-service Kafka config restored.\n"
                    "  session.timeout.ms: 3000 → 30000 (restored)\n"
                    "  Consumer group stabilized — no more rebalances.\n"
                    "  Consumer lag dropping rapidly.\n"
                    "  Verify health to confirm resolution.")
                reward = self._compute_reward("fix_applied_correctly")
                return self._build_observation(result), reward, False
            elif fix_type == "restart":
                result = (f"Restart applied to {service}. Consumer briefly caught up\n"
                    "but rebalance loop resumed — same config loaded on restart.")
                reward = self._compute_reward("wrong_fix_applied")
                return self._build_observation(result), reward, False
            else:
                self._wrong_fix_count += 1
                result = f"Fix {fix_type} applied to {service} — no effect on consumer lag."
                reward = self._compute_reward("no_effect")
                return self._build_observation(result), reward, False

        elif at == "verify_health":
            if self._order_service_fixed:
                self._resolution_verified = True
                self.incident_phase = IncidentPhase.RESOLVED
                result = ("Health check: ALL SERVICES HEALTHY\n"
                    "  order-service: consumer_lag=0, rebalances=0, latency_p99=1.8s\n"
                    "  api-gateway: error_rate=0.01, healthy\n"
                    "Incident resolved.")
                reward = self._compute_reward("resolution_verified")
                return self._build_observation(result), reward, True
            else:
                result = ("Health check: ISSUES PERSIST\n"
                    "  order-service: consumer_lag=12847, rebalances ongoing\n"
                    "Root cause not addressed.")
                reward = self._compute_reward("no_effect")
                return self._build_observation(result), reward, False

        elif at == "write_postmortem":
            content = params.get("content", "")
            if len(content) > 50:
                self._postmortem_written = True
                result = "Postmortem recorded successfully."
                reward = self._compute_reward("postmortem_written")
            else:
                result = "Postmortem too short."
                reward = self._compute_reward("no_effect")
            done = self.incident_phase == IncidentPhase.RESOLVED
            return self._build_observation(result), reward, done

        elif at == "escalate":
            self.hints_used += 1
            result = ("Escalation hint: order-service Kafka consumer is in a rebalance loop.\n"
                "Check recent deploys for Kafka config changes — session.timeout.ms may be too low.")
            reward = self._compute_reward("escalate_used")
            return self._build_observation(result), reward, False

        else:
            result = f"Unknown action type: {at}"
            reward = self._compute_reward("no_effect")
            return self._build_observation(result), reward, False

    def get_grader_score(self) -> float:
        score = 0.0
        score += self._consumer_lag_noticed * 0.15
        score += self._deploy_identified * 0.15
        score += self._session_timeout_identified * 0.15
        score += self._correct_fix_applied * 0.25
        score += self._resolution_verified * 0.13
        score += self._postmortem_written * 0.08
        time_bonus = max(0, 1 - (self.step_count / self.max_steps)) * 0.05
        score += time_bonus
        score += self._evidence_breadth_score()
        score += self._postmortem_quality_bonus(
            ["kafka", "consumer", "rebalance", "session.timeout", "3000", "lag"])
        score += self._severity_correct * 0.02
        score += (self._status_page_updated and self._status_page_before_fix) * 0.02
        if self._fix_applied and self._fix_before_any_breach:
            score += 0.02
        score -= self._wrong_fix_count * 0.05
        score -= self.hints_used * 0.05
        return round(min(0.999, max(0.001, score)), 4)
