"""
Task 13: Kubernetes Pod Eviction Storm — Hard difficulty.

Root cause: A batch processing job was deployed to the cluster without
resource limits. It consumes all available node memory, triggering
Kubernetes to evict payment-service pods due to memory pressure.
"""
from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.scenarios.base import BaseScenario
from app.data.log_templates import LogGenerator
from app.data.metrics import MetricsSimulator
from app.data.deploy_history import DeployHistory


class PodEvictionScenario(BaseScenario):
    task_id = "task13_pod_eviction"
    max_steps = 35

    def _correct_severity(self) -> str:
        return "SEV1"

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self._payment_down_noticed = False
        self._eviction_detected = False
        self._node_pressure_identified = False
        self._batch_job_identified = False
        self._correct_fix_applied = False
        self._service_fixed = False
        self._wrong_fix_count = 0

    def _get_alert_summary(self) -> str:
        if self._service_fixed:
            return "RESOLVED: payment-service pods rescheduled, payment processing restored."
        return ("CRIT: payment-service pods keep getting evicted. 0/3 replicas running. "
                "All payment transactions failing. Node memory pressure events detected. "
                "Started ~10 minutes ago.")

    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        statuses = {}
        for svc in self.graph.get_all_services():
            if self._service_fixed:
                statuses[svc] = ServiceStatus(name=svc, status="healthy", error_rate=0.01, latency_p99_ms=50, restarts_last_hour=0)
            elif svc == "payment-service":
                statuses[svc] = ServiceStatus(name=svc, status="down", error_rate=1.0, latency_p99_ms=0, restarts_last_hour=8)
            elif svc == "order-service":
                statuses[svc] = ServiceStatus(name=svc, status="degraded", error_rate=0.40, latency_p99_ms=5000, restarts_last_hour=0)
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
                self._payment_down_noticed = True
                self._eviction_detected = True
                result = (f"Logs for {service}:\n"
                    "[2026-03-26T08:00:05Z] [ERROR] Pod payment-svc-7b8c9-abc12 evicted: The node was low on resource: memory [service=payment-service]\n"
                    "[2026-03-26T08:00:08Z] [INFO]  Pod rescheduled on node-03 — starting up [service=payment-service]\n"
                    "[2026-03-26T08:01:15Z] [ERROR] Pod payment-svc-7b8c9-abc12 evicted AGAIN: memory pressure on node-03 [service=payment-service]\n"
                    "[2026-03-26T08:03:00Z] [ERROR] All 3 replicas evicted — 0/3 pods running [service=payment-service]\n"
                    "[2026-03-26T08:03:05Z] [WARN]  Kubernetes events: EvictionThresholdMet on node-01, node-02, node-03 [service=payment-service]\n"
                    "[2026-03-26T08:05:00Z] [ERROR] Pod pending: Insufficient memory on all available nodes [service=payment-service]\n"
                    "[2026-03-26T08:05:15Z] [WARN]  payment-service has BestEffort QoS (no resource requests/limits) — evicted first ⚠️ [service=payment-service]")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            elif service == "order-service":
                result = (f"Logs for {service}:\n"
                    "[2026-03-26T08:00:10Z] [ERROR] Connection refused to payment-service: no healthy endpoints [service=order-service]\n"
                    "[2026-03-26T08:02:00Z] [ERROR] Order ORD-9034: payment step failed — payment-service unavailable [service=order-service]\n"
                    "[2026-03-26T08:04:00Z] [WARN]  400 payment failures in last 4 minutes [service=order-service]")
                self._payment_down_noticed = True
                return self._build_observation(result), self._compute_reward("info_gathered"), False
            else:
                logs = LogGenerator.generate(self.task_id, service, self.seed, step_count=self.step_count)
                return self._build_observation(f"Logs for {service}:\n" + "\n".join(logs[:10])), self._compute_reward("info_gathered"), False

        elif at == "check_metrics":
            service = params.get("service", "")
            if service == "payment-service":
                self._payment_down_noticed = True
                result = (f"Metrics for {service}:\n"
                    "  error_rate: 1.0 ⚠️ (no pods running)\n  latency_p99_ms: N/A (service down)\n"
                    "  pods_running: 0/3 ⚠️\n  pods_evicted_10m: 8 ⚠️\n"
                    "  resource_requests_memory: NOT SET ⚠️\n"
                    "  resource_limits_memory: NOT SET ⚠️\n"
                    "  qos_class: BestEffort ⚠️")
                return self._build_observation(result), self._compute_reward("root_cause_progress"), False
            else:
                summary = MetricsSimulator.get_metrics_summary(service, self.task_id, self.seed)
                result = f"Metrics for {service}:\n"
                for k, v in summary.items():
                    if k != "service": result += f"  {k}: {v}\n"
                return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_alerts":
            result = ("Active alerts:\n"
                "  [CRIT] payment-service: 0/3 replicas running — all evicted\n"
                "  [CRIT] Kubernetes: EvictionThresholdMet on node-01, node-02, node-03\n"
                "  [WARN] Node memory pressure: node-01 (95%), node-02 (93%), node-03 (96%)\n"
                "  [WARN] order-service: payment failures cascading to order processing\n"
                "  [INFO] Batch job 'daily-report-generator' consuming 12GB memory on each node ⚠️")
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_deployments":
            service_filter = params.get("service")
            deploys = DeployHistory.get_deploys(self.seed, last_n=5, service_filter=service_filter)
            result = "Recent deployments:\n"
            for d in deploys:
                result += f"  [{d.deploy_id}] {d.service} — {d.description} ({d.minutes_ago}min ago)\n"
            result += ("\n  [deploy-batch-01] batch-job — 'daily-report-generator v2.0' (15min ago) ⚠️\n"
                       "    Resources: NO limits or requests set ⚠️\n"
                       "    DaemonSet: runs on every node\n"
                       "    Observed memory: 12GB per pod (node total: 16GB)")
            return self._build_observation(result), self._compute_reward("info_gathered"), False

        elif at == "check_dependencies":
            service = params.get("service", "")
            upstream = self.graph.get_upstream_of(service)
            downstream = self.graph.get_downstream(service)
            return self._build_observation(f"Dependencies for {service}:\n  Upstream: {upstream}\n  Downstream: {downstream}"), self._compute_reward("info_gathered"), False

        elif at == "run_diagnostic":
            service = params.get("service", "")
            diag_type = params.get("type", "general")
            if diag_type in ("kubernetes", "pods", "node", "eviction", "memory") or service == "payment-service":
                self._node_pressure_identified = True
                self._batch_job_identified = True
                result = ("Diagnostic (kubernetes/node) cluster-wide:\n"
                    "  Node status:\n"
                    "    node-01: memory 95% (15.2G/16G), MemoryPressure=True ⚠️\n"
                    "    node-02: memory 93% (14.9G/16G), MemoryPressure=True ⚠️\n"
                    "    node-03: memory 96% (15.4G/16G), MemoryPressure=True ⚠️\n\n"
                    "  Top memory consumers per node:\n"
                    "    daily-report-generator: 12.1GB per pod ⚠️ (DaemonSet, no limits)\n"
                    "    payment-service: was 1.2GB per pod (evicted — BestEffort QoS)\n"
                    "    order-service: 1.5GB per pod (Guaranteed QoS — not evicted)\n\n"
                    "  Eviction priority (BestEffort evicted first):\n"
                    "    1. payment-service (BestEffort) → EVICTED\n"
                    "    2. other services (Burstable/Guaranteed) → safe for now\n\n"
                    "  💡 Root cause: daily-report-generator deployed as DaemonSet without\n"
                    "  resource limits, consuming 12GB/node (~75% of 16GB nodes).\n"
                    "  payment-service has no resource requests (BestEffort) → first to evict.\n"
                    "  Fix: Delete/limit batch job + add resource requests to payment-service.")
                return self._build_observation(result), self._compute_reward("root_cause_identified"), False
            else:
                return self._build_observation(f"Diagnostic ({diag_type}) for {service}:\n  Normal."), self._compute_reward("info_gathered"), False

        elif at == "apply_fix":
            service = params.get("service", "")
            fix_type = params.get("fix_type", "")
            if fix_type in ("delete_batch_job", "kill_batch_job", "limit_batch_job",
                    "evict_batch_job", "remove_batch_daemonset", "scale_down_batch"):
                self._correct_fix_applied = True
                self._fix_applied = True
                self._root_cause_identified = True
                self._service_fixed = True
                self.incident_phase = IncidentPhase.VERIFYING
                result = ("Fix applied:\n"
                    "  daily-report-generator DaemonSet deleted.\n"
                    "  Node memory freed: 12GB per node recovered.\n"
                    "  Node memory pressure cleared on all nodes.\n"
                    "  payment-service pods rescheduling...\n"
                    "  3/3 payment-service replicas now running.\n"
                    "  Verify health to confirm payment processing restored.")
                return self._build_observation(result), self._compute_reward("fix_applied_correctly"), False
            elif service == "payment-service" and fix_type in ("add_resource_limits",
                    "set_resource_requests"):
                result = ("Resource limits added to payment-service deployment.\n"
                    "  However, nodes still at 95% memory — not enough room to schedule.\n"
                    "  Need to remove the batch job first to free memory.")
                return self._build_observation(result), self._compute_reward("wrong_fix_applied"), False
            elif fix_type == "restart":
                result = f"Pod restart attempted for {service} — evicted again within 30 seconds. Node memory still under pressure."
                return self._build_observation(result), self._compute_reward("wrong_fix_applied"), False
            else:
                self._wrong_fix_count += 1
                return self._build_observation(f"Fix {fix_type} on {service} — pod evictions continue."), self._compute_reward("no_effect"), False

        elif at == "verify_health":
            if self._service_fixed:
                self._resolution_verified = True
                self.incident_phase = IncidentPhase.RESOLVED
                result = ("Health check: ALL SERVICES HEALTHY\n"
                    "  payment-service: 3/3 pods running, error_rate=0.01\n"
                    "  Node memory: node-01=42%, node-02=40%, node-03=43%\n"
                    "  order-service: payment success rate 99.5%\n"
                    "Incident resolved.")
                return self._build_observation(result), self._compute_reward("resolution_verified"), True
            else:
                return self._build_observation("Health check: ISSUES PERSIST\n  payment-service: 0/3 pods running."), self._compute_reward("no_effect"), False

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
            result = "Hint: Node memory pressure from batch job. Check DaemonSet daily-report-generator — no resource limits set."
            return self._build_observation(result), self._compute_reward("escalate_used"), False

        else:
            return self._build_observation(f"Unknown action type: {at}"), self._compute_reward("no_effect"), False

    def get_grader_score(self) -> float:
        score = 0.0
        score += self._payment_down_noticed * 0.08
        score += self._eviction_detected * 0.12
        score += self._node_pressure_identified * 0.12
        score += self._batch_job_identified * 0.15
        score += self._correct_fix_applied * 0.22
        score += self._resolution_verified * 0.13
        score += self._postmortem_written * 0.08
        time_bonus = max(0, 1 - (self.step_count / self.max_steps)) * 0.05
        score += time_bonus
        score += self._evidence_breadth_score()
        score += self._postmortem_quality_bonus(["evict", "memory", "node", "batch", "daemonset", "resource", "limit"])
        score += self._severity_correct * 0.02
        score += (self._status_page_updated and self._status_page_before_fix) * 0.02
        if self._fix_applied and self._fix_before_any_breach:
            score += 0.02
        score -= self._wrong_fix_count * 0.05
        score -= self.hints_used * 0.05
        return round(min(0.999, max(0.001, score)), 4)
