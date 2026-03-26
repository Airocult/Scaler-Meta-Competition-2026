"""
Task 6: Split-Brain Network Partition — Hard difficulty.

Root cause: A network partition between inventory-service and inventory-db
has caused a split-brain condition. inventory-service continues serving
stale cached data while inventory-db has live data, leading to overselling.
The partition was caused by a misconfigured iptables rule deployed
via deploy-net-001 15 minutes ago.

Agent must: detect the inconsistency, identify the network partition,
find the causal deploy, rollback the iptables rule, verify data
consistency, and reconcile stale data.
"""
from app.models import Action, Observation, IncidentPhase, ServiceStatus
from app.scenarios.base import BaseScenario
from app.data.log_templates import LogGenerator
from app.data.metrics import MetricsSimulator


class NetworkPartitionScenario(BaseScenario):
    task_id = "task6_network_partition"
    max_steps = 40

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self._inventory_svc_investigated = False
        self._partition_detected = False
        self._deploy_identified = False
        self._iptables_checked = False
        self._partition_resolved = False
        self._data_reconciled = False
        self._wrong_fix_count = 0
        self._restart_count = 0

    def _get_alert_summary(self) -> str:
        if self._partition_resolved and self._data_reconciled:
            return "RESOLVED: inventory-service and inventory-db connectivity restored, data reconciled."
        if self._partition_resolved:
            return ("PARTIAL RESOLUTION: network partition resolved, but data inconsistency "
                    "may remain. Verify data reconciliation.")
        return ("CRITICAL: inventory-service showing stale stock levels. "
                "Orders succeeding for out-of-stock items. "
                "Data inconsistency detected between inventory-service cache and inventory-db. "
                "Started ~15 minutes ago.")

    def _get_service_statuses(self) -> dict[str, ServiceStatus]:
        statuses = {}
        for svc in self.graph.get_all_services():
            if self._partition_resolved and self._data_reconciled:
                statuses[svc] = ServiceStatus(
                    name=svc, status="healthy", error_rate=0.01,
                    latency_p99_ms=40, restarts_last_hour=0,
                )
            elif svc == "inventory-service":
                # Appears healthy but serving stale data — insidious
                statuses[svc] = ServiceStatus(
                    name=svc, status="degraded", error_rate=0.05,
                    latency_p99_ms=30, restarts_last_hour=0,
                )
            elif svc == "inventory-db":
                statuses[svc] = ServiceStatus(
                    name=svc, status="degraded", error_rate=0.02,
                    latency_p99_ms=15, restarts_last_hour=0,
                )
            elif svc == "order-service":
                statuses[svc] = ServiceStatus(
                    name=svc, status="degraded", error_rate=0.12,
                    latency_p99_ms=200, restarts_last_hour=0,
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
            result += ("\n  ⚠️  Note: inventory-service appears mostly healthy but customer reports "
                       "suggest stale data (items showing 'in stock' that are actually sold out).")
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "read_logs":
            service = params.get("service", "")
            if service == "inventory-service":
                self._inventory_svc_investigated = True
                result = (f"Logs for {service} (last 20 entries):\n"
                          "[2026-03-26T02:10:14Z] [WARN]  Connection to inventory-db timed out — using cached data [service=inventory-service]\n"
                          "[2026-03-26T02:10:18Z] [WARN]  Stale cache fallback: item SKU-1234 showing qty=45 (cache age: 14m) [service=inventory-service]\n"
                          "[2026-03-26T02:10:22Z] [ERROR] TCP connection to inventory-db:5432 failed: Connection timed out (110) [service=inventory-service]\n"
                          "[2026-03-26T02:10:25Z] [WARN]  Cache mode: STALE — last successful DB sync 15m ago [service=inventory-service]\n"
                          "[2026-03-26T02:10:30Z] [INFO]  Serving request for SKU-1234: qty=45 (from stale cache) [service=inventory-service]\n"
                          "[2026-03-26T02:10:35Z] [ERROR] Inventory write FAILED: cannot reach inventory-db [service=inventory-service]\n"
                          "[2026-03-26T02:10:40Z] [WARN]  Queued write operation to WAL (write-ahead-log) — will replay when DB reconnects [service=inventory-service]\n"
                          "[2026-03-26T02:10:45Z] [ERROR] Network unreachable: 10.0.2.50:5432 (inventory-db) [service=inventory-service]\n"
                          "[2026-03-26T02:10:50Z] [WARN]  Split-brain risk: serving reads from cache while DB has diverged [service=inventory-service]\n"
                          "[2026-03-26T02:10:55Z] [ERROR] 15 orders placed for out-of-stock items in last 10 minutes [service=inventory-service]")
                reward = self._compute_reward("root_cause_progress")
                return self._build_observation(result), reward, False

            elif service == "inventory-db":
                self._inventory_svc_investigated = True
                result = (f"Logs for {service} (last 20 entries):\n"
                          "[2026-03-26T02:10:10Z] [INFO]  Database healthy, 0 connections from inventory-service (normally 5-8) [service=inventory-db]\n"
                          "[2026-03-26T02:10:15Z] [WARN]  No heartbeat from inventory-service for 900s [service=inventory-db]\n"
                          "[2026-03-26T02:10:20Z] [INFO]  Local writes succeeding: updated SKU-1234 qty from 45 to 0 (sold out) [service=inventory-db]\n"
                          "[2026-03-26T02:10:25Z] [INFO]  Direct admin queries working — DB is fully functional [service=inventory-db]\n"
                          "[2026-03-26T02:10:30Z] [WARN]  inventory-service client IP 10.0.2.30 — no connection attempts received [service=inventory-db]\n"
                          "[2026-03-26T02:10:35Z] [INFO]  Connection from admin client 10.0.3.1 — query succeeded [service=inventory-db]")
                reward = self._compute_reward("root_cause_progress")
                return self._build_observation(result), reward, False

            elif service == "order-service":
                result = (f"Logs for {service} (last 20 entries):\n"
                          "[2026-03-26T02:10:12Z] [WARN]  Customer complaint: order for SKU-1234 accepted but item is out of stock [service=order-service]\n"
                          "[2026-03-26T02:10:20Z] [INFO]  Inventory check for SKU-1234: qty=45 (returned by inventory-service) [service=order-service]\n"
                          "[2026-03-26T02:10:25Z] [INFO]  Order placed successfully for SKU-1234 x 2 [service=order-service]\n"
                          "[2026-03-26T02:10:30Z] [WARN]  15 orders may be oversold — inventory data appears stale [service=order-service]")
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False

            else:
                logs = LogGenerator.generate("task1_memory_leak", service, self.seed)
                result = f"Logs for {service} (last 20 entries):\n" + "\n".join(logs[:8])
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False

        elif at == "check_metrics":
            service = params.get("service", "")
            if service == "inventory-service":
                self._inventory_svc_investigated = True
                result = (f"Metrics for {service}:\n"
                          "  error_rate: 0.05 (read errors low — stale cache hides issue)\n"
                          "  write_error_rate: 1.00 ⚠️ CRITICAL (all writes failing)\n"
                          "  latency_p99_ms: 30 (reads fast from cache)\n"
                          "  cpu_percent: 12.1\n"
                          "  memory_percent: 55.3 (cache holding all items)\n"
                          "  cache_hit_rate: 1.00 (all from cache — DB unreachable)\n"
                          "  cache_staleness_seconds: 920 ⚠️\n"
                          "  db_connection_pool: 0/10 active ⚠️\n"
                          "  wal_pending_writes: 47")
                reward = self._compute_reward("root_cause_progress")
                return self._build_observation(result), reward, False
            elif service == "inventory-db":
                result = (f"Metrics for {service}:\n"
                          "  error_rate: 0.02\n"
                          "  latency_p99_ms: 15\n"
                          "  cpu_percent: 8.3\n"
                          "  memory_percent: 30.2\n"
                          "  active_connections: 2 (admin only — normally 7-12)\n"
                          "  replication: N/A (single node)\n"
                          "  disk_usage: 42%")
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False
            else:
                summary = MetricsSimulator.get_metrics_summary(service, "task1_memory_leak", self.seed)
                result = f"Metrics for {service}:\n"
                for k, v in summary.items():
                    if k != "service":
                        result += f"  {k}: {v}\n"
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False

        elif at == "check_alerts":
            result = ("Active alerts:\n"
                      "  [CRITICAL] inventory-service: 100% write failure rate — DB unreachable\n"
                      "  [WARN] inventory-service: serving stale cached data (15+ minutes old)\n"
                      "  [WARN] order-service: possible overselling — 15 orders for out-of-stock items\n"
                      "  [INFO] inventory-db: healthy but 0 connections from inventory-service\n"
                      "  [INFO] Network change detected 15 minutes ago (deploy-net-001)")
            self._partition_detected = True
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "check_deployments":
            service_filter = params.get("service")
            from app.data.deploy_history import DeployHistory
            deploys = DeployHistory.get_deploys(self.seed, last_n=5, service_filter=service_filter)
            result = "Recent deployments:\n"
            for d in deploys:
                result += f"  [{d.deploy_id}] {d.service} — {d.description} ({d.minutes_ago}min ago)\n"
            result += ("\n  [deploy-net-001] infrastructure — 'Network security hardening: updated iptables rules' (15min ago) ⚠️\n"
                       "    Changed: iptables -A FORWARD -s 10.0.2.30 -d 10.0.2.50 -j DROP\n"
                       "    This blocks traffic from inventory-service (10.0.2.30) → inventory-db (10.0.2.50)")
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
            if service == "inventory-service":
                result += ("\n\n  ⚠️  inventory-service → inventory-db connection over TCP 10.0.2.30 → 10.0.2.50:5432\n"
                           "  Currently: NO active connections (expecting 5-8)")
                self._partition_detected = True
            reward = self._compute_reward("info_gathered")
            return self._build_observation(result), reward, False

        elif at == "run_diagnostic":
            service = params.get("service", "")
            diag_type = params.get("type", "general")

            if service == "inventory-service" and diag_type in ("network", "connectivity", "iptables", "general"):
                self._partition_detected = True
                self._iptables_checked = True
                result = (f"Diagnostic (network) for {service}:\n"
                          "  ping inventory-db (10.0.2.50): 100% packet loss ⚠️\n"
                          "  traceroute to 10.0.2.50: * * * (blocked at hop 1)\n"
                          "  TCP connect 10.0.2.50:5432: Connection timed out\n"
                          "  iptables -L FORWARD:\n"
                          "    Chain FORWARD (policy ACCEPT)\n"
                          "    DROP all -- 10.0.2.30 10.0.2.50  ⚠️ BLOCKING RULE\n\n"
                          "  💡 Root cause: iptables rule is blocking traffic from inventory-service\n"
                          "  (10.0.2.30) to inventory-db (10.0.2.50).\n"
                          "  This was added by deploy-net-001 15 minutes ago.\n"
                          "  Fix: rollback deploy-net-001 or remove the iptables rule, then\n"
                          "  reconcile stale data in inventory-service cache.")
                reward = self._compute_reward("root_cause_identified")
                return self._build_observation(result), reward, False

            elif service == "inventory-db" and diag_type in ("network", "connectivity", "general"):
                self._partition_detected = True
                result = (f"Diagnostic (network) for {service}:\n"
                          "  ping inventory-service (10.0.2.30): 100% packet loss ⚠️\n"
                          "  DB health: all internal checks pass\n"
                          "  DB data: SKU-1234 qty=0 (sold out), SKU-5678 qty=12\n"
                          "    (inventory-service cache shows SKU-1234 qty=45 — MISMATCH)")
                reward = self._compute_reward("root_cause_progress")
                return self._build_observation(result), reward, False

            else:
                result = f"Diagnostic ({diag_type}) for {service}:\n  All metrics within normal range."
                reward = self._compute_reward("info_gathered")
                return self._build_observation(result), reward, False

        elif at == "apply_fix":
            service = params.get("service", "")
            fix_type = params.get("fix_type", "")

            # Correct fix 1: remove iptables rule / rollback deploy
            if fix_type in ("rollback_deploy", "rollback", "remove_iptables_rule",
                            "remove_iptables", "delete_iptables_rule",
                            "restore_network", "fix_iptables"):
                self._partition_resolved = True
                self._fix_applied = True
                self._root_cause_identified = True
                self.incident_phase = IncidentPhase.VERIFYING
                result = ("Fix applied: iptables blocking rule removed.\n"
                          "  deploy-net-001 rolled back.\n"
                          "  inventory-service (10.0.2.30) → inventory-db (10.0.2.50): connectivity restored\n"
                          "  inventory-service reconnecting to inventory-db...\n"
                          "  DB connection pool: 5/10 active and growing.\n\n"
                          "  ⚠️  WARNING: inventory-service cache has stale data.\n"
                          "  47 WAL entries pending replay. Data reconciliation needed.\n"
                          "  Apply fix: reconcile_data / flush_cache on inventory-service.")
                reward = self._compute_reward("fix_applied_correctly")
                return self._build_observation(result), reward, False

            # Correct fix 2: reconcile data (after partition resolved)
            elif service == "inventory-service" and fix_type in ("reconcile_data", "flush_cache",
                                                                   "sync_cache", "invalidate_cache",
                                                                   "reconcile"):
                if self._partition_resolved:
                    self._data_reconciled = True
                    result = ("Data reconciliation complete:\n"
                              "  Cache invalidated — all entries refreshed from inventory-db.\n"
                              "  47 WAL entries replayed successfully.\n"
                              "  SKU-1234: cache updated 45 → 0 (correct).\n"
                              "  15 oversold orders flagged for manual review.\n"
                              "  inventory-service now serving live data.\n"
                              "Verify health to confirm full resolution.")
                    reward = self._compute_reward("resolution_verified")
                    return self._build_observation(result), reward, False
                else:
                    result = ("Cannot reconcile data — inventory-service still can't reach inventory-db.\n"
                              "Fix the network partition first.")
                    reward = self._compute_reward("no_effect")
                    return self._build_observation(result), reward, False

            elif service == "inventory-service" and fix_type in ("restart", "restart_service"):
                self._restart_count += 1
                if not self._partition_resolved:
                    result = ("inventory-service restarted, but still can't connect to inventory-db.\n"
                              "  Network partition still in effect — iptables rule blocking traffic.\n"
                              "  Service will load empty cache and start failing reads too.")
                    reward = self._compute_reward("wrong_fix_applied")
                else:
                    result = ("inventory-service restarted. Network is restored. "
                              "Cache rebuilding from inventory-db.")
                    reward = self._compute_reward("no_effect")
                return self._build_observation(result), reward, False

            else:
                self._wrong_fix_count += 1
                result = f"Fix applied to {service} ({fix_type}) — no effect on the network partition."
                reward = self._compute_reward("no_effect")
                return self._build_observation(result), reward, False

        elif at == "verify_health":
            if self._partition_resolved and self._data_reconciled:
                self._resolution_verified = True
                self.incident_phase = IncidentPhase.RESOLVED
                result = ("Health check: ALL SERVICES HEALTHY\n"
                          "  inventory-service: connected to DB, cache=live, error_rate=0.01\n"
                          "  inventory-db: 7 active connections, healthy\n"
                          "  order-service: no new overselling detected\n"
                          "  Network: 10.0.2.30 ↔ 10.0.2.50 connectivity confirmed\n"
                          "  Data: consistent between cache and DB\n"
                          "Incident resolved. Consider writing a postmortem.")
                reward = self._compute_reward("resolution_verified")
                return self._build_observation(result), reward, True
            elif self._partition_resolved:
                result = ("Health check: PARTIAL RESOLUTION\n"
                          "  Network partition resolved, connections restored.\n"
                          "  ⚠️  inventory-service cache still stale — data reconciliation needed.\n"
                          "  Apply fix: reconcile_data on inventory-service.")
                reward = self._compute_reward("no_effect")
                return self._build_observation(result), reward, False
            else:
                result = ("Health check: ISSUES PERSIST\n"
                          "  inventory-service → inventory-db: connection timed out\n"
                          "  Network partition still active.\n"
                          "  Stale data being served.")
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
            result = ("Escalation hint: inventory-service is serving stale cached data because\n"
                      "it can't reach inventory-db. Check network connectivity between them.\n"
                      "There was a recent infrastructure deploy that changed network rules.\n"
                      "After fixing the network, you'll need to reconcile the stale cache.")
            reward = self._compute_reward("escalate_used")
            return self._build_observation(result), reward, False

        else:
            result = f"Unknown action type: {at}"
            reward = self._compute_reward("no_effect")
            return self._build_observation(result), reward, False

    def get_grader_score(self) -> float:
        score = 0.0
        score += self._inventory_svc_investigated * 0.10
        score += self._partition_detected * 0.10
        score += self._deploy_identified * 0.10
        score += self._iptables_checked * 0.10
        score += self._partition_resolved * 0.20
        score += self._data_reconciled * 0.15
        score += self._resolution_verified * 0.10
        score += self._postmortem_written * 0.10

        # Time bonus
        time_bonus = max(0, 1 - (self.step_count / self.max_steps)) * 0.05
        score += time_bonus

        # Penalties
        score -= self._wrong_fix_count * 0.05
        score -= min(self._restart_count, 3) * 0.03
        score -= self.hints_used * 0.075

        return round(min(1.0, max(0.0, score)), 4)
