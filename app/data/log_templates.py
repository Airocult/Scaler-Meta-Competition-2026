"""
Realistic log line generator using templates + seeded randomness.
No external data files — everything generated programmatically.
"""
import random as _random_module
from datetime import datetime, timedelta


class LogGenerator:
    """Produces realistic log lines for each scenario and service."""

    MEMORY_LEAK_LOGS_AFFECTED = [
        "[ERROR] OOMKilled: container exceeded memory limit 512Mi",
        "[ERROR] java.lang.OutOfMemoryError: Java heap space",
        "[WARN]  Received signal 9 (SIGKILL) — container killed by kernel OOM",
        "[ERROR] Pod restarted. Restart count: {restart_count}",
        "[WARN]  GC overhead limit exceeded — 98% of time spent in garbage collection",
        "[ERROR] Memory usage at {mem_pct}% — approaching container limit",
        "[WARN]  Heap dump written to /tmp/heapdump-{pid}.hprof",
        "[ERROR] Failed to allocate {alloc_mb}MB — insufficient memory",
    ]

    MEMORY_LEAK_LOGS_HEALTHY = [
        "[INFO]  Request processed successfully in {latency}ms",
        "[INFO]  Health check passed — all subsystems OK",
        "[DEBUG] Cache hit ratio: {hit_ratio}%",
        "[INFO]  Connection pool: {pool_active}/{pool_max} active connections",
        "[INFO]  Upstream response from {upstream} in {latency}ms",
    ]

    DB_POOL_LOGS_AFFECTED = [
        "[ERROR] HikariPool-1 - Connection is not available, request timed out after 30000ms",
        "[ERROR] Too many connections (max: {max_conn})",
        "[ERROR] upstream connect error or disconnect/reset before headers",
        "[ERROR] Connection pool exhausted for datasource payment-db",
        "[WARN]  Retrying connection attempt {attempt}/3",
        "[ERROR] SqlTransientConnectionException: could not acquire connection from pool",
        "[WARN]  Active connections: {active}/{max_conn} — pool at capacity",
        "[ERROR] Query timeout after 30s waiting for available connection",
    ]

    DB_POOL_LOGS_DEGRADED = [
        "[WARN]  Increased latency detected — upstream payment-service responding slowly",
        "[ERROR] HTTP 503 Service Unavailable from payment-service",
        "[WARN]  Circuit breaker OPEN for payment-service — 5 consecutive failures",
        "[ERROR] Request failed: upstream dependency payment-service unhealthy",
    ]

    RACE_CONDITION_LOGS_AFFECTED = [
        "[ERROR] CacheKeyConflict: stale read detected on key inventory:{item_id}",
        "[ERROR] Inconsistent state: expected {expected} got {actual} for user {user_id}",
        "[ERROR] RedisLock acquisition timeout after 500ms",
        "[ERROR] HTTP 500 Internal Server Error — InventoryConsistencyException",
        "[WARN]  Lock contention detected: {contention_count} concurrent writers on key {key}",
        "[ERROR] Optimistic lock failure: version mismatch on inventory record",
    ]

    RACE_CONDITION_LOGS_NORMAL = [
        "[INFO]  Request processed successfully in {latency}ms",
        "[INFO]  Redis GET inventory:{item_id} — cache HIT ({hit_latency}ms)",
        "[INFO]  RedisLock acquired on key {key} in {lock_latency}ms",
        "[DEBUG] Cache TTL refresh for inventory:{item_id}",
    ]

    DNS_FAILURE_LOGS_AFFECTED = [
        "[ERROR] DNS resolution failed for user-db.internal: NXDOMAIN",
        "[ERROR] getaddrinfo ENOTFOUND user-db.internal",
        "[WARN]  DNS cache stale — TTL expired {cache_age}s ago for user-db.internal",
        "[ERROR] Connection refused to user-db: DNS resolution returned stale IP 10.0.0.99",
        "[ERROR] DNS retry failed — resolver 10.96.0.10 not responding",
        "[WARN]  Falling back to cached DNS entry (stale, age={cache_age}s)",
        "[ERROR] Connection timeout to 10.0.0.99:5432 (stale cached IP)",
        "[ERROR] Auth request failed: upstream user-db unreachable via DNS",
    ]

    DNS_FAILURE_LOGS_UPSTREAM = [
        "[ERROR] HTTP 503 from auth-service — authentication unavailable",
        "[WARN]  Auth dependency degraded, {error_count} failed requests in last 5m",
        "[ERROR] Request failed: upstream auth-service returned 503",
        "[WARN]  Circuit breaker HALF-OPEN for auth-service",
    ]

    DNS_FAILURE_LOGS_HEALTHY = [
        "[INFO]  Request processed successfully in {latency}ms",
        "[INFO]  Health check passed — all subsystems OK",
        "[INFO]  DNS resolution: user-db.internal → 10.0.1.15 in {dns_latency}ms",
        "[DEBUG] Connection pool: {pool_active}/{pool_max} active connections",
    ]

    CERT_EXPIRY_LOGS_AFFECTED = [
        "[ERROR] TLS handshake failed: certificate has expired",
        "[ERROR] SSL_ERROR_CERTIFICATE_EXPIRED: /CN=payment-service.internal expired",
        "[ERROR] mTLS to payment-db failed: client cert expired",
        "[ERROR] Unable to connect to payment-db: SSL handshake failure (CERT_HAS_EXPIRED)",
        "[WARN]  Connection pool to payment-db: 0/{pool_max} active (all failed SSL)",
        "[ERROR] Payment request rejected: TLS termination failed",
        "[ERROR] 0 of last {request_count} payment requests succeeded",
        "[WARN]  Certificate auto-renewal status: FAILED (cert-manager OOMKilled)",
    ]

    CERT_EXPIRY_LOGS_UPSTREAM = [
        "[ERROR] Failed to connect to payment-service: SSL handshake error",
        "[ERROR] PKIX path validation failed: certificate expired on remote host",
        "[WARN]  Payment service circuit breaker OPEN — 100% failure rate",
        "[ERROR] Order failed: payment processing unavailable (SSL error)",
    ]

    CERT_EXPIRY_LOGS_HEALTHY = [
        "[INFO]  Request processed successfully in {latency}ms",
        "[INFO]  TLS handshake completed in {tls_latency}ms",
        "[INFO]  Health check passed — all subsystems OK",
        "[DEBUG] Certificate valid, {cert_days_remaining} days remaining",
    ]

    NETWORK_PARTITION_LOGS_AFFECTED = [
        "[ERROR] TCP connection to inventory-db:5432 failed: Connection timed out (110)",
        "[WARN]  Stale cache fallback: item SKU-{item_id} showing cached qty={cache_qty}",
        "[WARN]  Cache mode: STALE — last successful DB sync {cache_age}s ago",
        "[ERROR] Inventory write FAILED: cannot reach inventory-db",
        "[ERROR] Network unreachable: 10.0.2.50:5432 (inventory-db)",
        "[WARN]  Split-brain risk: serving reads from cache while DB has diverged",
        "[WARN]  Queued write to WAL — {wal_pending} entries pending replay",
        "[ERROR] {oversold_count} orders placed for out-of-stock items in last 10m",
    ]

    NETWORK_PARTITION_LOGS_DB = [
        "[INFO]  Database healthy, {db_connections} connections (normally 5-8)",
        "[WARN]  No heartbeat from inventory-service for {heartbeat_age}s",
        "[INFO]  Local writes succeeding: updated SKU-{item_id} qty to {db_qty}",
        "[INFO]  Direct admin queries working — DB is fully functional",
        "[WARN]  Client IP 10.0.2.30 — no connection attempts received",
    ]

    NETWORK_PARTITION_LOGS_DOWNSTREAM = [
        "[WARN]  Customer complaint: order accepted but item may be out of stock",
        "[INFO]  Inventory check returned qty={cache_qty} (possibly stale)",
        "[WARN]  {oversold_count} orders may be oversold — inventory data appears stale",
        "[ERROR] Fulfillment failed: item SKU-{item_id} actually out of stock",
    ]

    NETWORK_PARTITION_LOGS_HEALTHY = [
        "[INFO]  Request processed successfully in {latency}ms",
        "[INFO]  Health check passed — all subsystems OK",
        "[DEBUG] Connection pool: {pool_active}/{pool_max} active connections",
        "[INFO]  Inventory sync healthy — last update {latency}ms ago",
    ]

    @classmethod
    def generate(cls, scenario: str, service: str, seed: int, n: int = 20) -> list[str]:
        """Generate n log lines for a given scenario and service."""
        rng = _random_module.Random(seed)
        base_time = datetime(2026, 3, 26, 3, 0, 0)
        lines = []

        for i in range(n):
            ts = base_time + timedelta(seconds=rng.randint(0, 3600))
            ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{rng.randint(0, 999):03d}Z"
            trace_id = f"{rng.randint(0, 0xFFFFFFFF):08x}"

            template = cls._pick_template(rng, scenario, service)
            msg = cls._fill_template(rng, template)
            lines.append(f"[{ts_str}] {msg} [service={service}] [trace={trace_id}]")

        lines.sort()  # Sort by timestamp for realism
        return lines

    @classmethod
    def _pick_template(cls, rng: _random_module.Random, scenario: str, service: str) -> str:
        if scenario == "task1_memory_leak":
            if service == "order-service":
                return rng.choice(cls.MEMORY_LEAK_LOGS_AFFECTED)
            return rng.choice(cls.MEMORY_LEAK_LOGS_HEALTHY)

        elif scenario == "task2_db_cascade":
            if service == "payment-db":
                return rng.choice(cls.DB_POOL_LOGS_AFFECTED)
            elif service in ("payment-service", "order-service", "api-gateway"):
                return rng.choice(cls.DB_POOL_LOGS_DEGRADED)
            return rng.choice(cls.MEMORY_LEAK_LOGS_HEALTHY)

        elif scenario == "task3_race_condition":
            if service == "inventory-service":
                return rng.choice(cls.RACE_CONDITION_LOGS_AFFECTED)
            elif service in ("order-service", "api-gateway"):
                # Downstream services see errors too
                return rng.choice(cls.DB_POOL_LOGS_DEGRADED)
            return rng.choice(cls.RACE_CONDITION_LOGS_NORMAL)

        elif scenario == "task4_dns_failure":
            if service == "auth-service":
                return rng.choice(cls.DNS_FAILURE_LOGS_AFFECTED)
            elif service in ("api-gateway",):
                return rng.choice(cls.DNS_FAILURE_LOGS_UPSTREAM)
            return rng.choice(cls.DNS_FAILURE_LOGS_HEALTHY)

        elif scenario == "task5_cert_expiry":
            if service == "payment-service":
                return rng.choice(cls.CERT_EXPIRY_LOGS_AFFECTED)
            elif service in ("order-service", "api-gateway"):
                return rng.choice(cls.CERT_EXPIRY_LOGS_UPSTREAM)
            return rng.choice(cls.CERT_EXPIRY_LOGS_HEALTHY)

        elif scenario == "task6_network_partition":
            if service == "inventory-service":
                return rng.choice(cls.NETWORK_PARTITION_LOGS_AFFECTED)
            elif service == "inventory-db":
                return rng.choice(cls.NETWORK_PARTITION_LOGS_DB)
            elif service in ("order-service",):
                return rng.choice(cls.NETWORK_PARTITION_LOGS_DOWNSTREAM)
            return rng.choice(cls.NETWORK_PARTITION_LOGS_HEALTHY)

        return rng.choice(cls.MEMORY_LEAK_LOGS_HEALTHY)

    @classmethod
    def _fill_template(cls, rng: _random_module.Random, template: str) -> str:
        replacements = {
            "{restart_count}": str(rng.randint(1, 8)),
            "{mem_pct}": str(rng.randint(92, 99)),
            "{pid}": str(rng.randint(1000, 9999)),
            "{alloc_mb}": str(rng.randint(64, 512)),
            "{latency}": str(rng.randint(2, 250)),
            "{hit_ratio}": str(rng.randint(60, 99)),
            "{pool_active}": str(rng.randint(1, 10)),
            "{pool_max}": "20",
            "{upstream}": rng.choice(["auth-service", "order-service", "payment-service"]),
            "{max_conn}": str(rng.randint(50, 200)),
            "{attempt}": str(rng.randint(1, 3)),
            "{active}": str(rng.randint(45, 50)),
            "{item_id}": str(rng.randint(10000, 99999)),
            "{expected}": str(rng.randint(1, 100)),
            "{actual}": str(rng.randint(1, 100)),
            "{user_id}": f"usr_{rng.randint(1000, 9999)}",
            "{contention_count}": str(rng.randint(3, 15)),
            "{key}": f"inventory:{rng.randint(10000, 99999)}",
            "{hit_latency}": str(rng.randint(1, 5)),
            "{lock_latency}": str(rng.randint(10, 200)),
            # DNS failure placeholders
            "{cache_age}": str(rng.randint(120, 960)),
            "{error_count}": str(rng.randint(50, 500)),
            "{dns_latency}": str(rng.randint(1, 8)),
            # Certificate expiry placeholders
            "{request_count}": str(rng.randint(50, 200)),
            "{tls_latency}": str(rng.randint(5, 25)),
            "{cert_days_remaining}": str(rng.randint(30, 365)),
            # Network partition placeholders
            "{cache_qty}": str(rng.randint(10, 200)),
            "{db_qty}": str(rng.randint(0, 5)),
            "{db_connections}": str(rng.randint(0, 2)),
            "{heartbeat_age}": str(rng.randint(600, 1200)),
            "{wal_pending}": str(rng.randint(10, 80)),
            "{oversold_count}": str(rng.randint(5, 30)),
        }
        result = template
        for k, v in replacements.items():
            result = result.replace(k, v)
        return result
