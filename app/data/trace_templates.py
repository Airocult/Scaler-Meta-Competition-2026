"""
Distributed trace generator — produces realistic request span waterfalls.

Each scenario seeds traces that reveal the root cause through latency
and error patterns. Traces vary by step_count for temporal variation.
"""
import random as _random_module


class Span:
    """Single span in a distributed trace."""
    __slots__ = ("span_id", "parent_span_id", "service", "operation",
                 "duration_ms", "status", "tags")

    def __init__(self, span_id: str, parent_span_id: str, service: str,
                 operation: str, duration_ms: int, status: str,
                 tags: dict | None = None):
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.service = service
        self.operation = operation
        self.duration_ms = duration_ms
        self.status = status
        self.tags = tags or {}

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "service": self.service,
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "tags": self.tags,
        }


class TraceGenerator:
    """Generates deterministic distributed traces per scenario/seed/step."""

    @staticmethod
    def generate(task_id: str, service: str, seed: int, step_count: int = 0) -> str:
        """Return a formatted trace waterfall string for the given context."""
        rng = _random_module.Random(seed + hash(task_id) + step_count * 7919)
        trace_id = f"trace-{rng.randint(100000, 999999)}"

        spans = _GENERATORS.get(task_id, _default_trace)(service, rng, step_count)

        # Format as ASCII waterfall
        lines = [f"Trace ID: {trace_id}  (service entry: {service})"]
        lines.append(f"{'Service':<22} {'Operation':<28} {'Duration':>10}  {'Status':<8} Tags")
        lines.append("─" * 95)
        for s in spans:
            tag_str = ", ".join(f"{k}={v}" for k, v in s.tags.items()) if s.tags else ""
            lines.append(
                f"  {s.service:<20} {s.operation:<28} {s.duration_ms:>8}ms  {s.status:<8} {tag_str}"
            )
        lines.append("─" * 95)

        # Summarise anomalies
        anomalies = [s for s in spans if s.status == "error" or s.duration_ms > 5000]
        if anomalies:
            lines.append("⚠️  Anomalies detected:")
            for a in anomalies:
                lines.append(f"  - {a.service}/{a.operation}: {a.duration_ms}ms ({a.status})")

        return "\n".join(lines)


# ── Per-scenario trace generators ────────────────────────────────────────────

def _trace_task1_memory_leak(service: str, rng: _random_module.Random, step: int) -> list[Span]:
    """order-service spans show high latency + GC pauses."""
    base_lat = 45 + rng.randint(-5, 5)
    order_lat = 12000 + rng.randint(-500, 1500) + step * 200  # gets worse
    gc_pause = 8000 + rng.randint(0, 2000)
    return [
        Span("s1", "", "api-gateway", "POST /api/orders", order_lat + 60, "error",
             {"http.status": 503}),
        Span("s2", "s1", "order-service", "processOrder", order_lat, "error",
             {"error": "OutOfMemoryError", "heap_used": "98%"}),
        Span("s3", "s2", "order-service", "GC.fullCollection", gc_pause, "ok",
             {"gc_type": "full", "freed_mb": rng.randint(2, 8)}),
        Span("s4", "s2", "inventory-service", "checkStock", base_lat, "ok", {}),
        Span("s5", "s2", "payment-service", "chargeCard", base_lat + 10, "ok", {}),
    ]


def _trace_task2_db_cascade(service: str, rng: _random_module.Random, step: int) -> list[Span]:
    """payment-db shows 30s connection wait; cascade visible."""
    base_lat = 40 + rng.randint(-5, 5)
    db_wait = 30000 + rng.randint(-1000, 2000)
    return [
        Span("s1", "", "api-gateway", "POST /api/orders", db_wait + 200, "error",
             {"http.status": 503}),
        Span("s2", "s1", "order-service", "processOrder", db_wait + 100, "error",
             {"error": "UpstreamTimeout"}),
        Span("s3", "s2", "payment-service", "processPayment", db_wait + 50, "error",
             {"error": "ConnectionPoolExhausted", "pool_active": "198/200"}),
        Span("s4", "s3", "payment-db", "connection_pool_wait", db_wait, "error",
             {"pool_usage": "0.99", "waiting_queries": 342, "error": "AcquireTimeoutException"}),
        Span("s5", "s2", "inventory-service", "checkStock", base_lat, "ok", {}),
    ]


def _trace_task3_race_condition(service: str, rng: _random_module.Random, step: int) -> list[Span]:
    """inventory-service shows intermittent lock timeout errors."""
    base_lat = 40 + rng.randint(-5, 5)
    lock_timeout = 500  # the bad config value
    has_error = rng.random() < 0.4  # intermittent
    inv_status = "error" if has_error else "ok"
    inv_lat = lock_timeout + rng.randint(10, 100) if has_error else base_lat + rng.randint(10, 30)
    return [
        Span("s1", "", "api-gateway", "GET /api/inventory", inv_lat + 100, inv_status,
             {"http.status": 500 if has_error else 200}),
        Span("s2", "s1", "order-service", "checkInventory", inv_lat + 50, inv_status, {}),
        Span("s3", "s2", "inventory-service", "getStockLevel", inv_lat, inv_status,
             {"error": "CacheKeyConflict: stale read"} if has_error else {}),
        Span("s4", "s3", "inventory-service", "redis.acquireLock", lock_timeout if has_error else 12, inv_status,
             {"lock_timeout_ms": 500, "key": f"inventory:SKU-{rng.randint(1000,9999)}"}),
        Span("s5", "s3", "inventory-db", "SELECT stock", base_lat, "ok", {}),
    ]


def _trace_task4_dns_failure(service: str, rng: _random_module.Random, step: int) -> list[Span]:
    """auth-service shows DNS resolution timeout spans."""
    base_lat = 40 + rng.randint(-5, 5)
    dns_timeout = 5000 + rng.randint(0, 1000)
    return [
        Span("s1", "", "api-gateway", "POST /api/login", dns_timeout + 200, "error",
             {"http.status": 503}),
        Span("s2", "s1", "auth-service", "authenticateUser", dns_timeout + 100, "error",
             {"error": "UpstreamUnreachable"}),
        Span("s3", "s2", "auth-service", "dns.resolve(user-db.internal)", dns_timeout, "error",
             {"resolver": "10.96.0.10", "error": "NXDOMAIN", "cached_ip": "10.0.0.99(stale)"}),
        Span("s4", "s2", "auth-service", "tcp.connect(10.0.0.99:5432)", 5000, "error",
             {"error": "ConnectionTimeout", "note": "stale_ip"}),
        Span("s5", "s1", "order-service", "processOrder", base_lat, "ok", {}),
    ]


def _trace_task5_cert_expiry(service: str, rng: _random_module.Random, step: int) -> list[Span]:
    """payment-service TLS handshake fails immediately."""
    base_lat = 40 + rng.randint(-5, 5)
    return [
        Span("s1", "", "api-gateway", "POST /api/orders", 120, "error",
             {"http.status": 503}),
        Span("s2", "s1", "order-service", "processPayment", 80, "error",
             {"error": "SSLHandshakeException"}),
        Span("s3", "s2", "order-service", "tls.handshake(payment-service)", 15, "error",
             {"error": "CERT_HAS_EXPIRED", "cert_cn": "payment-service.internal",
              "expired_at": "2026-03-25T23:59:59Z"}),
        Span("s4", "s2", "payment-service", "tls.accept", 5, "error",
             {"error": "certificate_expired", "serial": "0x1A2B3C4D"}),
        Span("s5", "s4", "payment-service", "mtls.connect(payment-db)", 5, "error",
             {"error": "client_cert_expired"}),
    ]


def _trace_task6_network_partition(service: str, rng: _random_module.Random, step: int) -> list[Span]:
    """inventory-service → inventory-db spans timeout 100%."""
    base_lat = 40 + rng.randint(-5, 5)
    return [
        Span("s1", "", "api-gateway", "GET /api/inventory/SKU-1234", 5100, "ok",
             {"http.status": 200, "note": "served_from_stale_cache"}),
        Span("s2", "s1", "order-service", "checkInventory", 5050, "ok", {}),
        Span("s3", "s2", "inventory-service", "getStockLevel", 5020, "ok",
             {"source": "stale_cache", "cache_age_s": 920 + step * 10}),
        Span("s4", "s3", "inventory-service", "tcp.connect(inventory-db:5432)", 5000, "error",
             {"error": "ConnectionTimedOut", "dst": "10.0.2.50:5432",
              "src": "10.0.2.30", "note": "iptables_DROP_rule"}),
        Span("s5", "s3", "inventory-service", "cache.fallback", 2, "ok",
             {"qty_returned": 45, "actual_qty": 0, "stale": True}),
    ]


def _trace_task7_kafka_lag(service: str, rng: _random_module.Random, step: int) -> list[Span]:
    """order-service shows Kafka consumer lag and rebalance churn."""
    base_lat = 45 + rng.randint(-5, 5)
    order_lat = 42000 + rng.randint(-2000, 5000)
    return [
        Span("s1", "", "api-gateway", "POST /api/orders", order_lat + 100, "error",
             {"http.status": 504}),
        Span("s2", "s1", "order-service", "processOrder", order_lat, "error",
             {"error": "KafkaConsumerLag", "lag_messages": 12847}),
        Span("s3", "s2", "order-service", "kafka.consume(orders-topic)", order_lat - 100, "error",
             {"session_timeout_ms": 3000, "rebalance_count": 47, "error": "ConsumerRebalanceTriggered"}),
        Span("s4", "s2", "inventory-service", "checkStock", base_lat, "ok", {}),
        Span("s5", "s2", "payment-service", "processPayment", base_lat + 10, "ok", {}),
    ]


def _trace_task8_redis_failover(service: str, rng: _random_module.Random, step: int) -> list[Span]:
    """inventory-service Redis miss cascades to inventory-db overload."""
    base_lat = 45 + rng.randint(-5, 5)
    db_lat = 3200 + rng.randint(-200, 500)
    return [
        Span("s1", "", "api-gateway", "GET /api/inventory", db_lat + 200, "error",
             {"http.status": 503}),
        Span("s2", "s1", "order-service", "checkInventory", db_lat + 150, "error", {}),
        Span("s3", "s2", "inventory-service", "getStockLevel", db_lat + 100, "error",
             {"cache_hit": False, "error": "RedisConnectionRefused"}),
        Span("s4", "s3", "inventory-service", "redis.connect(primary)", 5000, "error",
             {"error": "ConnectionRefused", "sentinel_quorum": "2/3 (need 3)"}),
        Span("s5", "s3", "inventory-db", "SELECT stock", db_lat, "ok",
             {"pool_usage": "148/150", "slow": True}),
    ]


def _trace_task9_disk_full(service: str, rng: _random_module.Random, step: int) -> list[Span]:
    """user-db write fails due to disk full, auth-service cascades."""
    base_lat = 45 + rng.randint(-5, 5)
    return [
        Span("s1", "", "api-gateway", "POST /api/login", 11200, "error",
             {"http.status": 500}),
        Span("s2", "s1", "auth-service", "authenticateUser", 11100, "error",
             {"error": "DatabaseWriteError"}),
        Span("s3", "s2", "auth-service", "updateLastLogin", 11000, "error",
             {"error": "DiskFullError: No space left on device"}),
        Span("s4", "s3", "user-db", "INSERT sessions", 10900, "error",
             {"error": "ENOSPC", "disk_usage": "100%", "wal_size_gb": 285}),
        Span("s5", "s1", "order-service", "processOrder", base_lat, "ok", {}),
    ]


def _trace_task10_rate_limit(service: str, rng: _random_module.Random, step: int) -> list[Span]:
    """api-gateway rate limiter rejects 90% of traffic immediately."""
    base_lat = 45 + rng.randint(-5, 5)
    return [
        Span("s1", "", "api-gateway", "GET /api/orders", 5, "error",
             {"http.status": 429, "rate_limit": "100/s", "actual_traffic": "1050/s"}),
        Span("s2", "s1", "api-gateway", "rateLimiter.check", 2, "error",
             {"error": "RateLimitExceeded", "limit": 100, "expected": 10000}),
        Span("s3", "", "api-gateway", "GET /api/orders", base_lat + 80, "ok",
             {"http.status": 200, "note": "1_in_10_passes_rate_limit"}),
        Span("s4", "s3", "order-service", "processOrder", base_lat + 20, "ok", {}),
    ]


def _trace_task11_db_migration_lock(service: str, rng: _random_module.Random, step: int) -> list[Span]:
    """payment-db ALTER TABLE holds exclusive lock, all writes blocked."""
    wait_ms = 30000 + rng.randint(-1000, 2000)
    return [
        Span("s1", "", "api-gateway", "POST /api/orders", wait_ms + 500, "error",
             {"http.status": 503}),
        Span("s2", "s1", "order-service", "processPayment", wait_ms + 300, "error",
             {"error": "UpstreamTimeout"}),
        Span("s3", "s2", "payment-service", "processPayment", wait_ms + 100, "error",
             {"error": "LockWaitTimeout"}),
        Span("s4", "s3", "payment-db", "INSERT payment_transactions", wait_ms, "error",
             {"error": "AccessExclusiveLock", "blocked_by_pid": 8234,
              "blocking_query": "ALTER TABLE payment_transactions ADD COLUMN refund_id"}),
    ]


def _trace_task12_health_flap(service: str, rng: _random_module.Random, step: int) -> list[Span]:
    """order-service deep health check times out on inventory dependency."""
    inv_lat = 1800 + rng.randint(-200, 600)
    total = inv_lat + 210
    exceeded = total > 2000
    return [
        Span("s1", "", "order-service", "/health/deep", total, "error" if exceeded else "ok",
             {"timeout_ms": 2000, "exceeded": exceeded}),
        Span("s2", "s1", "order-service", "localChecks", 5, "ok", {}),
        Span("s3", "s1", "order-service", "db.ping", 8, "ok", {}),
        Span("s4", "s1", "inventory-service", "/inventory/ping", inv_lat, "ok",
             {"gc_pause_ms": rng.randint(200, 1200), "slow": inv_lat > 1500}),
        Span("s5", "s1", "payment-service", "/payment/ping", 15, "ok", {}),
    ]


def _trace_task13_pod_eviction(service: str, rng: _random_module.Random, step: int) -> list[Span]:
    """payment-service pod evicted — no healthy endpoints."""
    return [
        Span("s1", "", "api-gateway", "POST /api/orders", 5100, "error",
             {"http.status": 503}),
        Span("s2", "s1", "order-service", "processPayment", 5050, "error",
             {"error": "NoHealthyUpstream: payment-service"}),
        Span("s3", "s2", "order-service", "dns.resolve(payment-service)", 5000, "error",
             {"error": "no endpoints available", "pods_running": "0/3",
              "reason": "Evicted: node memory pressure"}),
    ]


def _trace_task14_cascading_timeout(service: str, rng: _random_module.Random, step: int) -> list[Span]:
    """inventory-service slow query, gateway times out before response."""
    inv_lat = 24800 + rng.randint(-500, 1000)
    base_lat = 45 + rng.randint(-5, 5)
    return [
        Span("s1", "", "api-gateway", "GET /api/orders/ORD-1234", 10000, "error",
             {"http.status": 504, "upstream_timeout": "10s"}),
        Span("s2", "s1", "order-service", "getOrderDetails", inv_lat + 300, "ok",
             {"note": "completed_but_gateway_already_timed_out"}),
        Span("s3", "s2", "inventory-service", "getStockLevel", inv_lat, "ok",
             {"slow_query": True, "missing_index": "inventory.sku"}),
        Span("s4", "s3", "inventory-db", "SELECT * FROM inventory WHERE sku LIKE ...",
             inv_lat - 200, "ok",
             {"scan_type": "Seq Scan", "rows_scanned": 5000000}),
        Span("s5", "s2", "payment-service", "getPaymentStatus", base_lat, "ok", {}),
    ]


def _trace_task15_secret_rotation(service: str, rng: _random_module.Random, step: int) -> list[Span]:
    """payment-service external gateway call returns 401."""
    base_lat = 45 + rng.randint(-5, 5)
    return [
        Span("s1", "", "api-gateway", "POST /api/orders", 250, "error",
             {"http.status": 503}),
        Span("s2", "s1", "order-service", "processPayment", 200, "error",
             {"error": "PaymentGatewayAuthFailed"}),
        Span("s3", "s2", "payment-service", "chargeCard", 150, "error",
             {"error": "401 Unauthorized from gateway.payments.com",
              "api_key_version": "v2", "vault_current_version": "v3"}),
        Span("s4", "s3", "payment-service", "http.POST(gateway.payments.com/charge)", 100, "error",
             {"http.status": 401, "error": "Invalid API key"}),
    ]


def _trace_task16_log_storm(service: str, rng: _random_module.Random, step: int) -> list[Span]:
    """auth-service spans show high latency from log serialization overhead."""
    auth_lat = 3500 + rng.randint(-200, 500)
    base_lat = 45 + rng.randint(-5, 5)
    return [
        Span("s1", "", "api-gateway", "POST /api/login", auth_lat + 200, "ok",
             {"http.status": 200, "slow": True}),
        Span("s2", "s1", "auth-service", "authenticateUser", auth_lat, "ok",
             {"cpu_percent": 95, "log_level": "DEBUG", "log_lines_per_sec": 50000}),
        Span("s3", "s2", "auth-service", "log.serialize(DEBUG)", auth_lat - 500, "ok",
             {"overhead_percent": 70, "volume_gb_hr": 50}),
        Span("s4", "s2", "user-db", "SELECT * FROM users", base_lat, "ok", {}),
    ]


def _default_trace(service: str, rng: _random_module.Random, step: int) -> list[Span]:
    """Fallback: healthy trace."""
    lat = 40 + rng.randint(-5, 5)
    return [
        Span("s1", "", service, "handleRequest", lat, "ok", {}),
    ]


_GENERATORS = {
    "task1_memory_leak": _trace_task1_memory_leak,
    "task2_db_cascade": _trace_task2_db_cascade,
    "task3_race_condition": _trace_task3_race_condition,
    "task4_dns_failure": _trace_task4_dns_failure,
    "task5_cert_expiry": _trace_task5_cert_expiry,
    "task6_network_partition": _trace_task6_network_partition,
    "task7_kafka_lag": _trace_task7_kafka_lag,
    "task8_redis_failover": _trace_task8_redis_failover,
    "task9_disk_full": _trace_task9_disk_full,
    "task10_rate_limit": _trace_task10_rate_limit,
    "task11_db_migration_lock": _trace_task11_db_migration_lock,
    "task12_health_flap": _trace_task12_health_flap,
    "task13_pod_eviction": _trace_task13_pod_eviction,
    "task14_cascading_timeout": _trace_task14_cascading_timeout,
    "task15_secret_rotation": _trace_task15_secret_rotation,
    "task16_log_storm": _trace_task16_log_storm,
}
