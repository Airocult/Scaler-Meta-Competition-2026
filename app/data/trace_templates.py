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
}
