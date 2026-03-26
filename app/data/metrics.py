"""
Time-series metric simulator. All values seeded for reproducibility.
"""
import random as _random_module
from datetime import datetime, timedelta


class MetricsSimulator:
    """Returns metrics and time-series data for each service and scenario."""

    @classmethod
    def get_error_rate(cls, service: str, scenario: str, seed: int) -> float:
        rng = _random_module.Random(seed + hash(service))
        if scenario == "task1_memory_leak":
            if service == "order-service":
                return round(rng.uniform(0.45, 0.65), 2)
            return round(rng.uniform(0.0, 0.02), 3)
        elif scenario == "task2_db_cascade":
            rates = {
                "payment-db": round(rng.uniform(0.85, 0.95), 2),
                "payment-service": round(rng.uniform(0.82, 0.92), 2),
                "order-service": round(rng.uniform(0.55, 0.68), 2),
                "api-gateway": round(rng.uniform(0.50, 0.62), 2),
            }
            return rates.get(service, round(rng.uniform(0.0, 0.02), 3))
        elif scenario == "task3_race_condition":
            affected = {
                "inventory-service": round(rng.uniform(0.30, 0.45), 2),
                "order-service": round(rng.uniform(0.18, 0.28), 2),
                "api-gateway": round(rng.uniform(0.15, 0.25), 2),
            }
            return affected.get(service, round(rng.uniform(0.0, 0.02), 3))
        elif scenario == "task4_dns_failure":
            rates = {
                "auth-service": round(rng.uniform(0.38, 0.48), 2),
                "api-gateway": round(rng.uniform(0.25, 0.35), 2),
            }
            return rates.get(service, round(rng.uniform(0.0, 0.02), 3))
        elif scenario == "task5_cert_expiry":
            rates = {
                "payment-service": round(rng.uniform(0.90, 0.98), 2),
                "order-service": round(rng.uniform(0.60, 0.72), 2),
                "payment-db": round(rng.uniform(0.85, 0.95), 2),
                "api-gateway": round(rng.uniform(0.40, 0.50), 2),
            }
            return rates.get(service, round(rng.uniform(0.0, 0.02), 3))
        elif scenario == "task6_network_partition":
            rates = {
                "inventory-service": round(rng.uniform(0.03, 0.08), 2),
                "order-service": round(rng.uniform(0.08, 0.15), 2),
            }
            return rates.get(service, round(rng.uniform(0.0, 0.02), 3))
        return round(rng.uniform(0.0, 0.02), 3)

    @classmethod
    def get_latency_p99(cls, service: str, scenario: str, seed: int) -> int:
        rng = _random_module.Random(seed + hash(service) + 1)
        if scenario == "task1_memory_leak":
            if service == "order-service":
                return rng.randint(8000, 15000)
            return rng.randint(20, 150)
        elif scenario == "task2_db_cascade":
            slow = {"payment-db": (25000, 35000), "payment-service": (20000, 30000),
                    "order-service": (15000, 25000), "api-gateway": (12000, 20000)}
            if service in slow:
                lo, hi = slow[service]
                return rng.randint(lo, hi)
            return rng.randint(20, 150)
        elif scenario == "task3_race_condition":
            slow = {"inventory-service": (3000, 8000), "order-service": (2000, 5000),
                    "api-gateway": (1500, 4000)}
            if service in slow:
                lo, hi = slow[service]
                return rng.randint(lo, hi)
            return rng.randint(20, 150)
        elif scenario == "task4_dns_failure":
            slow = {"auth-service": (7000, 12000), "api-gateway": (5000, 8000)}
            if service in slow:
                lo, hi = slow[service]
                return rng.randint(lo, hi)
            return rng.randint(20, 150)
        elif scenario == "task5_cert_expiry":
            slow = {"payment-service": (30, 80), "order-service": (10000, 15000),
                    "api-gateway": (8000, 12000)}
            if service in slow:
                lo, hi = slow[service]
                return rng.randint(lo, hi)
            return rng.randint(20, 150)
        elif scenario == "task6_network_partition":
            slow = {"inventory-service": (20, 50), "order-service": (150, 300)}
            if service in slow:
                lo, hi = slow[service]
                return rng.randint(lo, hi)
            return rng.randint(20, 150)
        return rng.randint(20, 150)

    @classmethod
    def get_cpu_percent(cls, service: str, scenario: str, seed: int) -> float:
        rng = _random_module.Random(seed + hash(service) + 2)
        if scenario == "task1_memory_leak" and service == "order-service":
            return round(rng.uniform(75, 95), 1)
        elif scenario == "task4_dns_failure" and service == "auth-service":
            return round(rng.uniform(18, 30), 1)
        elif scenario == "task5_cert_expiry" and service == "payment-service":
            return round(rng.uniform(3, 8), 1)
        elif scenario == "task6_network_partition" and service == "inventory-service":
            return round(rng.uniform(10, 18), 1)
        return round(rng.uniform(10, 45), 1)

    @classmethod
    def get_memory_percent(cls, service: str, scenario: str, seed: int) -> float:
        rng = _random_module.Random(seed + hash(service) + 3)
        if scenario == "task1_memory_leak" and service == "order-service":
            return round(rng.uniform(95, 99), 1)
        elif scenario == "task4_dns_failure" and service == "auth-service":
            return round(rng.uniform(30, 42), 1)
        elif scenario == "task5_cert_expiry" and service == "payment-service":
            return round(rng.uniform(18, 25), 1)
        elif scenario == "task6_network_partition" and service == "inventory-service":
            return round(rng.uniform(50, 62), 1)
        return round(rng.uniform(20, 55), 1)

    @classmethod
    def get_connection_pool_usage(cls, service: str, scenario: str, seed: int) -> float:
        rng = _random_module.Random(seed + hash(service) + 4)
        if scenario == "task2_db_cascade" and service == "payment-db":
            return round(rng.uniform(0.96, 0.99), 2)
        elif scenario == "task4_dns_failure" and service == "auth-service":
            return round(rng.uniform(0.05, 0.18), 2)
        elif scenario == "task5_cert_expiry" and service == "payment-service":
            return round(rng.uniform(0.0, 0.02), 2)
        elif scenario == "task6_network_partition" and service == "inventory-service":
            return round(rng.uniform(0.0, 0.02), 2)
        return round(rng.uniform(0.10, 0.45), 2)

    @classmethod
    def get_timeseries(cls, service: str, metric: str, scenario: str, seed: int,
                       minutes: int = 60) -> list[dict]:
        """Returns list of {timestamp, value} for the last `minutes` minutes."""
        rng = _random_module.Random(seed + hash(service) + hash(metric))
        base_time = datetime(2026, 3, 26, 2, 0, 0)
        points = []

        for m in range(minutes):
            ts = base_time + timedelta(minutes=m)
            ts_str = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

            if scenario == "task3_race_condition" and service == "inventory-service" and metric == "error_rate":
                # Normal for first 48 minutes, spike after (deploy was at minute 48)
                if m < 48:
                    value = round(rng.uniform(0.001, 0.01), 4)
                else:
                    value = round(rng.uniform(0.35, 0.55), 3)
            elif scenario == "task1_memory_leak" and service == "order-service" and metric == "memory":
                # Memory climbing over time
                base_mem = 50 + (m * 0.75)
                value = round(min(99, base_mem + rng.uniform(-2, 2)), 1)
            elif scenario == "task2_db_cascade" and service == "payment-db" and metric == "connection_pool":
                # Pool usage climbing
                base_pool = 0.3 + (m * 0.01)
                value = round(min(0.99, base_pool + rng.uniform(-0.02, 0.02)), 3)
            elif scenario == "task4_dns_failure" and service == "auth-service" and metric == "error_rate":
                if m < 52:
                    value = round(rng.uniform(0.001, 0.01), 4)
                else:
                    value = round(rng.uniform(0.35, 0.50), 3)
            elif scenario == "task5_cert_expiry" and service == "payment-service" and metric == "error_rate":
                if m < 58:
                    value = round(rng.uniform(0.001, 0.01), 4)
                else:
                    value = round(rng.uniform(0.90, 0.98), 3)
            elif scenario == "task6_network_partition" and service == "inventory-service" and metric == "write_error_rate":
                if m < 45:
                    value = round(rng.uniform(0.001, 0.01), 4)
                else:
                    value = round(rng.uniform(0.95, 1.0), 3)
            else:
                value = round(rng.uniform(0.01, 0.15), 3)

            points.append({"timestamp": ts_str, "value": value})

        return points

    @classmethod
    def get_metrics_summary(cls, service: str, scenario: str, seed: int) -> dict:
        """Returns a full metrics summary for a service in a given scenario."""
        return {
            "service": service,
            "error_rate": cls.get_error_rate(service, scenario, seed),
            "latency_p99_ms": cls.get_latency_p99(service, scenario, seed),
            "cpu_percent": cls.get_cpu_percent(service, scenario, seed),
            "memory_percent": cls.get_memory_percent(service, scenario, seed),
            "connection_pool_usage": cls.get_connection_pool_usage(service, scenario, seed),
        }
