"""
Comprehensive tests for SREBench environment.
Uses httpx.AsyncClient with app=app for all tests (no server needed).

Tests the OpenEnv-standard API format:
  POST /reset  → {"observation": {...}, "reward": null, "done": false}
  POST /step   → {"observation": {...}, "reward": float, "done": bool}
"""
import pytest
import json
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.models import SREAction, SREObservation
from app.environment import SREBenchEnvironment

transport = ASGITransport(app=app)


# ─── Helper ────────────────────────────────────────────

async def reset_task(client: AsyncClient, task_id: str, seed: int = 42) -> dict:
    resp = await client.post("/reset", json={"task_id": task_id, "seed": seed})
    assert resp.status_code == 200
    data = resp.json()
    return data["observation"]


async def do_step(client: AsyncClient, action_type: str, parameters: dict = None,
                  reasoning: str = "Testing this action for investigation") -> dict:
    resp = await client.post("/step", json={
        "action": {
            "action_type": action_type,
            "parameters": parameters or {},
            "reasoning": reasoning,
        }
    })
    assert resp.status_code == 200
    return resp.json()


# ─── 1. Reset Tests ───────────────────────────────────

@pytest.mark.asyncio
async def test_reset_task1():
    """Reset returns valid Observation with all 8 services."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        obs = await reset_task(client, "task1_memory_leak")
        assert len(obs["service_statuses"]) == 8
        assert obs["step_count"] == 0
        assert obs["incident_phase"] == "investigating"
        assert "order-service" in obs["service_statuses"]
        assert obs["service_statuses"]["order-service"]["status"] == "down"


@pytest.mark.asyncio
async def test_reset_task2():
    """Reset returns api-gateway as degraded."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        obs = await reset_task(client, "task2_db_cascade")
        assert obs["service_statuses"]["api-gateway"]["status"] == "degraded"
        assert obs["service_statuses"]["payment-service"]["status"] == "down"


@pytest.mark.asyncio
async def test_reset_task3():
    """Reset returns multiple degraded services."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        obs = await reset_task(client, "task3_race_condition")
        degraded = [s for s, v in obs["service_statuses"].items() if v["status"] == "degraded"]
        assert len(degraded) >= 2


# ─── 2. Step requires reset ──────────────────────────────

@pytest.mark.asyncio
async def test_step_requires_reset():
    """Calling step without reset raises error."""
    env = SREBenchEnvironment()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(SREAction(
            action_type="list_services",
            parameters={},
            reasoning="Testing without reset"
        ))


# ─── 3. Task 1 Optimal Path ──────────────────────────────

@pytest.mark.asyncio
async def test_task1_optimal_path():
    """Full optimal episode — score >= 0.85."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await reset_task(client, "task1_memory_leak")

        # list_services
        await do_step(client, "list_services")

        # read_logs on order-service
        await do_step(client, "read_logs", {"service": "order-service"},
                      "Reading logs for order-service to find error cause")

        # check_metrics on order-service
        await do_step(client, "check_metrics",
                      {"service": "order-service", "metric": "memory"},
                      "Checking memory metrics to confirm OOM suspicion")

        # apply_fix
        await do_step(client, "apply_fix",
                      {"service": "order-service", "fix_type": "restart"},
                      "Restarting order-service to resolve OOM kill issue")

        # verify_health
        result = await do_step(client, "verify_health",
                               {"service": "order-service"},
                               "Verifying order-service is healthy after restart")

        # Check if resolved
        assert result["done"] is True
        assert result["observation"]["episode_score"] >= 0.85


# ─── 4. Wrong service penalty ────────────────────────────

@pytest.mark.asyncio
async def test_task1_wrong_service_penalty():
    """Apply fix to wrong service gets negative reward."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await reset_task(client, "task1_memory_leak")

        # Investigate enough to unlock fix attempts
        await do_step(client, "list_services")
        await do_step(client, "check_alerts")

        result = await do_step(client, "apply_fix",
                               {"service": "auth-service", "fix_type": "restart"},
                               "Applying fix to auth-service to test wrong service penalty")

        assert result["reward"] <= -0.05


# ─── 5. Task 2 naive fix capped ─────────────────────────

@pytest.mark.asyncio
async def test_task2_naive_fix_capped():
    """Restart api-gateway without tracing — grader score <= 0.35."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await reset_task(client, "task2_db_cascade")

        # Investigate then fix wrong service
        await do_step(client, "list_services")
        await do_step(client, "check_alerts")
        await do_step(client, "apply_fix",
                      {"service": "api-gateway", "fix_type": "restart"},
                      "Restarting api-gateway directly without tracing root cause")

        # Get grader score
        grader_resp = await client.get("/grader")
        assert grader_resp.status_code == 200
        score = grader_resp.json()["episode_score"]
        assert score <= 0.35


# ─── 6. Task 3 needs deploy check ───────────────────────

@pytest.mark.asyncio
async def test_task3_needs_deploy_check():
    """Score < 0.30 if check_deployments never called."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await reset_task(client, "task3_race_condition")

        # Try to fix without checking deploys
        await do_step(client, "list_services")
        await do_step(client, "check_metrics",
                      {"service": "inventory-service", "metric": "error_rate"},
                      "Checking error rate on inventory service for investigation")
        await do_step(client, "apply_fix",
                      {"service": "inventory-service", "fix_type": "restart"},
                      "Restarting inventory service to try to resolve errors")

        grader_resp = await client.get("/grader")
        score = grader_resp.json()["episode_score"]
        assert score <= 0.30


# ─── 7. Grader endpoint ─────────────────────────────────

@pytest.mark.asyncio
async def test_grader_endpoint():
    """GET /grader returns float in (0, 1)."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await reset_task(client, "task1_memory_leak")
        await do_step(client, "list_services")

        resp = await client.get("/grader")
        assert resp.status_code == 200
        score = resp.json()["episode_score"]
        assert 0.0 < score < 1.0


# ─── 8. Tasks endpoint ──────────────────────────────────

@pytest.mark.asyncio
async def test_tasks_endpoint():
    """GET /tasks returns 3 tasks with correct ids."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["tasks"]) == 16
        ids = {t["id"] for t in data["tasks"]}
        assert ids == {"task1_memory_leak", "task2_db_cascade", "task3_race_condition",
                       "task4_dns_failure", "task5_cert_expiry", "task6_network_partition",
                       "task7_kafka_lag", "task8_redis_failover", "task9_disk_full",
                       "task10_rate_limit", "task11_db_migration_lock", "task12_health_flap",
                       "task13_pod_eviction", "task14_cascading_timeout",
                       "task15_secret_rotation", "task16_log_storm"}

# ─── 8b. Task 4 Reset ───────────────────────────────────

@pytest.mark.asyncio
async def test_reset_task4():
    """Task 4 reset returns degraded auth-service."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        obs = await reset_task(client, "task4_dns_failure")
        assert obs["service_statuses"]["auth-service"]["status"] == "degraded"
        assert obs["service_statuses"]["user-db"]["status"] == "healthy"


# ─── 8c. Task 4 Optimal Path ────────────────────────────

@pytest.mark.asyncio
async def test_task4_optimal_path():
    """Full optimal episode for DNS failure — score >= 0.80."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await reset_task(client, "task4_dns_failure")
        await do_step(client, "list_services")
        await do_step(client, "read_logs", {"service": "auth-service"},
                      "Reading auth-service logs to find DNS errors")
        await do_step(client, "run_diagnostic", {"service": "auth-service", "type": "dns"},
                      "Running DNS diagnostic on auth-service")
        await do_step(client, "apply_fix",
                      {"service": "auth-service", "fix_type": "flush_dns"},
                      "Flushing stale DNS cache on auth-service")
        result = await do_step(client, "verify_health", {},
                               "Verifying auth-service health after DNS fix")
        assert result["done"] is True
        assert result["observation"]["episode_score"] >= 0.80


# ─── 8d. Task 5 Reset ───────────────────────────────────

@pytest.mark.asyncio
async def test_reset_task5():
    """Task 5 reset returns critical payment-service."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        obs = await reset_task(client, "task5_cert_expiry")
        assert obs["service_statuses"]["payment-service"]["status"] == "down"
        assert obs["service_statuses"]["order-service"]["status"] == "degraded"


# ─── 8e. Task 5 Optimal Path ────────────────────────────

@pytest.mark.asyncio
async def test_task5_optimal_path():
    """Full optimal episode for cert expiry — score >= 0.80."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await reset_task(client, "task5_cert_expiry")
        await do_step(client, "read_logs", {"service": "payment-service"},
                      "Reading payment-service logs to find TLS errors")
        await do_step(client, "run_diagnostic",
                      {"service": "payment-service", "type": "tls"},
                      "Running TLS diagnostic on payment-service")
        await do_step(client, "apply_fix",
                      {"service": "payment-service", "fix_type": "renew_cert"},
                      "Renewing expired TLS certificate on payment-service")
        result = await do_step(client, "verify_health", {},
                               "Verifying payment-service health after cert renewal")
        assert result["done"] is True
        assert result["observation"]["episode_score"] >= 0.80


# ─── 8f. Task 6 Reset ───────────────────────────────────

@pytest.mark.asyncio
async def test_reset_task6():
    """Task 6 reset returns degraded inventory-service."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        obs = await reset_task(client, "task6_network_partition")
        assert obs["service_statuses"]["inventory-service"]["status"] == "degraded"
        assert obs["service_statuses"]["order-service"]["status"] == "degraded"


# ─── 8g. Task 6 Optimal Path ────────────────────────────

@pytest.mark.asyncio
async def test_task6_optimal_path():
    """Full optimal episode for network partition — score >= 0.70."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await reset_task(client, "task6_network_partition")
        await do_step(client, "read_logs", {"service": "inventory-service"},
                      "Reading inventory-service logs to find partition evidence")
        await do_step(client, "check_deployments", {},
                      "Checking recent deploys for network changes")
        await do_step(client, "run_diagnostic",
                      {"service": "inventory-service", "type": "network"},
                      "Running network diagnostic on inventory-service")
        await do_step(client, "apply_fix",
                      {"fix_type": "rollback_deploy"},
                      "Rolling back the iptables deploy to fix network partition")
        await do_step(client, "apply_fix",
                      {"service": "inventory-service", "fix_type": "reconcile_data"},
                      "Reconciling stale cached data on inventory-service")
        result = await do_step(client, "verify_health", {},
                               "Verifying full system health after partition fix")
        assert result["done"] is True
        assert result["observation"]["episode_score"] >= 0.70

# ─── 9. Reproducibility ─────────────────────────────────

@pytest.mark.asyncio
async def test_reproducibility():
    """Same seed produces identical observations."""
    async def run_episode(client):
        obs = await reset_task(client, "task1_memory_leak", seed=42)
        result1 = await do_step(client, "list_services")
        result2 = await do_step(client, "check_metrics",
                                {"service": "order-service", "metric": "memory"},
                                "Checking memory metrics for reproducibility test")
        return [obs, result1, result2]

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        run1 = await run_episode(client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        run2 = await run_episode(client)

    assert json.dumps(run1, sort_keys=True) == json.dumps(run2, sort_keys=True)


# ─── 10. Reward bounds ──────────────────────────────────

@pytest.mark.asyncio
async def test_reward_bounds():
    """All step rewards are in [-0.25, +0.35]."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await reset_task(client, "task1_memory_leak")

        actions = [
            ("list_services", {}),
            ("read_logs", {"service": "order-service"}),
            ("check_metrics", {"service": "order-service", "metric": "memory"}),
            ("check_alerts", {}),
            ("apply_fix", {"service": "order-service", "fix_type": "restart"}),
        ]

        for action_type, params in actions:
            result = await do_step(client, action_type, params,
                                   f"Testing {action_type} for reward bounds check")
            reward = result["reward"]
            assert -0.25 <= reward <= 0.35, f"Reward {reward} out of bounds for {action_type}"


# ─── 11. Health endpoint ────────────────────────────────

@pytest.mark.asyncio
async def test_health_endpoint():
    """GET /health returns 200."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


# ─── 12. State endpoint ─────────────────────────────────

@pytest.mark.asyncio
async def test_state_endpoint():
    """GET /state returns valid dict even before reset."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/state")
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict)


# ─── 13. OpenEnv schema endpoint ────────────────────────

@pytest.mark.asyncio
async def test_schema_endpoint():
    """GET /schema returns action and observation schemas."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert "action" in data
        assert "observation" in data


# ─── 14. OpenEnv metadata endpoint ──────────────────────

@pytest.mark.asyncio
async def test_metadata_endpoint():
    """GET /metadata returns environment metadata."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "SREBench"
