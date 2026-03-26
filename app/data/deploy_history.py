"""
Synthetic deploy history with git-like deploy records.
"""
import random as _random_module
from typing import Optional
from pydantic import BaseModel


class DeployRecord(BaseModel):
    deploy_id: str
    service: str
    author: str
    timestamp: str  # ISO format
    minutes_ago: int  # relative to incident start
    commit_hash: str  # 7-char hex
    description: str
    config_diff: Optional[dict] = None


class DeployHistory:
    """Provides synthetic deploy records. The critical deploy for task 3 is always included."""

    @classmethod
    def get_deploys(cls, seed: int, last_n: int = 6, service_filter: Optional[str] = None) -> list[DeployRecord]:
        rng = _random_module.Random(seed)

        deploys = [
            # CRITICAL deploy for task 3 — inventory-service config change
            DeployRecord(
                deploy_id="deploy-a1b2c3",
                service="inventory-service",
                author="alice@company.com",
                timestamp="2026-03-26T02:48:00Z",
                minutes_ago=12,
                commit_hash="a1b2c3d",
                description="Update Redis cache config for performance",
                config_diff={
                    "redis.lock_timeout_ms": {"old": 5000, "new": 500},
                    "redis.connection_pool_size": {"old": 10, "new": 50},
                },
            ),
            # Red herring deploys
            DeployRecord(
                deploy_id="deploy-e4f5g6",
                service="auth-service",
                author="bob@company.com",
                timestamp="2026-03-26T01:30:00Z",
                minutes_ago=90,
                commit_hash="e4f5g6h",
                description="Update OAuth token expiry to 24h",
                config_diff=None,
            ),
            DeployRecord(
                deploy_id="deploy-h7i8j9",
                service="api-gateway",
                author="carol@company.com",
                timestamp="2026-03-26T00:15:00Z",
                minutes_ago=165,
                commit_hash="h7i8j9k",
                description="Add rate limiting middleware v2",
                config_diff=None,
            ),
            DeployRecord(
                deploy_id="deploy-k0l1m2",
                service="order-service",
                author="dave@company.com",
                timestamp="2026-03-25T22:00:00Z",
                minutes_ago=300,
                commit_hash="k0l1m2n",
                description="Fix order validation edge case for bulk orders",
                config_diff=None,
            ),
            DeployRecord(
                deploy_id="deploy-n3o4p5",
                service="payment-service",
                author="eve@company.com",
                timestamp="2026-03-25T20:30:00Z",
                minutes_ago=390,
                commit_hash="n3o4p5q",
                description="Upgrade Stripe SDK to v12.1.0",
                config_diff=None,
            ),
            DeployRecord(
                deploy_id="deploy-q6r7s8",
                service="user-db",
                author="frank@company.com",
                timestamp="2026-03-25T18:00:00Z",
                minutes_ago=540,
                commit_hash="q6r7s8t",
                description="Run database migration: add index on email column",
                config_diff=None,
            ),
        ]

        # Sort by minutes_ago ascending (most recent first)
        deploys.sort(key=lambda d: d.minutes_ago)

        if service_filter:
            deploys = [d for d in deploys if d.service == service_filter]

        return deploys[:last_n]
