"""
Tests for Redis Connection Management and Diagnostics.
"""

from app.core.redis import check_redis_connection, get_redis_client, get_redis_pool
from app.core.config import settings


def test_redis_connection_diagnostic_resilience():
    """Verify check_redis_connection returns structured diagnostic info without raising errors."""
    res = check_redis_connection()
    assert isinstance(res, dict)
    assert "status" in res
    assert res["status"] in ("connected", "offline", "disabled", "error")
    assert "latency_ms" in res or "message" in res


def test_redis_disabled_state(monkeypatch):
    """Verify clean handling when REDIS_URL is not configured."""
    monkeypatch.setattr(settings, "REDIS_URL", None)
    res = check_redis_connection()
    assert res["status"] == "disabled"
    assert "not configured" in res["message"]


def test_redis_pool_singleton():
    """Verify pool and client initialization factory functions."""
    pool = get_redis_pool()
    if settings.REDIS_URL:
        assert pool is not None
        client = get_redis_client()
        assert client is not None
