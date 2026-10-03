"""
DeepVision AI — Redis Engine & Client Management
Centralized Redis connection pooling, connectivity diagnostics, and foundation
for upcoming background jobs, video processing queues, caching, and real-time streams.
"""

import time
import logging
from typing import Optional, Dict, Any
import redis
from app.core.config import settings

logger = logging.getLogger("deepvision.redis")

# Global connection pool
_redis_pool: Optional[redis.ConnectionPool] = None


def get_redis_pool() -> Optional[redis.ConnectionPool]:
    """Initializes or returns the singleton Redis connection pool."""
    global _redis_pool
    if _redis_pool is None and settings.REDIS_URL:
        try:
            _redis_pool = redis.ConnectionPool.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
                max_connections=20,
            )
        except Exception as exc:
            logger.warning("Failed to initialize Redis connection pool: %s", exc)
            _redis_pool = None
    return _redis_pool


def get_redis_client() -> Optional[redis.Redis]:
    """
    Returns a Redis client using the shared connection pool.
    Returns None if Redis is not configured.
    """
    pool = get_redis_pool()
    if pool:
        return redis.Redis(connection_pool=pool)
    elif settings.REDIS_URL:
        return redis.Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_timeout=1.0,
            socket_connect_timeout=1.0,
        )
    return None


def check_redis_connection() -> Dict[str, Any]:
    """
    Verifies Redis server connectivity and measures ping latency.
    Designed for health checks and readiness probes.
    """
    if not settings.REDIS_URL:
        return {
            "status": "disabled",
            "message": "REDIS_URL is not configured",
        }

    start_time = time.perf_counter()
    try:
        client = get_redis_client()
        if not client:
            return {
                "status": "error",
                "message": "Unable to initialize Redis client",
                "target_url": settings.REDIS_URL,
            }

        # Issue PING command
        is_alive = client.ping()
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        if is_alive:
            return {
                "status": "connected",
                "latency_ms": latency_ms,
                "target_url": settings.REDIS_URL,
            }
        else:
            return {
                "status": "unresponsive",
                "latency_ms": latency_ms,
                "target_url": settings.REDIS_URL,
            }

    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.warning("Redis connectivity check failed: %s", exc)
        return {
            "status": "offline",
            "error": str(exc),
            "target_url": settings.REDIS_URL,
            "latency_ms": latency_ms,
        }


# ==============================================================================
# Architectural Hooks Prepared for Future Phases:
# - Background Jobs (e.g. Celery / ARQ / Redis Queue worker integration)
# - Video Processing Pipelines (frame pub/sub, chunk streaming buffers)
# - Caching Layers (detection result cache, session cache)
# - Real-time vision message broker (Websocket broadcast, channel subscriptions)
# ==============================================================================
