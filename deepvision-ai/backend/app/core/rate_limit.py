"""
DeepVision AI — API Rate Limiter
Provides sliding-window rate limiting for expensive AI detection and video tracking endpoints.
Supports Redis storage when available, with automatic thread-safe in-memory fallback.
"""

import time
import logging
import threading
from typing import Optional, Tuple, Dict, List
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.redis import get_redis_client

logger = logging.getLogger("deepvision.security.rate_limit")


def parse_rate_limit(limit_str: str) -> Tuple[int, int]:
    """
    Parses rate limit specification string into (max_requests, window_seconds).
    Examples:
        '60/minute' -> (60, 60)
        '10/minute' -> (10, 60)
        '5/second'  -> (5, 1)
        '100/hour'  -> (100, 3600)
        '30'        -> (30, 60)
    """
    if not limit_str:
        return 60, 60

    parts = str(limit_str).strip().lower().split("/")
    max_requests = int(parts[0]) if parts[0].isdigit() else 60

    if len(parts) < 2:
        return max_requests, 60

    unit = parts[1].strip()
    if unit in ("s", "sec", "second", "seconds"):
        window = 1
    elif unit in ("m", "min", "minute", "minutes"):
        window = 60
    elif unit in ("h", "hr", "hour", "hours"):
        window = 3600
    elif unit in ("d", "day", "days"):
        window = 86400
    else:
        window = 60

    return max_requests, window


class InMemorySlidingWindowLimiter:
    """Thread-safe in-memory sliding window rate limiter."""

    def __init__(self):
        self._lock = threading.Lock()
        self._records: Dict[str, List[float]] = {}
        self._last_cleanup = time.time()

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> Tuple[bool, int, int]:
        """
        Evaluates whether request under key is allowed.
        Returns:
            Tuple[is_allowed: bool, remaining: int, retry_after: int]
        """
        now = time.time()
        window_start = now - window_seconds

        with self._lock:
            # Periodic cleanup of expired keys (every 60s)
            if now - self._last_cleanup > 60:
                self._cleanup(now)
                self._last_cleanup = now

            timestamps = self._records.get(key, [])
            # Filter out entries outside the sliding window
            valid_timestamps = [t for t in timestamps if t > window_start]

            if len(valid_timestamps) >= max_requests:
                oldest = valid_timestamps[0]
                retry_after = max(1, int(oldest + window_seconds - now))
                self._records[key] = valid_timestamps
                return False, 0, retry_after

            valid_timestamps.append(now)
            self._records[key] = valid_timestamps
            remaining = max(0, max_requests - len(valid_timestamps))
            return True, remaining, 0

    def _cleanup(self, now: float) -> None:
        """Removes stale keys older than 1 hour."""
        cutoff = now - 3600
        stale_keys = [k for k, timestamps in self._records.items() if not timestamps or timestamps[-1] < cutoff]
        for k in stale_keys:
            self._records.pop(k, None)

    def reset(self) -> None:
        """Clears all records for testing purposes."""
        with self._lock:
            self._records.clear()


# Global in-memory fallback instance
_in_memory_limiter = InMemorySlidingWindowLimiter()


def get_client_ip(request: Request) -> str:
    """Extracts client IP, respecting X-Forwarded-For if present."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
        if client_ip:
            return client_ip
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


class RateLimiter:
    """FastAPI route dependency for sliding-window rate limiting."""

    def __init__(self, endpoint_name: str, limit_setting_getter):
        self.endpoint_name = endpoint_name
        self.limit_setting_getter = limit_setting_getter

    async def __call__(self, request: Request) -> None:
        if not getattr(settings, "RATE_LIMIT_ENABLED", True):
            return

        limit_str = self.limit_setting_getter()
        max_requests, window_seconds = parse_rate_limit(limit_str)

        client_ip = get_client_ip(request)
        rate_key = f"rate_limit:{self.endpoint_name}:{client_ip}"

        # Attempt Redis sliding window
        redis_client = get_redis_client()
        if redis_client:
            try:
                now = time.time()
                window_start = now - window_seconds
                pipe = redis_client.pipeline()
                pipe.zremrangebyscore(rate_key, 0, window_start)
                pipe.zcard(rate_key)
                pipe.zadd(rate_key, {str(now): now})
                pipe.expire(rate_key, window_seconds + 5)
                results = pipe.execute()

                current_count = results[1]
                if current_count >= max_requests:
                    # Remove the timestamp we just tentatively added
                    redis_client.zrem(rate_key, str(now))
                    retry_after = window_seconds
                    logger.warning(
                        "Rate limit exceeded on %s for client %s (%d/%ds)",
                        self.endpoint_name,
                        client_ip,
                        max_requests,
                        window_seconds,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                            "success": False,
                            "error": "rate_limit_exceeded",
                            "message": f"Rate limit of {max_requests} requests per {window_seconds}s exceeded for {self.endpoint_name}.",
                            "retry_after": retry_after,
                        },
                        headers={"Retry-After": str(retry_after)},
                    )
                return
            except HTTPException:
                raise
            except Exception as exc:
                logger.debug("Redis rate limiting error (%s), using in-memory limiter", exc)

        # In-Memory fallback
        allowed, remaining, retry_after = _in_memory_limiter.is_allowed(
            rate_key, max_requests, window_seconds
        )

        if not allowed:
            logger.warning(
                "Rate limit exceeded on %s for client %s (in-memory, retry in %ds)",
                self.endpoint_name,
                client_ip,
                retry_after,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "success": False,
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit of {max_requests} requests per {window_seconds}s exceeded for {self.endpoint_name}.",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )


# Factory functions for route dependencies
rate_limit_image_analysis = RateLimiter(
    "image_analysis",
    lambda: getattr(settings, "RATE_LIMIT_IMAGE_ANALYSIS", "60/minute"),
)

rate_limit_video_analysis = RateLimiter(
    "video_analysis",
    lambda: getattr(settings, "RATE_LIMIT_VIDEO_ANALYSIS", "10/minute"),
)

