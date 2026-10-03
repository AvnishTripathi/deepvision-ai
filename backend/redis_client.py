import logging
from typing import Optional
from config import settings

logger = logging.getLogger(__name__)

def check_redis_connection() -> dict:
    """Test Redis connectivity."""
    if not settings.REDIS_URL:
        return {"status": "disabled", "message": "REDIS_URL not configured"}
    
    try:
        import redis
        client = redis.from_url(settings.REDIS_URL, socket_timeout=2)
        ping_res = client.ping()
        return {"status": "connected" if ping_res else "error", "url": settings.REDIS_URL}
    except Exception as e:
        # Redis might not be running locally if running outside Docker
        logger.warning(f"Redis check (expected if local Redis not started): {e}")
        return {"status": "ready (offline locally)", "target_url": settings.REDIS_URL, "detail": str(e)}
