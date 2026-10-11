"""
DeepVision AI — Standalone Redis Background Worker Process
Usage:
  python -m app.workers.video_worker

Listens on the Redis job queue:
  deepvision:queue:video_jobs
and processes asynchronous video face detection, tracking, and annotation.
"""

import sys
import time
import signal
import logging
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.redis import get_redis_client
from app.services.video_job_manager import RedisVideoJobManager
from app.services.ai_service import init_ai_service

logger = logging.getLogger("deepvision.worker")


def run_worker() -> None:
    setup_logging()
    logger.info("Starting DeepVision AI Background Video Worker...")

    redis_client = get_redis_client()
    if not redis_client:
        logger.error("REDIS_URL is not configured or Redis is unreachable. Exiting.")
        sys.exit(1)

    try:
        redis_client.ping()
    except Exception as exc:
        logger.error("Cannot connect to Redis at %s: %s", settings.REDIS_URL, exc)
        sys.exit(1)

    # Preload AI models in the worker process
    logger.info("Pre-loading AI computer vision models in worker process...")
    init_ai_service()

    job_manager = RedisVideoJobManager()
    logger.info("Worker subscribed to queue: %s (Concurrency: %d)", job_manager.QUEUE_KEY, settings.VIDEO_MAX_CONCURRENT_JOBS)

    stop_event = False

    def handle_signal(sig, frame):
        nonlocal stop_event
        logger.info("Received termination signal (%s). Shutting down worker gracefully...", sig)
        stop_event = True
        job_manager._worker_running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while not stop_event:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break

    logger.info("DeepVision AI Background Video Worker terminated.")


if __name__ == "__main__":
    run_worker()

