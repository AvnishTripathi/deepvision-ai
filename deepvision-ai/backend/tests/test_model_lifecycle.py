"""
DeepVision AI — Model Lifecycle & Concurrency Test Suite (Step 3, Task 13)
Validates the production model lifecycle:
- Application startup -> Load model -> Keep detector available
- Sequential requests -> Reuses pre-loaded detector (load_count == 1)
- Concurrent requests -> Handled thread-safely without race conditions
- Lazy-loading safety -> Loaded exactly once across concurrent threads
- Application shutdown -> Cleanly unloads resources
"""

import os
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.services.ai_service import (
    AIService,
    get_ai_service,
    init_ai_service,
    shutdown_ai_service,
)
from app.core.config import settings

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample_face.jpg")


@pytest.fixture
def sample_bytes() -> bytes:
    assert os.path.exists(FIXTURE_PATH), f"Fixture not found at {FIXTURE_PATH}"
    with open(FIXTURE_PATH, "rb") as f:
        return f.read()


# -------------------------------------------------------------------------
# Test 1: Model Reused Across Multiple Sequential Requests (Load Once)
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_model_reused_across_sequential_requests(sample_bytes):
    """
    Verify the model is NOT reloaded for every request.
    Executes multiple requests and asserts:
    - load_count remains exactly 1.
    - inference_count increments sequentially.
    """
    ai_service = get_ai_service()
    initial_load_count = ai_service.detector.load_count
    assert initial_load_count == 1, "Model should be loaded exactly once"
    initial_inf_count = ai_service.detector.inference_count

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Run 5 consecutive requests
        for i in range(5):
            files = {"file": (f"test_{i}.jpg", sample_bytes, "image/jpeg")}
            resp = await client.post("/api/v1/ai/detect", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["face_count"] >= 1

    # Verify model was NOT reloaded: load_count remains unchanged!
    assert ai_service.detector.load_count == initial_load_count
    # Verify inference was executed for each request
    assert ai_service.detector.inference_count == initial_inf_count + 5


# -------------------------------------------------------------------------
# Test 2: Concurrent Requests Handled Correctly & Thread-Safely
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_concurrent_detection_requests_handled_safely(sample_bytes):
    """
    Verify simultaneous concurrent requests execute without race conditions:
    - Sends 10 concurrent requests using asyncio.gather.
    - All requests succeed with HTTP 200.
    - load_count remains 1.
    - Detector lock ensures mutual exclusion during forward passes.
    """
    ai_service = get_ai_service()
    start_load_count = ai_service.detector.load_count

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async def send_req(idx: int):
            files = {"file": (f"concurrent_{idx}.jpg", sample_bytes, "image/jpeg")}
            resp = await client.post("/api/v1/ai/detect", files=files)
            return resp

        # Dispatch 10 concurrent requests simultaneously
        tasks = [send_req(i) for i in range(10)]
        responses = await asyncio.gather(*tasks)

        for resp in responses:
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            assert body["face_count"] >= 1
            assert len(body["faces"]) == body["face_count"]

    # Model was never reloaded during concurrent execution
    assert ai_service.detector.load_count == start_load_count


# -------------------------------------------------------------------------
# Test 3: Thread-Safe Singleton Initialization (No Duplicate Loads Under Race)
# -------------------------------------------------------------------------

def test_thread_safe_singleton_lazy_initialization():
    """
    Simulate a thundering herd of 10 concurrent worker threads attempting
    to initialize get_ai_service() simultaneously.
    Asserts:
    - Exactly one singleton instance is created.
    - All threads receive the identical instance object.
    """
    # Temporarily reset singleton to test race conditions
    shutdown_ai_service()

    instances = []

    def worker():
        inst = get_ai_service()
        instances.append(inst)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(instances) == 10
    first_instance = instances[0]
    for inst in instances:
        assert inst is first_instance, "All threads must receive identical singleton instance"

    assert first_instance.detector.load_count == 1, "Model weights must only be loaded once"


# -------------------------------------------------------------------------
# Test 4: Application Lifespan Clean Shutdown
# -------------------------------------------------------------------------

def test_ai_service_clean_shutdown():
    """
    Verify shutdown_ai_service unloads the detector session and resets state cleanly.
    """
    service = get_ai_service()
    assert service.is_operational is True

    shutdown_ai_service()

    # Accessing get_ai_service again safely reinitializes on demand
    new_service = get_ai_service()
    assert new_service is not None
    assert new_service.is_operational is True
