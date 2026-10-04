"""
DeepVision AI — Image Test Suite Runner (Step 3, Task 10)
Direct verification script executing detection tests A through E:
- Test A — One face (Expected: face_count >= 1)
- Test B — Multiple faces (Expected: face_count >= 2)
- Test C — No face (Expected: face_count = 0)
- Test D — Different image sizes (small and large images)
- Test E — Invalid image (clean API validation errors)

Uses real images and real inference results. Zero mocked or fabricated values.
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from httpx import AsyncClient, ASGITransport

# Add backend directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.main import app

FIXTURES_DIR = Path(BACKEND_DIR) / "tests" / "fixtures"


async def run_scenario_tests():
    transport = ASGITransport(app=app)
    results = {}

    print("=" * 70)
    print(" DeepVision AI — Image Detection Test Scenarios (Step 3, Task 10)")
    print("=" * 70)

    async with AsyncClient(transport=transport, base_url="http://test") as client:

        # -------------------------------------------------------------
        # Test A: One Face
        # -------------------------------------------------------------
        print("\n[TEST A] One Face Scenario...")
        with open(FIXTURES_DIR / "sample_face.jpg", "rb") as f:
            bytes_a = f.read()
        resp_a = await client.post("/api/v1/ai/detect", files={"file": ("one_face.jpg", bytes_a, "image/jpeg")})
        assert resp_a.status_code == 200, f"Expected 200, got {resp_a.status_code}"
        data_a = resp_a.json()
        face_count_a = data_a.get("face_count", 0)
        assert face_count_a >= 1, f"Expected face_count >= 1, got {face_count_a}"
        f1 = data_a["faces"][0]
        box = f1["bbox"]
        print(f"  -> Status: {resp_a.status_code} OK | Faces Detected: {face_count_a} | Latency: {data_a.get('processing_time_ms')} ms")
        print(f"  -> Face #1: Confidence: {f1['confidence']*100:.2f}% | BBox: [{box['x1']}, {box['y1']}, {box['x2']}, {box['y2']}]")
        print(f"  -> Landmarks: {len(f1['landmarks'])} detected ({[lm['name'] for lm in f1['landmarks']]})")
        print("  -> Verdict: PASS (face_count >= 1)")
        results["Test A (One Face)"] = "PASS"

        # -------------------------------------------------------------
        # Test B: Multiple Faces
        # -------------------------------------------------------------
        print("\n[TEST B] Multiple Faces Scenario...")
        with open(FIXTURES_DIR / "multiple_faces.jpg", "rb") as f:
            bytes_b = f.read()
        resp_b = await client.post("/api/v1/ai/detect", files={"file": ("multiple_faces.jpg", bytes_b, "image/jpeg")})
        assert resp_b.status_code == 200, f"Expected 200, got {resp_b.status_code}"
        data_b = resp_b.json()
        face_count_b = data_b.get("face_count", 0)
        assert face_count_b >= 2, f"Expected face_count >= 2, got {face_count_b}"
        print(f"  -> Status: {resp_b.status_code} OK | Faces Detected: {face_count_b} | Resolution: {data_b['image']['width']}x{data_b['image']['height']}")
        for idx, face in enumerate(data_b["faces"]):
            b = face["bbox"]
            print(f"     Face #{idx+1}: Confidence: {face['confidence']*100:.2f}% | BBox: [{b['x1']}, {b['y1']}, {b['x2']}, {b['y2']}]")
        print("  -> Verdict: PASS (face_count >= 2)")
        results["Test B (Multiple Faces)"] = "PASS"

        # -------------------------------------------------------------
        # Test C: No Face
        # -------------------------------------------------------------
        print("\n[TEST C] No Face Scenario (Scenery/Landscape)...")
        with open(FIXTURES_DIR / "no_face.jpg", "rb") as f:
            bytes_c = f.read()
        resp_c = await client.post("/api/v1/ai/detect", files={"file": ("no_face.jpg", bytes_c, "image/jpeg")})
        assert resp_c.status_code == 200, f"Expected 200, got {resp_c.status_code}"
        data_c = resp_c.json()
        face_count_c = data_c.get("face_count", -1)
        assert face_count_c == 0, f"Expected face_count == 0, got {face_count_c}"
        assert data_c.get("faces") == [], f"Expected empty faces list, got {data_c.get('faces')}"
        print(f"  -> Status: {resp_c.status_code} OK | Faces Detected: {face_count_c} | Fake faces: NONE")
        print("  -> Verdict: PASS (face_count = 0)")
        results["Test C (No Face)"] = "PASS"

        # -------------------------------------------------------------
        # Test D: Different Image Sizes
        # -------------------------------------------------------------
        print("\n[TEST D] Different Image Sizes Scenario...")
        # Small image (300x300)
        with open(FIXTURES_DIR / "small_face.jpg", "rb") as f:
            bytes_small = f.read()
        resp_d_small = await client.post("/api/v1/ai/detect", files={"file": ("small_face.jpg", bytes_small, "image/jpeg")})
        assert resp_d_small.status_code == 200
        data_d_small = resp_d_small.json()
        print(f"  -> Small image: {data_d_small['image']['width']}x{data_d_small['image']['height']} | Latency: {data_d_small.get('processing_time_ms')} ms")

        # Large image (1200x900)
        with open(FIXTURES_DIR / "large_face.jpg", "rb") as f:
            bytes_large = f.read()
        resp_d_large = await client.post("/api/v1/ai/detect", files={"file": ("large_face.jpg", bytes_large, "image/jpeg")})
        assert resp_d_large.status_code == 200
        data_d_large = resp_d_large.json()
        assert data_d_large.get("face_count") >= 1
        print(f"  -> Large image: {data_d_large['image']['width']}x{data_d_large['image']['height']} | Faces: {data_d_large.get('face_count')} | Latency: {data_d_large.get('processing_time_ms')} ms")
        print("  -> Verdict: PASS (Both sizes handled reliably)")
        results["Test D (Image Sizes)"] = "PASS"

        # -------------------------------------------------------------
        # Test E: Invalid Image
        # -------------------------------------------------------------
        print("\n[TEST E] Invalid Image Scenarios...")
        # Subtest 1: Unsupported media type (text file)
        resp_e1 = await client.post("/api/v1/ai/detect", files={"file": ("test.txt", b"Invalid image data", "text/plain")})
        assert resp_e1.status_code == 400
        data_e1 = resp_e1.json()
        print(f"  -> Non-image file: Status {resp_e1.status_code} | Error: '{data_e1.get('error')}' | Msg: '{data_e1.get('message')}'")

        # Subtest 2: Corrupted byte stream
        corrupt_bytes = b"\xFF\xD8\xFF\xE0JFIF" + b"\x00\x00corrupted_payload" * 4
        resp_e2 = await client.post("/api/v1/ai/detect", files={"file": ("broken.jpg", corrupt_bytes, "image/jpeg")})
        assert resp_e2.status_code == 400
        data_e2 = resp_e2.json()
        print(f"  -> Corrupted bytes: Status {resp_e2.status_code} | Error: '{data_e2.get('error')}'")

        # Subtest 3: Empty file
        resp_e3 = await client.post("/api/v1/ai/detect", files={"file": ("empty.jpg", b"", "image/jpeg")})
        assert resp_e3.status_code == 400
        data_e3 = resp_e3.json()
        print(f"  -> Zero-byte file: Status {resp_e3.status_code} | Error: '{data_e3.get('error')}'")
        print("  -> Verdict: PASS (All invalid inputs rejected with clean HTTP 400 JSON errors)")
        results["Test E (Invalid Images)"] = "PASS"

    print("\n" + "=" * 70)
    print(" SUMMARY OF TEST RESULTS")
    print("=" * 70)
    for test_name, status in results.items():
        print(f"  {test_name:<30}: {status}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_scenario_tests())
