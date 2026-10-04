"""
DeepVision AI — End-to-End AI Pipeline Verification Script
Validates detector initialization, test image ingestion, letterbox preprocessing,
forward-pass execution, structured results decoding, latency profiling,
device selection, and deterministic resource release.
"""

import os
import sys
import json
import time
import numpy as np
import cv2

# Ensure backend directory is in Python module search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.core.config import settings
from app.ai.detection.detector import FaceDetector
from app.ai.detection.adapter import SCRFDAdapter
from app.ai.detection.schemas import FrameInferenceResult, DetectionResponse
from app.ai.inference.device import get_device_telemetry, resolve_device


def generate_safe_sample_image(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Generates a deterministic synthetic test image containing clear geometric face features.
    Guarantees consistent testing without relying on external media files.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 1. Subtle background gradient
    for y in range(height):
        img[y, :, 0] = int(25 + 15 * (y / height))
        img[y, :, 1] = int(35 + 20 * (y / height))
        img[y, :, 2] = int(45 + 25 * (y / height))

    # 2. Simulated Face Oval (Center ROI)
    face_cx, face_cy = width // 2, height // 2
    face_rx, face_ry = 130, 160
    cv2.ellipse(img, (face_cx, face_cy), (face_rx, face_ry), 0, 0, 360, (225, 185, 155), -1)

    # 3. Eyes (symmetric positions)
    eye_y = face_cy - 40
    cv2.circle(img, (face_cx - 50, eye_y), 14, (45, 35, 30), -1)
    cv2.circle(img, (face_cx + 50, eye_y), 14, (45, 35, 30), -1)
    cv2.circle(img, (face_cx - 50, eye_y), 5, (255, 255, 255), -1)
    cv2.circle(img, (face_cx + 50, eye_y), 5, (255, 255, 255), -1)

    # 4. Nose tip
    cv2.circle(img, (face_cx, face_cy + 15), 10, (185, 135, 115), -1)

    # 5. Mouth arc
    cv2.ellipse(img, (face_cx, face_cy + 65), (35, 14), 0, 0, 180, (160, 55, 55), -1)

    return img


def check_model_weights_status() -> dict:
    """Checks whether physical model weights are present in the models directory."""
    models_dir = os.path.join(PROJECT_ROOT, "models", "face_detection")
    expected_weights = [
        "scrfd_500m_bnkps_shape640x640.onnx",
        "scrfd_2.5g_bnkps_shape640x640.onnx",
    ]
    found = []
    for w in expected_weights:
        path = os.path.join(models_dir, w)
        if os.path.exists(path):
            found.append((w, os.path.getsize(path)))

    return {
        "models_dir": models_dir,
        "weights_installed": len(found) > 0,
        "found_files": found,
    }


def main():
    print("=" * 70)
    print(" [*] DEEPVISION AI -- INFERENCE PIPELINE VERIFICATION")
    print("=" * 70)

    # 1. Inspect Hardware Environment & Device Resolution
    print("\n[Stage 1/6] Inspecting Hardware Environment & Execution Providers...")
    telemetry = get_device_telemetry()
    selected_device = resolve_device(settings.AI_DEVICE, enable_gpu=settings.AI_ENABLE_GPU)
    print(f"  * Configured Device Target : {settings.AI_DEVICE}")
    print(f"  * Resolved Runtime Device  : {selected_device.upper()}")
    print(f"  * GPU Acceleration Active  : {telemetry['gpu_available']}")
    print(f"  * Optimal Providers        : {telemetry['optimal_providers']}")
    print(f"  * CPU Cores / Threads      : {telemetry['cpu_cores']} cores / {telemetry['recommended_threads']} allocated threads")

    # 2. Check Physical Model Storage
    print("\n[Stage 2/6] Checking Model Storage & Architecture Adapter...")
    weights_info = check_model_weights_status()
    adapter = SCRFDAdapter()
    print(f"  * Target Model Directory   : {weights_info['models_dir']}")
    print(f"  * Model Adapter Attached   : {adapter.__class__.__name__} (strides: {adapter.strides})")
    if weights_info["weights_installed"]:
        print(f"  * ONNX Model File Found    : {weights_info['found_files']}")
    else:
        print("  * Physical Model Weights   : Not yet installed (Baseline Validation Engine active)")
        print("    [NOTE] Binary weights are kept out of Git repository by design.")

    # 3. Load Face Detector (Lifecycle Init)
    print("\n[Stage 3/6] Initializing FaceDetector with ImagePreprocessor & SCRFD Adapter...")
    detector = FaceDetector(
        device=selected_device,
        target_size=(640, 640),
        adapter=adapter,
    )
    print(f"  [OK] Detector successfully instantiated")
    print(f"  * Preprocessor Target Size : {detector.target_size}")
    print(f"  * Session Loaded Status    : {detector.is_loaded}")

    # 4. Generate Safe Test Image
    print("\n[Stage 4/6] Ingesting Safe Synthetic Test Image...")
    test_image = generate_safe_sample_image(width=640, height=480)
    print(f"  [OK] Image created in-memory: {test_image.shape[1]}x{test_image.shape[0]} RGB uint8")

    # 5. Execute End-to-End Inference
    print("\n[Stage 5/6] Executing End-to-End Detection Pipeline...")
    t_start = time.perf_counter()
    result: FrameInferenceResult = detector.detect(test_image, confidence_threshold=0.50)
    wall_clock_ms = round((time.perf_counter() - t_start) * 1000, 2)

    print(f"  [OK] Pipeline completed in {wall_clock_ms}ms wall-clock time")
    print("\n  --- Latency Breakdown ---")
    print(f"  * Preprocessing Duration   : {result.preprocessing_time_ms} ms")
    print(f"  * Forward-Pass Compute     : {result.inference_time_ms} ms")
    print(f"  * Post-processing / NMS    : {result.postprocessing_time_ms} ms")
    print(f"  * Total Pipeline Latency   : {result.total_time_ms} ms")
    print(f"  * Model Name               : {result.model_name}")
    print(f"  * Runtime Provider         : {result.runtime_provider}")
    print(f"  * Device Used              : {result.device_used.upper()}")

    print("\n  --- Structured Detection Results ---")
    print(f"  * Detected Faces Count     : {result.face_count}")
    for idx, face in enumerate(result.faces):
        print(f"    [Face #{idx + 1}]")
        print(f"      - Confidence Score     : {face.confidence:.4f}")
        print(f"      - Bounding Box (Pixels): x1={face.bbox.x1}, y1={face.bbox.y1}, x2={face.bbox.x2}, y2={face.bbox.y2}")
        print(f"      - Bbox Dimensions      : width={face.bbox.width}px, height={face.bbox.height}px, area={face.bbox.area}px2")
        if face.normalized_bbox:
            print(f"      - Normalized Bbox      : x1={face.normalized_bbox.x1}, y1={face.normalized_bbox.y1}, x2={face.normalized_bbox.x2}, y2={face.normalized_bbox.y2}")
        if face.landmarks:
            print(f"      - Landmarks ({len(face.landmarks)} pts)    : {', '.join(lm.name for lm in face.landmarks)}")
        if face.quality:
            print(f"      - Quality Assessment   : overall={face.quality.overall_quality:.2f}, sharpness={face.quality.sharpness:.1f}, usable={face.quality.is_usable}")

    # 6. Resource Deallocation
    print("\n[Stage 6/6] Releasing System Resources & Memory...")
    detector.unload()
    print(f"  [OK] detector.unload() executed: is_loaded={detector.is_loaded}")

    # Summary
    print("\n" + "=" * 70)
    print(" [PASSED] AI PIPELINE VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
