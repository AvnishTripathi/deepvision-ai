"""
DeepVision AI — Development Face Detection Visualization Tool
Command-line utility for executing real face detection on an input image
and generating an annotated copy with bounding boxes, confidence scores,
and facial landmarks.

Usage:
    python scripts/visualize_detection.py --input path/to/input.jpg --output path/to/output_detected.jpg

Arguments:
    -i, --input         Path to input image file (JPEG, PNG, WEBP, etc.)
    -o, --output        Optional destination path for annotated image
                        (defaults to <input_stem>_detected.<ext>)
    -c, --confidence    Confidence threshold override (e.g. 0.50)
    -d, --device        Compute device override ('cpu' or 'gpu')
    --no-landmarks      Disable landmark rendering
    --no-confidence     Disable confidence text badge rendering

Development & Testing Tool:
    Does not persist or cache uploaded user data. Annotations are produced
    purely upon explicit developer CLI request.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add backend directory to Python sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.core.config import settings
from app.ai.detection.detector import FaceDetector
from app.ai.visualization.annotator import FaceVisualizer, annotate_image


def main():
    parser = argparse.ArgumentParser(
        description="DeepVision AI — Face Detection Visualization Development Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to input image file (e.g. input.jpg)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        type=str,
        help="Path to output annotated image (defaults to <input_stem>_detected.<ext>)",
    )
    parser.add_argument(
        "-c", "--confidence",
        default=None,
        type=float,
        help="Minimum confidence threshold (0.0 to 1.0, default from settings: 0.50)",
    )
    parser.add_argument(
        "-d", "--device",
        default=None,
        type=str,
        choices=["cpu", "gpu"],
        help="Inference compute device (default from settings: 'cpu')",
    )
    parser.add_argument(
        "--no-landmarks",
        action="store_true",
        help="Disable drawing 5-point facial landmarks",
    )
    parser.add_argument(
        "--no-confidence",
        action="store_true",
        help="Disable drawing confidence badges",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"[ERROR] Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = input_path.parent / f"{input_path.stem}_detected{input_path.suffix}"

    conf_thresh = args.confidence if args.confidence is not None else settings.AI_CONFIDENCE_THRESHOLD
    device = args.device or settings.AI_DEVICE

    print("=" * 65)
    print(" DeepVision AI — Development Visualization Tool")
    print("=" * 65)
    print(f"  Input Image:     {input_path}")
    print(f"  Output Image:    {output_path}")
    print(f"  Confidence Cut:  {conf_thresh:.2f}")
    print(f"  Device:          {device.upper()}")
    print("-" * 65)

    # Initialize FaceDetector
    try:
        model_path = settings.resolve_model_path()
        if not model_path or not os.path.exists(model_path):
            print(f"[ERROR] Detection model weights not found at: {model_path}", file=sys.stderr)
            sys.exit(1)

        print(f"  Loading model:   {os.path.basename(model_path)}...")
        detector = FaceDetector(
            model_path=model_path,
            device=device,
            confidence_threshold=conf_thresh,
        )
    except Exception as exc:
        print(f"[ERROR] Failed initializing detector: {exc}", file=sys.stderr)
        sys.exit(1)

    # Run detection & visualization
    try:
        t0 = time.perf_counter()
        annotated_img, result = annotate_image(
            input_path=input_path,
            output_path=output_path,
            detector=detector,
            confidence_threshold=conf_thresh,
            show_landmarks=not args.no_landmarks,
        )
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        print(f"  Image Size:      {result.image_width}x{result.image_height}")
        print(f"  Model Latency:   {result.processing_time_ms:.2f} ms")
        print(f"  Total Duration:  {total_time_ms:.2f} ms")
        print(f"  Faces Detected:  {result.face_count}")
        print("-" * 65)

        for i, face in enumerate(result.faces):
            box = face.bbox
            print(f"  Face #{i + 1}:")
            print(f"    - BBox:        [{box.x1:.1f}, {box.y1:.1f}, {box.x2:.1f}, {box.y2:.1f}] (size: {box.width:.1f}x{box.height:.1f})")
            print(f"    - Confidence:  {face.confidence * 100:.2f}%")
            if face.landmarks:
                lm_desc = ", ".join([f"{lm.name}: ({lm.x:.1f}, {lm.y:.1f})" for lm in face.landmarks])
                print(f"    - Landmarks:   {lm_desc}")

        print("=" * 65)
        print(f" [SUCCESS] Annotated image written to: {output_path.resolve()}")
        print("=" * 65)

    except Exception as exc:
        print(f"[ERROR] Processing failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
