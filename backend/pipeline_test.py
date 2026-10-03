"""Quick pipeline test - run from backend/ with venv active"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=== DEEPFACEGUARD FULL ML PIPELINE TEST ===\n")

# 1. ModelManager + EfficientNet-B7
print("[1/4] Loading ModelManager + EfficientNet-B7 (downloads weights on first run)...")
t = time.time()
from ml_engine.model_manager import ModelManager
mm = ModelManager()
elapsed = time.time() - t
print(f"      PASS  {elapsed:.1f}s | device={mm.device} | version={mm.MODEL_VERSION}\n")

# 2. FaceDetector
print("[2/4] Loading FaceDetector (MTCNN)...")
t = time.time()
from ml_engine.face_detector import FaceDetector
fd = FaceDetector()
print(f"      PASS  {(time.time()-t):.1f}s\n")

# 3. DeepFakeClassifier
print("[3/4] Running DeepFakeClassifier on synthetic 224x224 face...")
import numpy as np
from PIL import Image
from ml_engine.deepfake_classifier import DeepFakeClassifier

clf = DeepFakeClassifier()
face = Image.fromarray(np.uint8(np.random.randint(100, 200, (224, 224, 3))))
t = time.time()
result = clf.classify_face(face)
ms = (time.time() - t) * 1000
print(f"      PASS  {ms:.0f}ms")
print(f"      Score:  {result.deepfake_score:.4f}")
print(f"      Label:  {result.label}")
freq = result.signals['frequency']
blur = result.signals['blur']
color = result.signals['color_consistency']
comp = result.signals['compression']
model_feat = result.signals['model_features']
print(f"      Signals: freq={freq:.3f} blur={blur:.3f} color={color:.3f} compression={comp:.3f} model={model_feat:.3f}\n")

# 4. Heatmap + Chart generation
print("[4/4] Generating heatmap + signal chart...")
from ml_engine.heatmap_generator import HeatmapGenerator
from ml_engine.face_detector import FaceDetection

hg = HeatmapGenerator()
test_img = Image.fromarray(np.uint8(np.random.randint(50, 200, (480, 640, 3))))
fake_face = FaceDetection(bbox=[100, 80, 300, 280], confidence=0.99, face_crop=face)
t = time.time()
heatmap_b64 = hg.generate_face_heatmap(test_img, [fake_face], [result])
chart_b64 = hg.generate_signal_chart(result.signals)
print(f"      PASS  {(time.time()-t)*1000:.0f}ms")
print(f"      Heatmap size: {len(heatmap_b64):,} chars (base64 PNG)")
print(f"      Chart size:   {len(chart_b64):,} chars (base64 PNG)")

print("\n==========================================")
print("  ALL TESTS PASSED!")
print("  DeepFaceGuard ML Engine is fully ready.")
print("==========================================\n")
print("Next: restart the backend and upload a real photo to test!")
