"""
app.py — Convenient alias for main.py

Allows running:
    python app.py
or:
    uvicorn app:app --reload
"""
import sys
import os

# Ensure the backend directory and project root are in Python's path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import app

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  DeepVision AI Backend running on: http://localhost:8000")
    print("  Swagger API documentation:        http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
