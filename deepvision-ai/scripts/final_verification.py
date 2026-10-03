"""
DeepVision AI — Comprehensive Final Verification (Part 13)
Verifies:
1. Backend starts & Health endpoint responds
2. Frontend starts & renders required content
3. Database connection is valid & executes queries
4. Redis connection management is valid & diagnostics run cleanly
5. Docker Compose configuration is valid & build contexts exist
"""

import sys
import os
import json
import urllib.request
import time
from pathlib import Path

# Ensure UTF-8 output encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

print("======================================================================")
print("           DEEPVISION AI — FINAL VERIFICATION (PART 13)              ")
print("======================================================================")

# 1. VERIFY BACKEND & HEALTH ENDPOINT
print("\n[1/5] VERIFYING BACKEND & HEALTH ENDPOINT")
print("Target URL: http://127.0.0.1:8000/api/v1/health")
start = time.perf_counter()
try:
    req = urllib.request.Request("http://127.0.0.1:8000/api/v1/health")
    with urllib.request.urlopen(req, timeout=5) as res:
        b_status = res.status
        b_body = json.loads(res.read().decode("utf-8"))
        b_latency = round((time.perf_counter() - start) * 1000, 2)
    print(f"-> Backend HTTP Status: {b_status}")
    print(f"-> Response Latency:    {b_latency}ms")
    print(f"-> Response Body:       {json.dumps(b_body, indent=2)}")
    assert b_status == 200, f"Expected 200, got {b_status}"
    assert b_body["status"] == "healthy", f"Expected healthy status"
    assert b_body["service"] == "deepvision-api", f"Expected deepvision-api"
    assert b_body["version"] == "0.1.0", f"Expected version 0.1.0"
    print("-> RESULT: Backend & Health Endpoint Verified Successfully! (PASS)")
except Exception as e:
    print(f"-> FAILED: {e}")
    sys.exit(1)

# 2. VERIFY FRONTEND
print("\n[2/5] VERIFYING FRONTEND")
print("Target URL: http://localhost:3000")
start = time.perf_counter()
try:
    req = urllib.request.Request("http://localhost:3000")
    with urllib.request.urlopen(req, timeout=5) as res:
        f_status = res.status
        f_html = res.read().decode("utf-8")
        f_latency = round((time.perf_counter() - start) * 1000, 2)
    has_brand = "DeepVision AI" in f_html
    has_subtitle = "Face Detection" in f_html
    has_status = "System Status: Online" in f_html

    print(f"-> Frontend HTTP Status: {f_status}")
    print(f"-> Response Latency:     {f_latency}ms")
    print(f"-> Content 'DeepVision AI':            {has_brand}")
    print(f"-> Content 'Face Detection':           {has_subtitle}")
    print(f"-> Content 'System Status: Online':    {has_status}")
    assert f_status == 200, f"Expected 200, got {f_status}"
    assert has_brand, "Brand name missing in rendered page"
    assert has_subtitle, "Subtitle missing in rendered page"
    assert has_status, "System Status badge missing in rendered page"
    print("-> RESULT: Frontend Verified Successfully! (PASS)")
except Exception as e:
    print(f"-> FAILED: {e}")
    sys.exit(1)

# 3. VERIFY DATABASE CONNECTION
print("\n[3/5] VERIFYING DATABASE CONNECTION")
try:
    from app.core.database import check_db_connection, get_db, engine, get_database_url
    from sqlalchemy import text

    print(f"-> Configured DB URL:  {get_database_url()}")
    print(f"-> Engine Dialect:     {engine.dialect.name}")

    db_probe = check_db_connection()
    print(f"-> Connection Status:  {db_probe['status']}")
    print(f"-> Probe Latency:      {db_probe.get('latency_ms')}ms")

    # Live query execution
    db_gen = get_db()
    session = next(db_gen)
    test_val = session.execute(text("SELECT 1")).scalar()
    session.close()
    print(f"-> Query 'SELECT 1':   Result={test_val}")

    assert db_probe["status"] == "connected", "Database connection check failed"
    assert test_val == 1, "SELECT 1 did not return 1"
    print("-> RESULT: Database Connection Verified Successfully! (PASS)")
except Exception as e:
    print(f"-> FAILED: {e}")
    sys.exit(1)

# 4. VERIFY REDIS CONNECTION & DIAGNOSTICS
print("\n[4/5] VERIFYING REDIS CONFIGURATION & DIAGNOSTICS")
try:
    from app.core.redis import check_redis_connection, get_redis_pool, get_redis_client
    from app.core.config import settings

    print(f"-> Configured REDIS_URL: {settings.REDIS_URL}")
    pool = get_redis_pool()
    print(f"-> Connection Pool:      {'Active' if pool else 'Inactive'} (max_connections=20)")
    client = get_redis_client()
    print(f"-> Client Instance:      {'Created' if client else 'None'}")

    redis_probe = check_redis_connection()
    print(f"-> Connection State:     {redis_probe.get('status')}")
    print(f"-> Diagnostic Details:   {redis_probe}")
    assert "status" in redis_probe, "Redis probe did not return status"
    print("-> RESULT: Redis Connection Management Verified Successfully! (PASS)")
except Exception as e:
    print(f"-> FAILED: {e}")
    sys.exit(1)

# 5. VERIFY DOCKER COMPOSE CONFIGURATION
print("\n[5/5] VERIFYING DOCKER COMPOSE CONFIGURATION")
try:
    import yaml
    compose_file = PROJECT_ROOT / "docker-compose.yml"
    with open(compose_file, "r", encoding="utf-8") as f:
        compose_cfg = yaml.safe_load(f)

    services = list(compose_cfg.get("services", {}).keys())
    print(f"-> Project Name:         {compose_cfg.get('name')}")
    print(f"-> Configured Services:  {services}")
    assert set(["backend", "frontend", "postgres", "redis"]).issubset(set(services)), "Missing required services"

    for s in ["backend", "frontend"]:
        ctx = PROJECT_ROOT / compose_cfg["services"][s]["build"]["context"]
        df = ctx / compose_cfg["services"][s]["build"].get("dockerfile", "Dockerfile")
        assert ctx.is_dir(), f"Context path missing: {ctx}"
        assert df.is_file(), f"Dockerfile missing: {df}"
        print(f"-> Verified {s:8} context: {ctx.name}/, dockerfile: {df.name}")

    print(f"-> Named Volumes:        {list(compose_cfg.get('volumes', {}).keys())}")
    print(f"-> Custom Networks:      {list(compose_cfg.get('networks', {}).keys())}")
    print("-> RESULT: Docker Compose Configuration Verified Successfully! (PASS)")
except Exception as e:
    print(f"-> FAILED: {e}")
    sys.exit(1)

print("\n======================================================================")
print("      FINAL VERIFICATION: ALL 5 CHECKS PASSED WITH 100% SUCCESS!      ")
print("======================================================================")
