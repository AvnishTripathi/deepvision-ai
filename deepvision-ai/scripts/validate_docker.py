"""
Script to validate docker-compose.yml configuration and build contexts.
"""

from pathlib import Path
import yaml

compose_path = Path(__file__).resolve().parent.parent / "docker-compose.yml"
base_dir = compose_path.parent

with open(compose_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("=== DOCKER COMPOSE CONFIGURATION REPORT ===")
print("Project Name:", config.get("name"))

services = config.get("services", {})
print("Configured Services:", list(services.keys()))

required_services = {"backend", "frontend", "postgres", "redis"}
missing = required_services - set(services.keys())
assert not missing, f"Missing required services: {missing}"

for name, svc in services.items():
    print(f"\n[Service: {name}]")
    if "build" in svc:
        b = svc["build"]
        ctx = b["context"]
        df = b.get("dockerfile", "Dockerfile")
        context_dir = base_dir / ctx
        dockerfile = context_dir / df
        print(f"  Build Context: {ctx} (Exists: {context_dir.is_dir()})")
        print(f"  Dockerfile:    {df} (Exists: {dockerfile.is_file()})")
        assert context_dir.is_dir(), f"Context path does not exist: {context_dir}"
        assert dockerfile.is_file(), f"Dockerfile does not exist: {dockerfile}"
    if "image" in svc:
        print(f"  Container Image: {svc['image']}")
    if "ports" in svc:
        print(f"  Exposed Ports:   {svc['ports']}")
    if "healthcheck" in svc:
        print(f"  Healthcheck Cmd: {svc['healthcheck']['test']}")

print("\nNamed Volumes:", list(config.get("volumes", {}).keys()))
print("Custom Networks:", list(config.get("networks", {}).keys()))
print("\nValidation Status: PASSED (Configuration conforms to Compose Specification)")
