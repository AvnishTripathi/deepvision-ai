"""
DeepVision AI — Comprehensive Environment & Hardware Inspection Script
"""

import sys
import os
import platform
import subprocess
import json
from pathlib import Path

# Ensure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 70)
print("🔍 DEEPVISION AI — ENVIRONMENT & HARDWARE DIAGNOSTIC REPORT")
print("=" * 70)

# 1. OPERATING SYSTEM
print("\n[1] OPERATING SYSTEM")
print(f"  OS Name:         {platform.system()}")
print(f"  OS Release:      {platform.release()} (Build {platform.version()})")
print(f"  Architecture:    {platform.machine()} ({platform.architecture()[0]})")
print(f"  Platform String: {platform.platform()}")

# 2. CPU & HARDWARE
print("\n[2] CPU SPECIFICATIONS")
try:
    cpu_info = subprocess.check_output(
        ["wmic", "cpu", "get", "name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed", "/format:csv"],
        text=True,
        stderr=subprocess.DEVNULL
    )
    lines = [line.strip() for line in cpu_info.strip().splitlines() if line.strip() and not line.startswith("Node")]
    if len(lines) >= 2:
        parts = lines[1].split(",")
        # Node,MaxClockSpeed,Name,NumberOfCores,NumberOfLogicalProcessors
        print(f"  Processor Name:   {parts[2] if len(parts) > 2 else platform.processor()}")
        print(f"  Physical Cores:   {parts[3] if len(parts) > 3 else 'Unknown'}")
        print(f"  Logical Threads:  {parts[4] if len(parts) > 4 else 'Unknown'}")
        print(f"  Max Clock Speed:  {parts[1] if len(parts) > 1 else 'Unknown'} MHz")
    else:
        print(f"  Processor Name:   {platform.processor()}")
except Exception:
    print(f"  Processor Name:   {platform.processor()}")

# 3. RAM (MEMORY)
print("\n[3] SYSTEM MEMORY (RAM)")
try:
    mem_info = subprocess.check_output(
        ["wmic", "OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory", "/format:csv"],
        text=True,
        stderr=subprocess.DEVNULL
    )
    lines = [line.strip() for line in mem_info.strip().splitlines() if line.strip() and not line.startswith("Node")]
    if len(lines) >= 2:
        parts = lines[1].split(",")
        # Node,FreePhysicalMemory,TotalVisibleMemorySize
        free_kb = float(parts[1])
        total_kb = float(parts[2])
        total_gb = round(total_kb / (1024 * 1024), 2)
        free_gb = round(free_kb / (1024 * 1024), 2)
        used_gb = round(total_gb - free_gb, 2)
        used_pct = round((used_gb / total_gb) * 100, 1)
        print(f"  Total Physical RAM:     {total_gb} GB")
        print(f"  Available (Free) RAM:   {free_gb} GB")
        print(f"  Used RAM:               {used_gb} GB ({used_pct}%)")
except Exception as e:
    print(f"  RAM query notice: {e}")

# 4. GPU & NVIDIA CUDA CHECK
print("\n[4] GPU & HARDWARE ACCELERATION")
try:
    gpu_info = subprocess.check_output(
        ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM,DriverVersion", "/format:csv"],
        text=True,
        stderr=subprocess.DEVNULL
    )
    gpu_lines = [l.strip() for l in gpu_info.strip().splitlines() if l.strip() and not l.startswith("Node")]
    for line in gpu_lines[1:]:
        p = line.split(",")
        if len(p) >= 4:
            adapter_ram = round(float(p[1]) / (1024 * 1024 * 1024), 2) if p[1].isdigit() else "Shared"
            print(f"  Detected GPU Adapter:   {p[3]} (Driver: {p[2]}, VRAM: {adapter_ram} GB)")
except Exception as e:
    print(f"  GPU query notice: {e}")

# nvidia-smi
print("\n  NVIDIA Tooling Diagnostic:")
try:
    res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if res.returncode == 0:
        print("  ✓ nvidia-smi: Available and responding")
        print(res.stdout)
    else:
        print("  ✗ nvidia-smi: Returned non-zero status code")
except FileNotFoundError:
    print("  ✗ nvidia-smi: Not found in system PATH (No discrete NVIDIA GPU detected)")

# 5. PYTHON & VIRTUAL ENVIRONMENT
print("\n[5] PYTHON RUNTIME & ENVIRONMENT")
print(f"  System Python:          {sys.version.split()[0]} ({sys.executable})")

venv_dir = Path(__file__).resolve().parent.parent / "backend" / "venv"
venv_python = venv_dir / "Scripts" / "python.exe"
print(f"  Target Venv Path:       {venv_dir}")
print(f"  Venv Python Exists:     {venv_python.is_file()}")
if venv_python.is_file():
    v_ver = subprocess.check_output([str(venv_python), "--version"], text=True).strip()
    print(f"  Venv Python Version:    {v_ver}")

    # Check CUDA access via Python
    print("\n  Python CUDA Access Check:")
    cuda_check_code = (
        "try:\n"
        "    import torch\n"
        "    print('    PyTorch CUDA Available:', torch.cuda.is_available())\n"
        "except ImportError:\n"
        "    print('    PyTorch: Not installed (no torch CUDA runtime)')\n"
        "try:\n"
        "    import onnxruntime as ort\n"
        "    print('    ONNX Runtime Providers:', ort.get_available_providers())\n"
        "except ImportError:\n"
        "    print('    ONNX Runtime: Not installed')\n"
    )
    cuda_res = subprocess.check_output([str(venv_python), "-c", cuda_check_code], text=True)
    print(cuda_res.strip())

# 6. NODE.JS & NPM
print("\n[6] NODE.JS & JAVASCRIPT TOOLING")
for tool in ["node", "npm"]:
    try:
        ver = subprocess.check_output([tool, "--version"], text=True).strip()
        print(f"  {tool.capitalize():8} Version: {ver}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"  {tool.capitalize():8} Version: NOT installed / not in PATH")

# 7. DOCKER STATUS
print("\n[7] DOCKER & CONTAINERIZATION STATUS")
for tool in ["docker", "docker-compose"]:
    try:
        d_ver = subprocess.check_output([tool, "--version"], text=True, stderr=subprocess.STDOUT).strip()
        print(f"  {tool:15} Version: {d_ver}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"  {tool:15} Status:  NOT installed / not in PATH")

# 8. BACKEND DEPENDENCIES
print("\n[8] INSTALLED BACKEND DEPENDENCIES")
if venv_python.is_file():
    pip_freeze = subprocess.check_output([str(venv_python), "-m", "pip", "list"], text=True).strip()
    for line in pip_freeze.splitlines()[:25]:
        print(f"  {line}")
    total_pkgs = len(pip_freeze.splitlines()) - 2
    print(f"  ... (Total: {total_pkgs} packages installed)")

# 9. FRONTEND DEPENDENCIES
print("\n[9] FRONTEND DEPENDENCIES (package.json)")
pkg_json = Path(__file__).resolve().parent.parent / "frontend" / "package.json"
if pkg_json.is_file():
    with open(pkg_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("  Production Dependencies:")
    for k, v in data.get("dependencies", {}).items():
        print(f"    - {k}: {v}")
    print("  Dev Dependencies:")
    for k, v in data.get("devDependencies", {}).items():
        print(f"    - {k}: {v}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE — NO HARDWARE OR LIBRARY ASSUMPTIONS MADE")
print("=" * 70)
