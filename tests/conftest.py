"""
conftest.py — Shared pytest configuration and fixtures.

Adds every src/ subdirectory to sys.path once, at collection time, so
individual test files can import project modules (model, losses,
kinematics, inspector, etc.) directly by their bare module names,
matching how the modules themselves import each other.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC = PROJECT_ROOT / "src"

for subdir in ["vision", "digital_twin", "agents", "integration", "monitoring", "api"]:
    sys.path.insert(0, str(SRC / subdir))