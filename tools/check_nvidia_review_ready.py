#!/usr/bin/env python3
"""Compatibility entry point for the NVIDIA review readiness guard."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(
    str(ROOT / ".agents" / "checks" / "check_nvidia_review_ready.py"),
    run_name="__main__",
)
