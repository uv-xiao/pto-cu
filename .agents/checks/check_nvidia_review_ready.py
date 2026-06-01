#!/usr/bin/env python3
"""Check review-facing CUDA backend docs, viewer data, and examples."""

from __future__ import annotations

from pathlib import Path
import sys

CHECKS_ROOT = Path(__file__).resolve().parent
if str(CHECKS_ROOT) not in sys.path:
    sys.path.insert(0, str(CHECKS_ROOT))

from nvidia_review_guard.orchestrator import main  # noqa: E402


if __name__ == "__main__":
    main()
