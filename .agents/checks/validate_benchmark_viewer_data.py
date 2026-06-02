#!/usr/bin/env python3
"""Validate the CUDA benchmark viewer's review-facing data contract."""

from __future__ import annotations

from pathlib import Path
import sys

CHECKS_ROOT = Path(__file__).resolve().parent
if str(CHECKS_ROOT) not in sys.path:
    sys.path.insert(0, str(CHECKS_ROOT))

from benchmark_viewer_validation.orchestrator import (  # noqa: E402
    main,
    validate_viewer_data,
)
from benchmark_viewer_validation.baseline_probes import (  # noqa: E402,F401
    validate_paper_baseline_probes,
)
from benchmark_viewer_validation.baseline_runs import (  # noqa: E402,F401
    validate_paper_baseline_runs,
)


if __name__ == "__main__":
    main()
