"""Path helpers for paper baseline environment attempts."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_PLANS = VIEWER_DATA / "paper_baseline_environment_plans.json"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "tmp" / "cuda-backend" / "paper-baselines" / "environment-attempts"
)
DEFAULT_VIEWER_OUTPUT = VIEWER_DATA / "paper_baseline_environment_attempts.json"


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
