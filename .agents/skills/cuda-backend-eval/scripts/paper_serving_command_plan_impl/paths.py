"""Path helpers for paper serving command plans."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_SERVING = VIEWER_DATA / "serving_workloads.json"
DEFAULT_RUNS = VIEWER_DATA / "paper_baseline_runs.json"


def artifact_dir(root: str, baseline_id: str, policy_id: str) -> str:
    return f"{root.rstrip('/')}/{baseline_id}/{policy_id}"


def path_from_cwd(repo_relative_path: str, cwd: str) -> str:
    return os.path.relpath(ROOT / repo_relative_path, ROOT / cwd)
