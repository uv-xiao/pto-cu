from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_BASELINES = VIEWER_DATA / "paper_baselines.json"
DEFAULT_RUNS = VIEWER_DATA / "paper_baseline_runs.json"
DEFAULT_PROBES = VIEWER_DATA / "paper_baseline_probes.json"
DEFAULT_ENV_PLANS = VIEWER_DATA / "paper_baseline_environment_plans.json"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "tmp" / "cuda-backend" / "paper-baselines" / "run-readiness"
)
DEFAULT_VIEWER_OUTPUT = VIEWER_DATA / "paper_baseline_run_readiness.json"
