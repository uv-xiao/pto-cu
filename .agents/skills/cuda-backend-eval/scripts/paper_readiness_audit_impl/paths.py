from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
VIEWER_DATA = ROOT / "evaluations" / "nvidia" / "benchmark-viewer" / "data"
DEFAULT_MATRIX = VIEWER_DATA / "paper_evaluation_matrix.json"
DEFAULT_RUNS = VIEWER_DATA / "paper_baseline_runs.json"
DEFAULT_PROBES = VIEWER_DATA / "paper_baseline_probes.json"
DEFAULT_RUN_READINESS = VIEWER_DATA / "paper_baseline_run_readiness.json"
DEFAULT_ATTEMPTS = VIEWER_DATA / "paper_baseline_execution_attempts.json"
DEFAULT_RESULTS = VIEWER_DATA / "results.json"
