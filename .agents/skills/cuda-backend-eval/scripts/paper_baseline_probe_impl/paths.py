from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_BASELINES = VIEWER_DATA / "paper_baselines.json"
DEFAULT_PROBES = VIEWER_DATA / "paper_baseline_probes.json"
